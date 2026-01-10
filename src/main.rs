use cat_detector::{app, camera, config, detector, notifier, service, storage};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tokio::sync::watch;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "cat-detector")]
#[command(about = "Webcam-based cat detection with notifications")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the cat detector daemon
    Run {
        /// Path to configuration file
        #[arg(short, long, default_value = "config.toml")]
        config: PathBuf,
    },
    /// Capture a single frame from camera and run detection
    #[cfg(feature = "real-camera")]
    TestCamera {
        /// Camera device path
        #[arg(short, long, default_value = "/dev/video0")]
        device: String,
        /// Save captured frame to this path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Test detection on an image file
    TestImage {
        /// Path to image file
        image: PathBuf,
        /// Path to ONNX model file
        #[arg(short, long, default_value = "models/yolox_s.onnx")]
        model: PathBuf,
        /// Confidence threshold (0.0 - 1.0)
        #[arg(short, long, default_value = "0.5")]
        threshold: f32,
        /// Model input size (416 for tiny, 640 for s/m/l)
        #[arg(short, long, default_value = "640")]
        size: u32,
    },
    /// Install as a systemd service
    InstallService {
        /// Path to configuration file (will be used by the service)
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Uninstall the systemd service
    UninstallService,
    /// Show service status
    Status,
    /// Start the systemd service
    Start {
        /// Path to configuration file
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Stop the systemd service
    Stop {
        /// Path to configuration file
        #[arg(short, long)]
        config: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run { config } => run_daemon(config).await,
        #[cfg(feature = "real-camera")]
        Commands::TestCamera { device, output } => test_camera(device, output).await,
        Commands::TestImage {
            image,
            model,
            threshold,
            size,
        } => test_image(image, model, threshold, size).await,
        Commands::InstallService { config } => install_service(config).await,
        Commands::UninstallService => uninstall_service().await,
        Commands::Status => show_status().await,
        Commands::Start { config } => start_service(config).await,
        Commands::Stop { config } => stop_service(config).await,
    }
}

async fn run_daemon(config_path: PathBuf) -> Result<()> {
    info!("Loading configuration from {:?}", config_path);

    let config = config::Config::load(&config_path)
        .with_context(|| format!("Failed to load config from {:?}", config_path))?;

    info!("Configuration loaded successfully");

    // Initialize camera
    #[cfg(feature = "real-camera")]
    let camera = camera::V4L2Camera::new(
        &config.camera.device_path,
        config.camera.frame_width,
        config.camera.frame_height,
        config.camera.fps,
    )
    .context("Failed to initialize camera")?;

    #[cfg(not(feature = "real-camera"))]
    let camera = camera::StubCamera::new(
        &config.camera.device_path,
        config.camera.frame_width,
        config.camera.frame_height,
        config.camera.fps,
    )
    .context("Failed to initialize camera")?;

    // Initialize detector
    let detector = detector::OnnxDetector::new_with_size(
        &config.detector.model_path,
        config.detector.confidence_threshold,
        config.detector.cat_class_id,
        config.detector.input_size,
    )
    .context("Failed to initialize detector")?;

    // Initialize storage
    let storage = storage::FileSystemStorage::new(
        config.storage.output_dir.clone(),
        config.storage.image_format.clone(),
        config.storage.jpeg_quality,
    )
    .context("Failed to initialize storage")?;

    // Initialize notifier
    let notifier = if config.notification.enabled {
        let signal_path = config
            .notification
            .signal_cli_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("/usr/local/bin/signal-cli"));
        let recipient = config
            .notification
            .recipient
            .clone()
            .expect("Recipient required when notifications enabled");

        notifier::SignalNotifier::new(
            signal_path,
            recipient,
            config.notification.notify_on_enter,
            config.notification.notify_on_exit,
        )
    } else {
        notifier::SignalNotifier::disabled()
    };

    // Create application
    let mut app = app::App::new(camera, detector, storage, notifier, &config.tracking);

    // Setup shutdown signal handler
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Handle Ctrl+C
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Received Ctrl+C, initiating shutdown...");
        let _ = shutdown_tx.send(true);
    });

    // Run the application
    info!("Starting cat detection...");
    app.run(shutdown_rx).await.map_err(|e| {
        error!("Application error: {}", e);
        anyhow::anyhow!("Application error: {}", e)
    })?;

    info!("Cat detector stopped");
    Ok(())
}

#[cfg(feature = "real-camera")]
async fn test_camera(device: String, output: Option<PathBuf>) -> Result<()> {
    use camera::CameraCapture;
    use detector::CatDetector;

    info!("Testing camera: {}", device);

    // Initialize camera
    let mut cam = camera::V4L2Camera::new(&device, 640, 480, 30)
        .context("Failed to initialize camera")?;
    info!("Camera initialized");

    // Capture a frame
    info!("Capturing frame...");
    let frame = cam.capture_frame().await.context("Failed to capture frame")?;
    info!("Captured frame: {}x{}", frame.width(), frame.height());

    // Save if output path specified
    if let Some(ref path) = output {
        frame.save(path).context("Failed to save frame")?;
        info!("Saved frame to {:?}", path);
    }

    // Run detection
    info!("Running detection...");
    let detector = detector::OnnxDetector::new_with_size(
        std::path::Path::new("models/yolox_s.onnx"),
        0.5,
        15,
        640,
    )
    .context("Failed to initialize detector")?;

    let start = std::time::Instant::now();
    let detections = detector.detect(&frame).await?;
    let elapsed = start.elapsed();
    info!("Detection completed in {:?}", elapsed);

    if detections.is_empty() {
        println!("\nNo objects detected");
    } else {
        println!("\nDetected {} object(s):", detections.len());
        for (i, det) in detections.iter().enumerate() {
            let class_name = coco_class_name(det.class_id);
            let is_cat = detector.is_cat(det);
            println!(
                "  {}. {} (class {}) - confidence: {:.1}%{}",
                i + 1,
                class_name,
                det.class_id,
                det.confidence * 100.0,
                if is_cat { " [CAT!]" } else { "" }
            );
        }

        let cat_count = detections.iter().filter(|d| detector.is_cat(d)).count();
        if cat_count > 0 {
            println!("\n==> {} cat(s) found!", cat_count);
        }
    }

    Ok(())
}

async fn test_image(image_path: PathBuf, model_path: PathBuf, threshold: f32, input_size: u32) -> Result<()> {
    use detector::CatDetector;

    info!("Testing detection on {:?}", image_path);
    info!("Using model: {:?}", model_path);
    info!("Confidence threshold: {}", threshold);
    info!("Input size: {}x{}", input_size, input_size);

    // Load the image
    let img = image::open(&image_path)
        .with_context(|| format!("Failed to open image: {:?}", image_path))?;
    info!(
        "Loaded image: {}x{}",
        img.width(),
        img.height()
    );

    // Initialize detector
    let detector = detector::OnnxDetector::new_with_size(&model_path, threshold, 15, input_size)
        .context("Failed to initialize detector")?;
    info!("Detector initialized");

    // Run detection
    info!("Running inference...");
    let start = std::time::Instant::now();
    let detections = detector.detect(&img).await?;
    let elapsed = start.elapsed();
    info!("Inference completed in {:?}", elapsed);

    // Report results
    if detections.is_empty() {
        println!("\nNo objects detected above threshold {}", threshold);
    } else {
        println!("\nDetected {} object(s):", detections.len());
        for (i, det) in detections.iter().enumerate() {
            let class_name = coco_class_name(det.class_id);
            let is_cat = detector.is_cat(det);
            println!(
                "  {}. {} (class {}) - confidence: {:.1}% - bbox: ({:.0}, {:.0}, {:.0}x{:.0}){}",
                i + 1,
                class_name,
                det.class_id,
                det.confidence * 100.0,
                det.bbox.x,
                det.bbox.y,
                det.bbox.width,
                det.bbox.height,
                if is_cat { " [CAT DETECTED!]" } else { "" }
            );
        }

        let cat_count = detections.iter().filter(|d| detector.is_cat(d)).count();
        if cat_count > 0 {
            println!("\n==> {} cat(s) found!", cat_count);
        } else {
            println!("\n==> No cats found");
        }
    }

    Ok(())
}

fn coco_class_name(class_id: u32) -> &'static str {
    const COCO_CLASSES: [&str; 80] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ];
    COCO_CLASSES.get(class_id as usize).unwrap_or(&"unknown")
}

async fn install_service(config_path: PathBuf) -> Result<()> {
    info!("Installing systemd service...");

    let config_path = config_path
        .canonicalize()
        .with_context(|| format!("Config file not found: {:?}", config_path))?;

    let service = service::SystemdService::new(config_path)?;
    service.install().await?;

    info!("Service installed successfully");
    info!("Start with: sudo systemctl start cat-detector");
    info!("View logs with: journalctl -u cat-detector -f");

    Ok(())
}

async fn uninstall_service() -> Result<()> {
    info!("Uninstalling systemd service...");

    let service = service::SystemdService::new(PathBuf::new())?;
    service.uninstall().await?;

    info!("Service uninstalled successfully");
    Ok(())
}

async fn show_status() -> Result<()> {
    let status = service::SystemdService::status()?;
    println!("{}", status);
    Ok(())
}

async fn start_service(config_path: PathBuf) -> Result<()> {
    let service = service::SystemdService::new(config_path)?;
    service.start().await?;
    info!("Service started");
    Ok(())
}

async fn stop_service(config_path: PathBuf) -> Result<()> {
    let service = service::SystemdService::new(config_path)?;
    service.stop().await?;
    info!("Service stopped");
    Ok(())
}
