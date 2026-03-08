use cat_detector::{camera, config, detector, notifier, service, storage, web};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{watch, RwLock};
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
    /// Run only the web dashboard (for testing)
    #[cfg(feature = "web")]
    Web {
        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,
        /// Address to bind to
        #[arg(short, long, default_value = "0.0.0.0")]
        bind: String,
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
        #[cfg(feature = "web")]
        Commands::Web { port, bind } => run_web_only(bind, port).await,
    }
}

async fn run_daemon(config_path: PathBuf) -> Result<()> {
    info!("Loading configuration from {:?}", config_path);

    let config = config::Config::load(&config_path)
        .with_context(|| format!("Failed to load config from {:?}", config_path))?;

    info!("Configuration loaded successfully");

    // Setup shutdown signal handler
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let shutdown_rx_web = shutdown_rx.clone();

    // Handle Ctrl+C
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Received Ctrl+C, initiating shutdown...");
        let _ = shutdown_tx.send(true);
    });

    // Setup web state and frame channel if web is enabled
    #[cfg(feature = "web")]
    let (web_state, frame_tx) = if config.web.enabled {
        let state = Arc::new(RwLock::new(web::WebAppState::new()));
        let (tx, rx) = web::create_frame_channel();

        // Start web server
        let web_config = config.web.clone();
        let web_state_clone = state.clone();
        tokio::spawn(async move {
            if let Err(e) = web::run_server(&web_config, web_state_clone, rx, shutdown_rx_web).await
            {
                error!("Web server error: {}", e);
            }
        });

        info!(
            "Web dashboard enabled at http://{}:{}",
            config.web.bind_address, config.web.port
        );
        (Some(state), Some(tx))
    } else {
        (None, None)
    };

    #[cfg(not(feature = "web"))]
    let (web_state, frame_tx): (
        Option<Arc<RwLock<web::WebAppState>>>,
        Option<web::FrameSender>,
    ) = (None, None);

    // Initialize camera
    #[cfg(feature = "real-camera")]
    let mut cam = camera::V4L2Camera::new(
        &config.camera.device_path,
        config.camera.frame_width,
        config.camera.frame_height,
        config.camera.fps,
    )
    .context("Failed to initialize camera")?;

    #[cfg(not(feature = "real-camera"))]
    let mut cam = camera::StubCamera::new(
        &config.camera.device_path,
        config.camera.frame_width,
        config.camera.frame_height,
        config.camera.fps,
    )
    .context("Failed to initialize camera")?;

    // Initialize detector
    let det = Arc::new(
        detector::OnnxDetector::new_with_size(
            &config.detector.model_path,
            config.detector.confidence_threshold,
            config.detector.cat_class_id,
            config.detector.input_size,
        )
        .context("Failed to initialize detector")?,
    );

    // Initialize storage
    let stor = Arc::new(
        storage::FileSystemStorage::new(
            config.storage.output_dir.clone(),
            config.storage.image_format.clone(),
            config.storage.jpeg_quality,
        )
        .context("Failed to initialize storage")?,
    );

    // Initialize notifier
    let notif = Arc::new(if config.notification.enabled {
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
        )?
    } else {
        notifier::SignalNotifier::disabled()
    });

    // Create tracker
    let mut tracker = cat_detector::tracker::CatTracker::new(
        config.tracking.enter_threshold,
        config.tracking.exit_threshold,
        config.tracking.sample_interval_secs,
    );

    // Mark as detecting in web state
    if let Some(ref state) = web_state {
        let mut s = state.write().await;
        s.detecting = true;
    }

    info!("Starting cat detection...");

    // Frame rate limiting for MJPEG stream
    let frame_interval =
        tokio::time::Duration::from_millis(1000 / config.web.stream_fps.max(1) as u64);
    let mut last_frame_sent = tokio::time::Instant::now() - frame_interval;

    // Exponential backoff for error recovery
    let base_interval = tokio::time::Duration::from_millis(config.tracking.detection_interval_ms);
    let max_backoff = tokio::time::Duration::from_secs(60);
    let mut consecutive_errors: u32 = 0;

    // Main detection loop with web integration
    let mut shutdown = shutdown_rx.clone();
    loop {
        // Check for shutdown
        if *shutdown.borrow() {
            info!("Shutdown signal received");
            break;
        }

        // Capture frame
        use camera::CameraCapture;
        use detector::CatDetector;
        let frame = match cam.capture_frame().await {
            Ok(f) => f,
            Err(e) => {
                consecutive_errors += 1;
                let backoff = base_interval * 2u32.saturating_pow(consecutive_errors.min(6));
                let backoff = backoff.min(max_backoff);
                tracing::warn!(
                    "Camera error (retry in {:?}, {} consecutive): {}",
                    backoff,
                    consecutive_errors,
                    e
                );
                tokio::time::sleep(backoff).await;
                continue;
            }
        };

        let timestamp = chrono::Utc::now();

        // Run detection
        let detections = match det.detect(&frame).await {
            Ok(d) => d,
            Err(e) => {
                consecutive_errors += 1;
                let backoff = base_interval * 2u32.saturating_pow(consecutive_errors.min(6));
                let backoff = backoff.min(max_backoff);
                tracing::warn!(
                    "Detector error (retry in {:?}, {} consecutive): {}",
                    backoff,
                    consecutive_errors,
                    e
                );
                tokio::time::sleep(backoff).await;
                continue;
            }
        };

        // Reset backoff on success
        consecutive_errors = 0;

        let cat_detected = detections.iter().any(|d| det.is_cat(d));

        // Update web state with frame and detections
        if let Some(ref state) = web_state {
            let mut s = state.write().await;
            s.update_frame(&frame, &detections);
            s.cat_present = cat_detected;
            if cat_detected {
                s.last_detection = Some(timestamp);
            }
        }

        // Send frame to MJPEG stream (throttled by stream_fps)
        if let Some(ref tx) = frame_tx {
            let now = tokio::time::Instant::now();
            if now.duration_since(last_frame_sent) >= frame_interval {
                if let Some(ref state) = web_state {
                    let s = state.read().await;
                    if let Some(ref frame_data) = s.latest_frame {
                        let _ = tx.send(Some(frame_data.clone()));
                        last_frame_sent = now;
                    }
                }
            }
        }

        // Process through tracker
        let events = tracker.process_detection(cat_detected, timestamp);

        // Handle events
        for event in events {
            use cat_detector::tracker::TrackerEvent;
            use notifier::Notifier;
            use storage::ImageStorage;

            match event {
                TrackerEvent::CatEntered { timestamp } => {
                    info!("Cat entered at {}", timestamp);

                    let saved = stor
                        .save_image(&frame, storage::ImageType::Entry, timestamp)
                        .await?;
                    info!("Saved entry image: {:?}", saved.path);

                    // Update web captures
                    if let Some(ref state) = web_state {
                        let mut s = state.write().await;
                        s.add_capture(web::CaptureInfo::new(
                            saved.path.clone(),
                            timestamp,
                            storage::ImageType::Entry,
                        ));
                    }

                    if notif.is_enabled() {
                        if let Err(e) = notif
                            .notify(notifier::NotificationEvent::CatEntered { timestamp })
                            .await
                        {
                            tracing::warn!("Failed to send entry notification: {}", e);
                        }
                    }
                }

                TrackerEvent::CatExited {
                    timestamp,
                    entry_time,
                } => {
                    let duration_secs = (timestamp - entry_time).num_seconds().max(0) as u64;
                    info!("Cat exited at {} (duration: {}s)", timestamp, duration_secs);

                    let saved = stor
                        .save_image(&frame, storage::ImageType::Exit, timestamp)
                        .await?;
                    info!("Saved exit image: {:?}", saved.path);

                    // Update web captures
                    if let Some(ref state) = web_state {
                        let mut s = state.write().await;
                        s.add_capture(web::CaptureInfo::new(
                            saved.path.clone(),
                            timestamp,
                            storage::ImageType::Exit,
                        ));
                    }

                    if notif.is_enabled() {
                        if let Err(e) = notif
                            .notify(notifier::NotificationEvent::CatExited {
                                timestamp,
                                duration_secs,
                            })
                            .await
                        {
                            tracing::warn!("Failed to send exit notification: {}", e);
                        }
                    }
                }

                TrackerEvent::SampleDue { timestamp } => {
                    tracing::debug!("Sample due at {}", timestamp);

                    let saved = stor
                        .save_image(&frame, storage::ImageType::Sample, timestamp)
                        .await?;
                    tracing::debug!("Saved sample image: {:?}", saved.path);

                    // Update web captures
                    if let Some(ref state) = web_state {
                        let mut s = state.write().await;
                        s.add_capture(web::CaptureInfo::new(
                            saved.path.clone(),
                            timestamp,
                            storage::ImageType::Sample,
                        ));
                    }
                }
            }
        }

        // Wait for next detection interval or shutdown
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(config.tracking.detection_interval_ms)) => {}
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("Shutdown signal received during sleep");
                    break;
                }
            }
        }
    }

    info!("Cat detector stopped");
    Ok(())
}

#[cfg(feature = "real-camera")]
async fn test_camera(device: String, output: Option<PathBuf>) -> Result<()> {
    use camera::CameraCapture;
    use detector::CatDetector;

    info!("Testing camera: {}", device);

    // Initialize camera
    let mut cam =
        camera::V4L2Camera::new(&device, 640, 480, 30).context("Failed to initialize camera")?;
    info!("Camera initialized");

    // Capture a frame
    info!("Capturing frame...");
    let frame = cam
        .capture_frame()
        .await
        .context("Failed to capture frame")?;
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

async fn test_image(
    image_path: PathBuf,
    model_path: PathBuf,
    threshold: f32,
    input_size: u32,
) -> Result<()> {
    use detector::CatDetector;

    info!("Testing detection on {:?}", image_path);
    info!("Using model: {:?}", model_path);
    info!("Confidence threshold: {}", threshold);
    info!("Input size: {}x{}", input_size, input_size);

    // Load the image
    let img = image::open(&image_path)
        .with_context(|| format!("Failed to open image: {:?}", image_path))?;
    info!("Loaded image: {}x{}", img.width(), img.height());

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
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
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

#[cfg(feature = "web")]
async fn run_web_only(bind: String, port: u16) -> Result<()> {
    info!("Starting web dashboard (standalone mode)");

    let web_config = config::WebConfig {
        enabled: true,
        bind_address: bind,
        port,
        stream_fps: 5,
    };

    let state = Arc::new(RwLock::new(web::WebAppState::new()));
    let (frame_tx, frame_rx) = web::create_frame_channel();

    // Setup shutdown signal handler
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Handle Ctrl+C
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Received Ctrl+C, initiating shutdown...");
        let _ = shutdown_tx.send(true);
    });

    // Generate test frames in a background task
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut counter = 0u32;
        loop {
            // Create a simple test frame
            let mut img = image::DynamicImage::new_rgb8(640, 480);
            if let image::DynamicImage::ImageRgb8(ref mut rgb) = img {
                // Draw a gradient to show something is happening
                for y in 0..480 {
                    for x in 0..640 {
                        let r = ((x + counter) % 256) as u8;
                        let g = ((y + counter) % 256) as u8;
                        let b = 128u8;
                        rgb.put_pixel(x, y, image::Rgb([r, g, b]));
                    }
                }
            }

            // Update state
            {
                let mut s = state_clone.write().await;
                s.update_frame(&img, &[]);
                if let Some(ref frame_data) = s.latest_frame {
                    let _ = frame_tx.send(Some(frame_data.clone()));
                }
            }

            counter = counter.wrapping_add(5);
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }
    });

    info!(
        "Web dashboard available at http://{}:{}",
        web_config.bind_address, web_config.port
    );

    web::run_server(&web_config, state, frame_rx, shutdown_rx)
        .await
        .map_err(|e| anyhow::anyhow!("Web server error: {}", e))?;

    Ok(())
}
