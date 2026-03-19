use cat_detector::detector::CatDetector;
use cat_detector::watchdog::{StorageStatus, StorageWatchdog};
use cat_detector::{camera, config, detector, notifier, recorder, service, storage, web};

use anyhow::{Context, Result};
use camera::CameraCapture;
use clap::{Parser, Subcommand};
use notifier::Notifier;
use recorder::VideoRecorder;
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
        /// Path to CLIP ONNX model file
        #[arg(short, long, default_value = "models/clip_vitb32_image.onnx")]
        model: PathBuf,
        /// Confidence threshold (0.0 - 1.0 for softmax, ~-0.3 to 0.4 for catness)
        #[arg(short, long, default_value = "0.5")]
        threshold: f32,
        /// Path to catness direction file (enables catness direction mode)
        #[arg(long)]
        catness_direction: Option<PathBuf>,
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
    /// Test Signal notification setup (verify signal-cli and send test message)
    TestNotification {
        /// Path to configuration file
        #[arg(short, long, default_value = "config.toml")]
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
            catness_direction,
        } => test_image(image, model, threshold, catness_direction).await,
        Commands::TestNotification { config } => test_notification(config).await,
        Commands::InstallService { config } => install_service(config).await,
        Commands::UninstallService => uninstall_service().await,
        Commands::Status => show_status().await,
        Commands::Start { config } => start_service(config).await,
        Commands::Stop { config } => stop_service(config).await,
        #[cfg(feature = "web")]
        Commands::Web { port, bind } => run_web_only(bind, port).await,
    }
}

struct DaemonContext {
    stor: Arc<dyn storage::ImageStorage>,
    notif: Arc<notifier::SignalNotifier>,
    session_mgr: cat_detector::session::SessionManager,
    video_recorder: recorder::FfmpegRecorder,
    watchdog: StorageWatchdog,
    #[cfg(feature = "web")]
    web_state: Option<Arc<RwLock<web::WebAppState>>>,
}

impl DaemonContext {
    async fn handle_cat_entered(
        &mut self,
        frame: &image::DynamicImage,
        config: &config::Config,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        info!("Cat entered at {}", timestamp);

        // Start new session
        self.session_mgr.start_session(timestamp);

        // Start video recording (skip if storage is critical)
        if self.watchdog.is_recording_allowed() {
            if let Err(e) = self
                .video_recorder
                .start_recording(&config.storage.output_dir, timestamp)
            {
                tracing::warn!("Failed to start recording: {}", e);
            }
        } else {
            tracing::warn!("Skipping video recording — storage at critical threshold");
        }

        let saved = self
            .stor
            .save_image(frame, storage::ImageType::Entry, timestamp)
            .await?;
        info!("Saved entry image: {:?}", saved.path);

        self.session_mgr.set_entry_image(saved.path.clone());

        #[cfg(feature = "web")]
        if let Some(ref state) = self.web_state {
            let mut s = state.write().await;
            s.add_capture(web::CaptureInfo::new(
                saved.path.clone(),
                timestamp,
                storage::ImageType::Entry,
            ));
        }

        if self.notif.is_enabled() {
            if let Err(e) = self
                .notif
                .notify(notifier::NotificationEvent::CatEntered { timestamp })
                .await
            {
                tracing::warn!("Failed to send entry notification: {}", e);
            }
        }

        Ok(())
    }

    async fn handle_cat_exited(
        &mut self,
        frame: &image::DynamicImage,
        timestamp: chrono::DateTime<chrono::Utc>,
        entry_time: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        let duration_secs = (timestamp - entry_time).num_seconds().max(0) as u64;
        info!("Cat exited at {} (duration: {}s)", timestamp, duration_secs);

        // Stop video recording
        let video_path = match self.video_recorder.stop_recording() {
            Ok(Some(vp)) => {
                info!("Video saved to {:?}", vp);
                self.session_mgr.set_video_path(vp.clone());
                Some(vp)
            }
            Ok(None) => None,
            Err(e) => {
                tracing::warn!("Failed to stop recording: {}", e);
                None
            }
        };

        let saved = self
            .stor
            .save_image(frame, storage::ImageType::Exit, timestamp)
            .await?;
        info!("Saved exit image: {:?}", saved.path);

        self.session_mgr.set_exit_image(saved.path.clone());

        // End and persist session
        match self.session_mgr.end_session(timestamp) {
            Ok(Some(session)) => {
                info!(
                    "Session {} completed: {}s, {} samples",
                    session.id,
                    session.duration_secs().unwrap_or(0),
                    session.sample_images.len()
                );
            }
            Ok(None) => {}
            Err(e) => tracing::warn!("Failed to persist session: {}", e),
        }

        #[cfg(feature = "web")]
        if let Some(ref state) = self.web_state {
            let mut s = state.write().await;
            s.add_capture(web::CaptureInfo::new(
                saved.path.clone(),
                timestamp,
                storage::ImageType::Exit,
            ));
        }

        if self.notif.is_enabled() {
            if let Err(e) = self
                .notif
                .notify(notifier::NotificationEvent::CatExited {
                    timestamp,
                    duration_secs,
                    video_path,
                })
                .await
            {
                tracing::warn!("Failed to send exit notification: {}", e);
            }
        }

        Ok(())
    }

    async fn handle_sample_due(
        &mut self,
        frame: &image::DynamicImage,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        tracing::debug!("Sample due at {}", timestamp);

        let saved = self
            .stor
            .save_image(frame, storage::ImageType::Sample, timestamp)
            .await?;
        tracing::debug!("Saved sample image: {:?}", saved.path);

        self.session_mgr.add_sample_image(saved.path.clone());

        #[cfg(feature = "web")]
        if let Some(ref state) = self.web_state {
            let mut s = state.write().await;
            s.add_capture(web::CaptureInfo::new(
                saved.path.clone(),
                timestamp,
                storage::ImageType::Sample,
            ));
        }

        Ok(())
    }

    #[cfg(feature = "web")]
    async fn update_web_state(
        &self,
        frame: &image::DynamicImage,
        detections: &[cat_detector::detector::Detection],
        cat_detected: bool,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) {
        if let Some(ref state) = self.web_state {
            let mut s = state.write().await;
            s.update_frame(frame, detections);
            s.cat_present = cat_detected;
            if cat_detected {
                s.last_detection = Some(timestamp);
            }
        }
    }

    #[cfg(feature = "web")]
    async fn send_mjpeg_frame(
        &self,
        tx: &web::FrameSender,
        now: tokio::time::Instant,
        last_sent: &mut tokio::time::Instant,
        interval: tokio::time::Duration,
    ) {
        if now.duration_since(*last_sent) >= interval {
            if let Some(ref state) = self.web_state {
                let s = state.read().await;
                if let Some(ref frame_data) = s.latest_frame {
                    let _ = tx.send(Some(frame_data.clone()));
                    *last_sent = now;
                }
            }
        }
    }

    fn record_frame(&mut self, frame: &image::DynamicImage) {
        if self.video_recorder.is_recording() {
            if let Err(e) = self.video_recorder.add_frame(frame) {
                tracing::warn!("Failed to record frame: {}", e);
            }
        }
    }

    /// Runs detection and updates state. Returns `true` if detection succeeded.
    async fn run_detection(
        &self,
        det: &Arc<dyn detector::CatDetector>,
        frame: &image::DynamicImage,
        latest_detections: &mut Vec<cat_detector::detector::Detection>,
        cat_detected: &mut bool,
    ) -> bool {
        match det.detect(frame).await {
            Ok(detections) => {
                *cat_detected = detections.iter().any(|d| det.is_cat(d));
                *latest_detections = detections;
                true
            }
            Err(e) => {
                tracing::warn!("Detector error (will retry): {}", e);
                false
            }
        }
    }

    async fn dispatch_tracker_events(
        &mut self,
        tracker: &mut cat_detector::tracker::CatTracker,
        cat_detected: bool,
        frame: &image::DynamicImage,
        config: &config::Config,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        use cat_detector::tracker::TrackerEvent;

        let events = tracker.process_detection(cat_detected, timestamp);
        for event in events {
            match event {
                TrackerEvent::CatEntered { timestamp } => {
                    self.handle_cat_entered(frame, config, timestamp).await?;
                }
                TrackerEvent::CatExited {
                    timestamp,
                    entry_time,
                } => {
                    self.handle_cat_exited(frame, timestamp, entry_time).await?;
                }
                TrackerEvent::SampleDue { timestamp } => {
                    self.handle_sample_due(frame, timestamp).await?;
                }
            }
        }
        Ok(())
    }
}

/// Captures a frame, applying exponential backoff on errors.
/// Returns `Ok(Some(frame))` on success, `Ok(None)` if capture failed (should continue loop),
/// or the sleep future completes and the loop should retry.
async fn capture_with_backoff(
    cam: &mut impl CameraCapture,
    consecutive_errors: &mut u32,
    base_interval: tokio::time::Duration,
    max_backoff: tokio::time::Duration,
) -> Option<image::DynamicImage> {
    match cam.capture_frame().await {
        Ok(f) => {
            *consecutive_errors = 0;
            Some(f)
        }
        Err(e) => {
            *consecutive_errors += 1;
            let backoff = base_interval * 2u32.saturating_pow((*consecutive_errors).min(6));
            let backoff = backoff.min(max_backoff);
            tracing::warn!(
                "Camera error (retry in {:?}, {} consecutive): {}",
                backoff,
                consecutive_errors,
                e
            );
            tokio::time::sleep(backoff).await;
            None
        }
    }
}

fn init_notifier(config: &config::Config) -> Result<notifier::SignalNotifier> {
    if config.notification.enabled {
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

        let timezone = config
            .notification
            .timezone
            .as_deref()
            .map(notifier::parse_timezone)
            .transpose()
            .context("Invalid timezone in config")?;

        Ok(notifier::SignalNotifier::new(
            signal_path,
            recipient,
            config.notification.account.clone(),
            config.notification.notify_on_enter,
            config.notification.notify_on_exit,
            config.notification.send_video,
            std::time::Duration::from_secs(config.notification.attachment_timeout_secs),
            timezone,
        )?)
    } else {
        Ok(notifier::SignalNotifier::disabled())
    }
}

fn init_detector(config: &config::Config) -> Result<Arc<dyn detector::CatDetector>> {
    match config.detector.detection_mode {
        config::DetectionMode::Softmax => {
            let text_emb_path = config
                .detector
                .text_embeddings_path
                .clone()
                .unwrap_or_else(|| {
                    config
                        .detector
                        .model_path
                        .with_file_name("clip_text_embeddings.bin")
                });
            Ok(Arc::new(
                detector::ClipDetector::new(
                    &config.detector.model_path,
                    &text_emb_path,
                    config.detector.confidence_threshold,
                    config.detector.cat_class_id,
                )
                .context("Failed to initialize CLIP detector (softmax mode)")?,
            ))
        }
        config::DetectionMode::CatnessDirection => {
            let direction_path = config
                .detector
                .catness_direction_path
                .as_ref()
                .expect("catness_direction_path required (should be caught by validation)");
            Ok(Arc::new(
                detector::ClipDetector::new_catness_direction(
                    &config.detector.model_path,
                    direction_path,
                    config.detector.confidence_threshold,
                    config.detector.cat_class_id,
                )
                .context("Failed to initialize CLIP detector (catness direction mode)")?,
            ))
        }
    }
}

#[cfg(feature = "real-camera")]
fn init_camera(config: &config::Config) -> Result<camera::V4L2Camera> {
    camera::V4L2Camera::new(
        &config.camera.device_path,
        config.camera.frame_width,
        config.camera.frame_height,
        config.camera.fps,
    )
    .context("Failed to initialize camera")
}

#[cfg(not(feature = "real-camera"))]
fn init_camera(config: &config::Config) -> Result<camera::StubCamera> {
    camera::StubCamera::new(
        &config.camera.device_path,
        config.camera.frame_width,
        config.camera.frame_height,
        config.camera.fps,
    )
    .context("Failed to initialize camera")
}

#[cfg(feature = "web")]
fn setup_web(
    config: &config::Config,
    shutdown_rx: watch::Receiver<bool>,
) -> (
    Option<Arc<RwLock<web::WebAppState>>>,
    Option<web::FrameSender>,
) {
    if config.web.enabled {
        let mut web_app_state = web::WebAppState::with_dirs(
            config.storage.output_dir.join("sessions"),
            config.storage.output_dir.clone(),
        );
        web_app_state.system_info = Some(web::SystemInfoResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            model_name: config
                .detector
                .model_path
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            model_format: config.detector.model_format.clone(),
            confidence_threshold: config.detector.confidence_threshold,
            detection_interval_ms: config.tracking.detection_interval_ms,
            camera_resolution: format!(
                "{}x{}",
                config.camera.frame_width, config.camera.frame_height
            ),
        });
        web_app_state.config = Some(config.clone());
        web_app_state.storage_warn_threshold_bytes =
            (config.storage.warn_threshold_gb * 1_000_000_000.0) as u64;
        web_app_state.storage_critical_threshold_bytes =
            (config.storage.critical_threshold_gb * 1_000_000_000.0) as u64;
        let state = Arc::new(RwLock::new(web_app_state));
        let (tx, rx) = web::create_frame_channel();

        // Start web server
        let web_config = config.web.clone();
        let web_state_clone = state.clone();
        tokio::spawn(async move {
            if let Err(e) = web::run_server(&web_config, web_state_clone, rx, shutdown_rx).await {
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
    }
}

async fn run_daemon(config_path: PathBuf) -> Result<()> {
    info!("Loading configuration from {:?}", config_path);

    let config = config::Config::load(&config_path)
        .with_context(|| format!("Failed to load config from {:?}", config_path))?;

    info!("Configuration loaded successfully");

    // Setup shutdown signal handler
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Handle Ctrl+C
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Received Ctrl+C, initiating shutdown...");
        let _ = shutdown_tx.send(true);
    });

    // Setup web state and frame channel if web is enabled
    #[cfg(feature = "web")]
    let (web_state, frame_tx) = setup_web(&config, shutdown_rx.clone());

    // Initialize camera
    let mut cam = init_camera(&config)?;

    // Initialize detector
    let det = init_detector(&config)?;

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
    let notif = Arc::new(init_notifier(&config)?);

    // Create tracker
    let mut tracker = cat_detector::tracker::CatTracker::new(
        config.tracking.enter_threshold,
        config.tracking.exit_threshold,
        config.tracking.sample_interval_secs,
    );

    // Initialize session manager and video recorder
    let session_mgr =
        cat_detector::session::SessionManager::new(config.storage.output_dir.join("sessions"));
    let video_recorder = recorder::FfmpegRecorder::new(
        "ffmpeg",
        config.camera.frame_width,
        config.camera.frame_height,
        config.camera.fps,
    );

    // Initialize storage watchdog
    let watchdog = StorageWatchdog::new(
        config.storage.warn_threshold_gb,
        config.storage.critical_threshold_gb,
        config.storage.output_dir.clone(),
    );

    // Create daemon context
    let mut ctx = DaemonContext {
        stor: stor.clone(),
        notif: notif.clone(),
        session_mgr,
        video_recorder,
        watchdog,
        #[cfg(feature = "web")]
        web_state: web_state.clone(),
    };

    // Mark as detecting in web state
    #[cfg(feature = "web")]
    if let Some(ref state) = web_state {
        let mut s = state.write().await;
        s.detecting = true;
    }

    info!("Starting cat detection...");

    // Frame timing
    let camera_fps = config.camera.fps.max(1);
    let frame_interval_ms = 1000 / camera_fps as u64;
    let detection_interval =
        tokio::time::Duration::from_millis(config.tracking.detection_interval_ms);
    let mut last_detection_time = tokio::time::Instant::now() - detection_interval; // detect on first frame

    // MJPEG stream throttling
    #[cfg(feature = "web")]
    let stream_interval =
        tokio::time::Duration::from_millis(1000 / config.web.stream_fps.max(1) as u64);
    #[cfg(feature = "web")]
    let mut last_frame_sent = tokio::time::Instant::now() - stream_interval;

    // Exponential backoff for error recovery
    let base_interval = tokio::time::Duration::from_millis(frame_interval_ms);
    let max_backoff = tokio::time::Duration::from_secs(60);
    let mut consecutive_errors: u32 = 0;

    // Storage watchdog timing
    let watchdog_interval = tokio::time::Duration::from_secs(config.storage.watchdog_interval_secs);
    let mut last_watchdog_check = tokio::time::Instant::now() - watchdog_interval; // check on startup

    // Track latest detections for non-detection frames
    let mut latest_detections = Vec::new();
    let mut cat_detected = false;

    // Main frame loop — captures at camera fps, detects periodically
    let mut shutdown = shutdown_rx.clone();
    loop {
        // Check for shutdown
        if *shutdown.borrow() {
            info!("Shutdown signal received");
            break;
        }

        // Capture frame with exponential backoff on errors
        let Some(frame) = capture_with_backoff(
            &mut cam,
            &mut consecutive_errors,
            base_interval,
            max_backoff,
        )
        .await
        else {
            continue;
        };

        let timestamp = chrono::Utc::now();
        let now = tokio::time::Instant::now();

        // Run detection only when interval has elapsed
        let detection_due = now.duration_since(last_detection_time) >= detection_interval;
        if detection_due
            && ctx
                .run_detection(&det, &frame, &mut latest_detections, &mut cat_detected)
                .await
        {
            last_detection_time = now;
        }

        // Update web state with every frame (smoother MJPEG stream)
        #[cfg(feature = "web")]
        ctx.update_web_state(&frame, &latest_detections, cat_detected, timestamp)
            .await;

        // Send frame to MJPEG stream (throttled by stream_fps)
        #[cfg(feature = "web")]
        if let Some(ref tx) = frame_tx {
            ctx.send_mjpeg_frame(tx, now, &mut last_frame_sent, stream_interval)
                .await;
        }

        // Periodic storage watchdog check
        if now.duration_since(last_watchdog_check) >= watchdog_interval {
            last_watchdog_check = now;
            let status = ctx.watchdog.check();
            let used_bytes = match &status {
                StorageStatus::Ok { used_bytes } => *used_bytes,
                StorageStatus::Warning { used_bytes } => *used_bytes,
                StorageStatus::Critical { used_bytes } => *used_bytes,
            };
            match &status {
                StorageStatus::Ok { .. } => {
                    let gb = used_bytes as f64 / 1_000_000_000.0;
                    tracing::debug!("Storage watchdog: {:.2} GB used (ok)", gb);
                }
                StorageStatus::Warning { .. } => {
                    let gb = used_bytes as f64 / 1_000_000_000.0;
                    tracing::warn!(
                        "Storage watchdog: {:.2} GB used (warning threshold: {:.1} GB)",
                        gb,
                        config.storage.warn_threshold_gb
                    );
                }
                StorageStatus::Critical { .. } => {
                    let gb = used_bytes as f64 / 1_000_000_000.0;
                    tracing::error!(
                        "Storage watchdog: {:.2} GB used (CRITICAL — recording disabled, threshold: {:.1} GB)",
                        gb,
                        config.storage.critical_threshold_gb
                    );
                }
            }
            #[cfg(feature = "web")]
            if let Some(ref ws) = ctx.web_state {
                ws.write().await.storage_used_bytes = used_bytes;
            }
        }

        // Add every frame to video recorder if recording
        ctx.record_frame(&frame);

        // Process tracker and handle events only on detection frames
        if detection_due {
            ctx.dispatch_tracker_events(&mut tracker, cat_detected, &frame, &config, timestamp)
                .await?;
        }

        // Wait for next frame interval or shutdown
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(frame_interval_ms)) => {}
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

async fn test_notification(config_path: PathBuf) -> Result<()> {
    info!("Testing Signal notification setup...");

    let config = config::Config::load(&config_path)
        .with_context(|| format!("Failed to load config from {:?}", config_path))?;

    if !config.notification.enabled {
        anyhow::bail!(
            "Notifications are disabled in config. Set [notification] enabled = true first."
        );
    }

    let notif = init_notifier(&config)?;

    // Verify signal-cli is installed
    println!("Checking signal-cli...");
    match notif.verify_setup().await {
        Ok(version) => println!("signal-cli found: {}", version),
        Err(e) => {
            anyhow::bail!("signal-cli verification failed: {}", e);
        }
    }

    // Send a test message
    println!("Sending test notification...");
    notif
        .notify(notifier::NotificationEvent::CatEntered {
            timestamp: chrono::Utc::now(),
        })
        .await
        .context("Failed to send test notification")?;

    println!("Test notification sent successfully!");
    Ok(())
}

#[cfg(feature = "real-camera")]
async fn test_camera(device: String, output: Option<PathBuf>) -> Result<()> {
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
    let capture_model_path = std::path::Path::new("models/clip_vitb32_image.onnx");
    let capture_emb_path = std::path::Path::new("models/clip_text_embeddings.bin");
    let detector = detector::ClipDetector::new(capture_model_path, capture_emb_path, 0.5, 15)
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
    catness_direction: Option<PathBuf>,
) -> Result<()> {
    info!("Testing detection on {:?}", image_path);
    info!("Using model: {:?}", model_path);
    info!("Confidence threshold: {}", threshold);

    // Load the image
    let img = image::open(&image_path)
        .with_context(|| format!("Failed to open image: {:?}", image_path))?;
    info!("Loaded image: {}x{}", img.width(), img.height());

    // Initialize CLIP detector
    let det = if let Some(ref direction_path) = catness_direction {
        info!("Using catness direction mode: {:?}", direction_path);
        detector::ClipDetector::new_catness_direction(&model_path, direction_path, threshold, 15)
            .context("Failed to initialize CLIP detector (catness direction)")?
    } else {
        let text_emb_path = model_path.with_file_name("clip_text_embeddings.bin");
        detector::ClipDetector::new(&model_path, &text_emb_path, threshold, 15)
            .context("Failed to initialize CLIP detector")?
    };
    info!("Detector initialized");

    // Run detection
    info!("Running inference...");
    let start = std::time::Instant::now();
    let detections = det.detect(&img).await?;
    let elapsed = start.elapsed();
    info!("Inference completed in {:?}", elapsed);

    // Report results
    if detections.is_empty() {
        println!("\nNo objects detected above threshold {}", threshold);
    } else {
        println!("\nDetected {} object(s):", detections.len());
        for (i, d) in detections.iter().enumerate() {
            let class_name = coco_class_name(d.class_id);
            let is_cat = det.is_cat(d);
            println!(
                "  {}. {} (class {}) - confidence: {:.1}% - bbox: ({:.0}, {:.0}, {:.0}x{:.0}){}",
                i + 1,
                class_name,
                d.class_id,
                d.confidence * 100.0,
                d.bbox.x,
                d.bbox.y,
                d.bbox.width,
                d.bbox.height,
                if is_cat { " [CAT DETECTED!]" } else { "" }
            );
        }

        let cat_count = detections.iter().filter(|d| det.is_cat(d)).count();
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

    // Load config if available, for storage thresholds
    let standalone_config = config::Config::load(std::path::Path::new("config.toml")).ok();
    let mut web_state = web::WebAppState::new();
    web_state.system_info = Some(web::SystemInfoResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        model_name: "(standalone)".to_string(),
        model_format: "none".to_string(),
        confidence_threshold: 0.0,
        detection_interval_ms: 0,
        camera_resolution: "640x480".to_string(),
    });
    if let Some(ref cfg) = standalone_config {
        web_state.storage_warn_threshold_bytes =
            (cfg.storage.warn_threshold_gb * 1_000_000_000.0) as u64;
        web_state.storage_critical_threshold_bytes =
            (cfg.storage.critical_threshold_gb * 1_000_000_000.0) as u64;
        // Do an initial size check
        let mut watchdog = cat_detector::watchdog::StorageWatchdog::new(
            cfg.storage.warn_threshold_gb,
            cfg.storage.critical_threshold_gb,
            cfg.storage.output_dir.clone(),
        );
        if let cat_detector::watchdog::StorageStatus::Ok { used_bytes }
        | cat_detector::watchdog::StorageStatus::Warning { used_bytes }
        | cat_detector::watchdog::StorageStatus::Critical { used_bytes } = watchdog.check()
        {
            web_state.storage_used_bytes = used_bytes;
        }
    }
    let state = Arc::new(RwLock::new(web_state));
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
