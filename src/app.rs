use crate::camera::CameraCapture;
use crate::config::TrackingConfig;
use crate::detector::CatDetector;
use crate::notifier::{NotificationEvent, Notifier};
use crate::storage::{ImageStorage, ImageType};
use crate::tracker::{CatTracker, TrackerEvent};
use chrono::Utc;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Camera error: {0}")]
    Camera(#[from] crate::camera::CameraError),
    #[error("Detector error: {0}")]
    Detector(#[from] crate::detector::DetectorError),
    #[error("Storage error: {0}")]
    Storage(#[from] crate::storage::StorageError),
    #[error("Notification error: {0}")]
    Notification(#[from] crate::notifier::NotifierError),
    #[allow(dead_code)]
    #[error("Application shutdown")]
    Shutdown,
}

pub struct App<C, D, S, N>
where
    C: CameraCapture,
    D: CatDetector,
    S: ImageStorage,
    N: Notifier,
{
    camera: C,
    detector: Arc<D>,
    storage: Arc<S>,
    notifier: Arc<N>,
    tracker: CatTracker,
    detection_interval_ms: u64,
    frame_interval_ms: u64,
    last_detection_time: Option<tokio::time::Instant>,
}

impl<C, D, S, N> App<C, D, S, N>
where
    C: CameraCapture,
    D: CatDetector,
    S: ImageStorage,
    N: Notifier,
{
    pub fn new(
        camera: C,
        detector: D,
        storage: S,
        notifier: N,
        tracking_config: &TrackingConfig,
    ) -> Self {
        Self::new_with_fps(camera, detector, storage, notifier, tracking_config, 30)
    }

    pub fn new_with_fps(
        camera: C,
        detector: D,
        storage: S,
        notifier: N,
        tracking_config: &TrackingConfig,
        camera_fps: u32,
    ) -> Self {
        let tracker = CatTracker::new(
            tracking_config.enter_threshold,
            tracking_config.exit_threshold,
            tracking_config.sample_interval_secs,
        );

        Self {
            camera,
            detector: Arc::new(detector),
            storage: Arc::new(storage),
            notifier: Arc::new(notifier),
            tracker,
            detection_interval_ms: tracking_config.detection_interval_ms,
            frame_interval_ms: 1000 / camera_fps.max(1) as u64,
            last_detection_time: None,
        }
    }

    pub async fn run(&mut self, mut shutdown: watch::Receiver<bool>) -> Result<(), AppError> {
        info!("Starting cat detection loop");

        loop {
            // Check for shutdown signal
            if *shutdown.borrow() {
                info!("Shutdown signal received");
                return Ok(());
            }

            // Process one frame cycle
            if let Err(e) = self.process_cycle().await {
                match &e {
                    AppError::Camera(ce) => {
                        warn!("Camera error (will retry): {}", ce);
                    }
                    AppError::Detector(de) => {
                        warn!("Detector error (will retry): {}", de);
                    }
                    _ => {
                        error!("Error in detection cycle: {}", e);
                    }
                }
            }

            // Wait for next frame interval or shutdown
            tokio::select! {
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(self.frame_interval_ms)) => {}
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        info!("Shutdown signal received during sleep");
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Returns true if detection was run this cycle
    fn detection_due(&mut self) -> bool {
        let now = tokio::time::Instant::now();
        match self.last_detection_time {
            None => {
                self.last_detection_time = Some(now);
                true
            }
            Some(last) => {
                if now.duration_since(last).as_millis() as u64 >= self.detection_interval_ms {
                    self.last_detection_time = Some(now);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Process one frame cycle. Captures a frame every call,
    /// but only runs detection when detection_interval has elapsed.
    /// Returns the captured frame.
    pub async fn process_cycle(&mut self) -> Result<Option<image::DynamicImage>, AppError> {
        // Capture frame every cycle
        let frame = self.camera.capture_frame().await?;
        let timestamp = Utc::now();

        // Only run detection when interval has elapsed
        if self.detection_due() {
            let detections = self.detector.detect(&frame).await?;
            let cat_detected = detections.iter().any(|d| self.detector.is_cat(d));

            debug!("Detection result: cat_detected={}", cat_detected);

            // Process through tracker
            let events = self.tracker.process_detection(cat_detected, timestamp);

            // Handle events
            for event in events {
                self.handle_event(&event, &frame).await?;
            }
        }

        Ok(Some(frame))
    }

    async fn handle_event(
        &self,
        event: &TrackerEvent,
        frame: &image::DynamicImage,
    ) -> Result<(), AppError> {
        match event {
            TrackerEvent::CatEntered { timestamp } => {
                info!("Cat entered at {}", timestamp);

                // Save entry image
                let saved = self
                    .storage
                    .save_image(frame, ImageType::Entry, *timestamp)
                    .await?;
                info!("Saved entry image: {:?}", saved.path);

                // Send notification
                if self.notifier.is_enabled() {
                    if let Err(e) = self
                        .notifier
                        .notify(NotificationEvent::CatEntered {
                            timestamp: *timestamp,
                        })
                        .await
                    {
                        warn!("Failed to send entry notification: {}", e);
                    }
                }
            }

            TrackerEvent::CatExited {
                timestamp,
                entry_time,
            } => {
                let duration_secs = (*timestamp - *entry_time).num_seconds().max(0) as u64;
                info!("Cat exited at {} (duration: {}s)", timestamp, duration_secs);

                // Save exit image
                let saved = self
                    .storage
                    .save_image(frame, ImageType::Exit, *timestamp)
                    .await?;
                info!("Saved exit image: {:?}", saved.path);

                // Send notification
                if self.notifier.is_enabled() {
                    if let Err(e) = self
                        .notifier
                        .notify(NotificationEvent::CatExited {
                            timestamp: *timestamp,
                            duration_secs,
                        })
                        .await
                    {
                        warn!("Failed to send exit notification: {}", e);
                    }
                }
            }

            TrackerEvent::SampleDue { timestamp } => {
                debug!("Sample due at {}", timestamp);

                // Save sample image
                let saved = self
                    .storage
                    .save_image(frame, ImageType::Sample, *timestamp)
                    .await?;
                debug!("Saved sample image: {:?}", saved.path);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::MockCamera;
    use crate::config::TrackingConfig;
    use crate::detector::MockDetector;
    use crate::notifier::MockNotifier;
    use crate::storage::MockStorage;
    use image::DynamicImage;
    use std::sync::Arc;

    fn create_test_app(
        detector_sequence: Vec<bool>,
    ) -> (
        App<MockCamera, MockDetector, MockStorage, MockNotifier>,
        Arc<MockStorage>,
        Arc<MockNotifier>,
    ) {
        create_test_app_with_fps(detector_sequence, 10)
    }

    fn create_test_app_with_fps(
        detector_sequence: Vec<bool>,
        _camera_fps: u32,
    ) -> (
        App<MockCamera, MockDetector, MockStorage, MockNotifier>,
        Arc<MockStorage>,
        Arc<MockNotifier>,
    ) {
        let camera = MockCamera::from_solid_colors(vec![[128, 128, 128]], 640, 480);
        let detector = MockDetector::with_sequence(detector_sequence);
        let storage = Arc::new(MockStorage::new());
        let notifier = Arc::new(MockNotifier::new());

        // For existing behavior tests, use frame_interval = detection_interval
        // so detection runs every cycle (backwards compatible)
        let app = App {
            camera,
            detector: Arc::new(detector),
            storage: storage.clone(),
            notifier: notifier.clone(),
            tracker: CatTracker::new(2, 2, 10),
            detection_interval_ms: 100,
            frame_interval_ms: 100,
            last_detection_time: None,
        };

        (app, storage, notifier)
    }

    #[tokio::test]
    async fn test_cat_enter_saves_image_and_notifies() {
        let (mut app, storage, notifier) = create_test_app(vec![true, true]);

        // Process two cycles to trigger entry
        app.process_cycle().await.unwrap();
        // Wait for detection interval to elapse
        tokio::time::sleep(tokio::time::Duration::from_millis(110)).await;
        app.process_cycle().await.unwrap();

        // Verify storage
        assert_eq!(storage.get_images_by_type(ImageType::Entry).len(), 1);

        // Verify notification
        assert_eq!(notifier.get_enter_notifications().len(), 1);
    }

    #[tokio::test]
    async fn test_cat_exit_saves_image_and_notifies() {
        let (mut app, storage, notifier) = create_test_app(vec![true, true, false, false]);

        // Process four cycles: enter then exit
        for _ in 0..4 {
            tokio::time::sleep(tokio::time::Duration::from_millis(110)).await;
            app.process_cycle().await.unwrap();
        }

        // Verify storage
        assert_eq!(storage.get_images_by_type(ImageType::Entry).len(), 1);
        assert_eq!(storage.get_images_by_type(ImageType::Exit).len(), 1);

        // Verify notifications
        assert_eq!(notifier.get_enter_notifications().len(), 1);
        assert_eq!(notifier.get_exit_notifications().len(), 1);
    }

    #[tokio::test]
    async fn test_full_sequence_enter_stay_exit() {
        let (mut app, storage, _notifier) =
            create_test_app(vec![true, true, true, true, true, false, false]);

        // Process all cycles with detection interval waits
        for _ in 0..7 {
            tokio::time::sleep(tokio::time::Duration::from_millis(110)).await;
            app.process_cycle().await.unwrap();
        }

        // Verify we got entry and exit
        assert_eq!(storage.get_images_by_type(ImageType::Entry).len(), 1);
        assert_eq!(storage.get_images_by_type(ImageType::Exit).len(), 1);
    }

    #[tokio::test]
    async fn test_shutdown_signal_stops_app() {
        let (mut app, _storage, _notifier) = create_test_app(vec![true]);

        let (tx, rx) = watch::channel(false);

        // Send shutdown immediately
        tx.send(true).unwrap();

        // Run should return Ok immediately
        let result = app.run(rx).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_disabled_notifier_doesnt_fail() {
        let camera = MockCamera::from_solid_colors(vec![[128, 128, 128]], 640, 480);
        let detector = MockDetector::with_sequence(vec![true, true]);
        let storage = Arc::new(MockStorage::new());
        let notifier = Arc::new(MockNotifier::disabled());

        let mut app = App {
            camera,
            detector: Arc::new(detector),
            storage: storage.clone(),
            notifier: notifier.clone(),
            tracker: CatTracker::new(2, 2, 10),
            detection_interval_ms: 100,
            frame_interval_ms: 100,
            last_detection_time: None,
        };

        // Process two cycles with detection interval waits
        app.process_cycle().await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(110)).await;
        app.process_cycle().await.unwrap();

        // Storage should still work
        assert_eq!(storage.get_images_by_type(ImageType::Entry).len(), 1);
    }

    #[tokio::test]
    async fn test_decoupled_loop_captures_every_frame_detects_periodically() {
        // detection_interval=500ms, 5 rapid calls with no sleep between
        // Camera captures every time, detection only on first call
        let camera = MockCamera::from_solid_colors(vec![[128, 128, 128]], 640, 480);
        // Provide enough results for potential detections
        let detector = MockDetector::with_sequence(vec![true, true, true, true, true]);
        let storage = Arc::new(MockStorage::new());
        let notifier = Arc::new(MockNotifier::new());

        let detector_arc = Arc::new(detector);

        let mut app = App {
            camera,
            detector: detector_arc.clone(),
            storage: storage.clone(),
            notifier: notifier.clone(),
            tracker: CatTracker::new(2, 2, 10),
            detection_interval_ms: 500,
            frame_interval_ms: 33,
            last_detection_time: None,
        };

        // Call process_cycle 5 times rapidly (no sleep between)
        let mut frames_captured = 0;
        for _ in 0..5 {
            let result = app.process_cycle().await.unwrap();
            if result.is_some() {
                frames_captured += 1;
            }
        }

        // All 5 frames should be captured
        assert_eq!(frames_captured, 5);

        // But detector should only have been called once (first call is always due)
        assert_eq!(detector_arc.call_count(), 1);

        // Now wait for detection interval and call again
        tokio::time::sleep(tokio::time::Duration::from_millis(510)).await;
        app.process_cycle().await.unwrap();

        // Detector should now have been called twice
        assert_eq!(detector_arc.call_count(), 2);
    }
}
