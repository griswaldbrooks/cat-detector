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

            // Process one detection cycle
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

            // Wait for next detection interval or shutdown
            tokio::select! {
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(self.detection_interval_ms)) => {}
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        info!("Shutdown signal received during sleep");
                        return Ok(());
                    }
                }
            }
        }
    }

    async fn process_cycle(&mut self) -> Result<(), AppError> {
        // Capture frame
        let frame = self.camera.capture_frame().await?;
        let timestamp = Utc::now();

        // Run detection
        let detections = self.detector.detect(&frame).await?;
        let cat_detected = detections.iter().any(|d| self.detector.is_cat(d));

        debug!("Detection result: cat_detected={}", cat_detected);

        // Process through tracker
        let events = self.tracker.process_detection(cat_detected, timestamp);

        // Handle events
        for event in events {
            self.handle_event(&event, &frame).await?;
        }

        Ok(())
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
                info!(
                    "Cat exited at {} (duration: {}s)",
                    timestamp, duration_secs
                );

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
        let camera = MockCamera::from_solid_colors(vec![[128, 128, 128]], 640, 480);
        let detector = MockDetector::with_sequence(detector_sequence);
        let storage = MockStorage::new();
        let notifier = MockNotifier::new();

        let storage_arc = Arc::new(storage);
        let notifier_arc = Arc::new(notifier);

        let tracking_config = TrackingConfig {
            sample_interval_secs: 10,
            enter_threshold: 2,
            exit_threshold: 2,
            detection_interval_ms: 100,
        };

        // We need to create a new app that uses Arc for storage and notifier
        // For testing, we'll create a simpler version
        let app = App {
            camera,
            detector: Arc::new(detector),
            storage: storage_arc.clone(),
            notifier: notifier_arc.clone(),
            tracker: CatTracker::new(2, 2, 10),
            detection_interval_ms: 100,
        };

        (app, storage_arc, notifier_arc)
    }

    #[tokio::test]
    async fn test_cat_enter_saves_image_and_notifies() {
        let camera = MockCamera::from_solid_colors(vec![[128, 128, 128]], 640, 480);
        let detector = MockDetector::with_sequence(vec![true, true]); // 2 detections to enter
        let storage = Arc::new(MockStorage::new());
        let notifier = Arc::new(MockNotifier::new());

        let tracking_config = TrackingConfig {
            sample_interval_secs: 10,
            enter_threshold: 2,
            exit_threshold: 2,
            detection_interval_ms: 100,
        };

        let mut app = App {
            camera,
            detector: Arc::new(detector),
            storage: storage.clone(),
            notifier: notifier.clone(),
            tracker: CatTracker::new(2, 2, 10),
            detection_interval_ms: 100,
        };

        // Process two cycles to trigger entry
        app.process_cycle().await.unwrap();
        app.process_cycle().await.unwrap();

        // Verify storage
        assert_eq!(storage.get_images_by_type(ImageType::Entry).len(), 1);

        // Verify notification
        assert_eq!(notifier.get_enter_notifications().len(), 1);
    }

    #[tokio::test]
    async fn test_cat_exit_saves_image_and_notifies() {
        let camera = MockCamera::from_solid_colors(vec![[128, 128, 128]], 640, 480);
        // 2 detections (enter) + 2 non-detections (exit)
        let detector = MockDetector::with_sequence(vec![true, true, false, false]);
        let storage = Arc::new(MockStorage::new());
        let notifier = Arc::new(MockNotifier::new());

        let mut app = App {
            camera,
            detector: Arc::new(detector),
            storage: storage.clone(),
            notifier: notifier.clone(),
            tracker: CatTracker::new(2, 2, 10),
            detection_interval_ms: 100,
        };

        // Process four cycles: enter then exit
        for _ in 0..4 {
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
        let camera = MockCamera::from_solid_colors(vec![[128, 128, 128]], 640, 480);
        // Enter (2) + Stay (3) + Exit (2)
        let detector =
            MockDetector::with_sequence(vec![true, true, true, true, true, false, false]);
        let storage = Arc::new(MockStorage::new());
        let notifier = Arc::new(MockNotifier::new());

        let mut app = App {
            camera,
            detector: Arc::new(detector),
            storage: storage.clone(),
            notifier: notifier.clone(),
            tracker: CatTracker::new(2, 2, 1), // 1 second sample interval for testing
            detection_interval_ms: 100,
        };

        // Process all cycles
        for _ in 0..7 {
            app.process_cycle().await.unwrap();
            // Small delay to ensure timestamps differ enough for samples
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        // Verify we got entry and exit
        assert_eq!(storage.get_images_by_type(ImageType::Entry).len(), 1);
        assert_eq!(storage.get_images_by_type(ImageType::Exit).len(), 1);

        // Note: Sample timing depends on actual time elapsed, may or may not have samples
    }

    #[tokio::test]
    async fn test_shutdown_signal_stops_app() {
        let camera = MockCamera::from_solid_colors(vec![[128, 128, 128]], 640, 480);
        let detector = MockDetector::always_detect_cat();
        let storage = Arc::new(MockStorage::new());
        let notifier = Arc::new(MockNotifier::new());

        let mut app = App {
            camera,
            detector: Arc::new(detector),
            storage: storage.clone(),
            notifier: notifier.clone(),
            tracker: CatTracker::new(2, 2, 10),
            detection_interval_ms: 100,
        };

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
        };

        // Should not fail even with disabled notifier
        app.process_cycle().await.unwrap();
        app.process_cycle().await.unwrap();

        // Storage should still work
        assert_eq!(storage.get_images_by_type(ImageType::Entry).len(), 1);
    }
}
