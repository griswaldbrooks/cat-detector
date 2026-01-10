use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NotifierError {
    #[error("Failed to send notification: {0}")]
    SendError(String),
    #[error("Signal CLI not found: {0}")]
    SignalCliNotFound(String),
    #[allow(dead_code)]
    #[error("Notification disabled")]
    Disabled,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NotificationEvent {
    CatEntered { timestamp: DateTime<Utc> },
    CatExited { timestamp: DateTime<Utc>, duration_secs: u64 },
}

impl NotificationEvent {
    pub fn format_message(&self) -> String {
        match self {
            NotificationEvent::CatEntered { timestamp } => {
                format!(
                    "🐱 Cat detected! Entered at {}",
                    timestamp.format("%Y-%m-%d %H:%M:%S UTC")
                )
            }
            NotificationEvent::CatExited { timestamp, duration_secs } => {
                let duration = format_duration(*duration_secs);
                format!(
                    "🐱 Cat left at {}. Visit duration: {}",
                    timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
                    duration
                )
            }
        }
    }
}

fn format_duration(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

#[async_trait]
pub trait Notifier: Send + Sync {
    async fn notify(&self, event: NotificationEvent) -> Result<(), NotifierError>;
    fn is_enabled(&self) -> bool;
}

pub struct SignalNotifier {
    signal_cli_path: PathBuf,
    recipient: String,
    enabled: bool,
    notify_on_enter: bool,
    notify_on_exit: bool,
}

impl SignalNotifier {
    pub fn new(
        signal_cli_path: PathBuf,
        recipient: String,
        notify_on_enter: bool,
        notify_on_exit: bool,
    ) -> Self {
        Self {
            signal_cli_path,
            recipient,
            enabled: true,
            notify_on_enter,
            notify_on_exit,
        }
    }

    pub fn disabled() -> Self {
        Self {
            signal_cli_path: PathBuf::new(),
            recipient: String::new(),
            enabled: false,
            notify_on_enter: false,
            notify_on_exit: false,
        }
    }

    fn should_notify(&self, event: &NotificationEvent) -> bool {
        if !self.enabled {
            return false;
        }
        match event {
            NotificationEvent::CatEntered { .. } => self.notify_on_enter,
            NotificationEvent::CatExited { .. } => self.notify_on_exit,
        }
    }
}

#[async_trait]
impl Notifier for SignalNotifier {
    async fn notify(&self, event: NotificationEvent) -> Result<(), NotifierError> {
        if !self.should_notify(&event) {
            return Ok(());
        }

        if !self.signal_cli_path.exists() {
            return Err(NotifierError::SignalCliNotFound(
                format!("signal-cli not found at {:?}", self.signal_cli_path)
            ));
        }

        let message = event.format_message();

        let output = tokio::process::Command::new(&self.signal_cli_path)
            .args(["send", "-m", &message, &self.recipient])
            .output()
            .await
            .map_err(|e| NotifierError::SendError(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(NotifierError::SendError(format!(
                "signal-cli failed: {}",
                stderr
            )));
        }

        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Mock notifier for testing
#[cfg(test)]
pub struct MockNotifier {
    notifications: std::sync::Mutex<Vec<NotificationEvent>>,
    enabled: bool,
    should_fail: bool,
}

#[cfg(test)]
impl MockNotifier {
    pub fn new() -> Self {
        Self {
            notifications: std::sync::Mutex::new(Vec::new()),
            enabled: true,
            should_fail: false,
        }
    }

    pub fn disabled() -> Self {
        Self {
            notifications: std::sync::Mutex::new(Vec::new()),
            enabled: false,
            should_fail: false,
        }
    }

    pub fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }

    pub fn get_notifications(&self) -> Vec<NotificationEvent> {
        self.notifications.lock().unwrap().clone()
    }

    pub fn count(&self) -> usize {
        self.notifications.lock().unwrap().len()
    }

    pub fn get_enter_notifications(&self) -> Vec<NotificationEvent> {
        self.notifications
            .lock()
            .unwrap()
            .iter()
            .filter(|n| matches!(n, NotificationEvent::CatEntered { .. }))
            .cloned()
            .collect()
    }

    pub fn get_exit_notifications(&self) -> Vec<NotificationEvent> {
        self.notifications
            .lock()
            .unwrap()
            .iter()
            .filter(|n| matches!(n, NotificationEvent::CatExited { .. }))
            .cloned()
            .collect()
    }

    pub fn clear(&self) {
        self.notifications.lock().unwrap().clear();
    }
}

#[cfg(test)]
impl Default for MockNotifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[async_trait]
impl Notifier for MockNotifier {
    async fn notify(&self, event: NotificationEvent) -> Result<(), NotifierError> {
        if !self.enabled {
            return Err(NotifierError::Disabled);
        }

        if self.should_fail {
            return Err(NotifierError::SendError("Mock failure".to_string()));
        }

        self.notifications.lock().unwrap().push(event);
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_notifier_records_notifications() {
        let notifier = MockNotifier::new();
        let timestamp = Utc::now();

        notifier.notify(NotificationEvent::CatEntered { timestamp }).await.unwrap();
        assert_eq!(notifier.count(), 1);

        notifier.notify(NotificationEvent::CatExited {
            timestamp,
            duration_secs: 120
        }).await.unwrap();
        assert_eq!(notifier.count(), 2);
    }

    #[tokio::test]
    async fn test_mock_notifier_filters_by_type() {
        let notifier = MockNotifier::new();
        let timestamp = Utc::now();

        notifier.notify(NotificationEvent::CatEntered { timestamp }).await.unwrap();
        notifier.notify(NotificationEvent::CatEntered { timestamp }).await.unwrap();
        notifier.notify(NotificationEvent::CatExited {
            timestamp,
            duration_secs: 60
        }).await.unwrap();

        assert_eq!(notifier.get_enter_notifications().len(), 2);
        assert_eq!(notifier.get_exit_notifications().len(), 1);
    }

    #[tokio::test]
    async fn test_mock_notifier_failure() {
        let notifier = MockNotifier::new().with_failure();

        let result = notifier.notify(NotificationEvent::CatEntered {
            timestamp: Utc::now()
        }).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_disabled_notifier_returns_error() {
        let notifier = MockNotifier::disabled();

        let result = notifier.notify(NotificationEvent::CatEntered {
            timestamp: Utc::now()
        }).await;

        assert!(matches!(result.unwrap_err(), NotifierError::Disabled));
    }

    #[test]
    fn test_enter_message_format() {
        let timestamp = DateTime::parse_from_rfc3339("2024-01-15T10:30:45Z")
            .unwrap()
            .with_timezone(&Utc);

        let event = NotificationEvent::CatEntered { timestamp };
        let message = event.format_message();

        assert!(message.contains("Cat detected"));
        assert!(message.contains("2024-01-15 10:30:45 UTC"));
    }

    #[test]
    fn test_exit_message_format() {
        let timestamp = DateTime::parse_from_rfc3339("2024-01-15T10:30:45Z")
            .unwrap()
            .with_timezone(&Utc);

        let event = NotificationEvent::CatExited {
            timestamp,
            duration_secs: 3725  // 1h 2m 5s
        };
        let message = event.format_message();

        assert!(message.contains("Cat left"));
        assert!(message.contains("1h 2m 5s"));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(45), "45s");
        assert_eq!(format_duration(65), "1m 5s");
        assert_eq!(format_duration(3600), "1h 0m 0s");
        assert_eq!(format_duration(3665), "1h 1m 5s");
    }

    #[test]
    fn test_signal_notifier_should_notify() {
        let notifier = SignalNotifier::new(
            PathBuf::from("/usr/bin/signal-cli"),
            "+1234567890".to_string(),
            true,
            false,
        );

        assert!(notifier.should_notify(&NotificationEvent::CatEntered {
            timestamp: Utc::now()
        }));
        assert!(!notifier.should_notify(&NotificationEvent::CatExited {
            timestamp: Utc::now(),
            duration_secs: 60,
        }));
    }

    #[test]
    fn test_disabled_signal_notifier() {
        let notifier = SignalNotifier::disabled();
        assert!(!notifier.is_enabled());
    }
}
