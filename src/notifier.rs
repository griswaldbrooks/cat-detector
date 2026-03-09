use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NotifierError {
    #[error("Failed to send notification: {0}")]
    SendError(String),
    #[error("Signal CLI not found: {0}")]
    SignalCliNotFound(String),
    #[error("Notification timed out after {0} seconds")]
    Timeout(u64),
    #[error("Invalid recipient: {0}")]
    InvalidRecipient(String),
    #[allow(dead_code)]
    #[error("Notification disabled")]
    Disabled,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NotificationEvent {
    CatEntered {
        timestamp: DateTime<Utc>,
    },
    CatExited {
        timestamp: DateTime<Utc>,
        duration_secs: u64,
        video_path: Option<PathBuf>,
    },
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
            NotificationEvent::CatExited {
                timestamp,
                duration_secs,
                ..
            } => {
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

const SIGNAL_CLI_TIMEOUT: Duration = Duration::from_secs(30);

pub struct SignalNotifier {
    signal_cli_path: PathBuf,
    recipient: String,
    account: Option<String>,
    enabled: bool,
    notify_on_enter: bool,
    notify_on_exit: bool,
    timeout: Duration,
    send_video: bool,
    attachment_timeout: Duration,
}

/// Validate that a recipient looks like a phone number or signal group ID.
/// Accepts: +1234567890, +44.7911.123456, group IDs starting with "group."
fn validate_recipient(recipient: &str) -> Result<(), NotifierError> {
    let r = recipient.trim();
    if r.is_empty() {
        return Err(NotifierError::InvalidRecipient(
            "empty recipient".to_string(),
        ));
    }
    if let Some(rest) = r.strip_prefix('+') {
        // Phone number: must be digits (and optional dots/dashes) after the +
        if rest.is_empty()
            || !rest
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
        {
            return Err(NotifierError::InvalidRecipient(format!(
                "invalid phone number: {}",
                recipient
            )));
        }
        return Ok(());
    }
    if r.starts_with("group.") {
        return Ok(());
    }
    Err(NotifierError::InvalidRecipient(format!(
        "recipient must start with '+' (phone) or 'group.' (group ID): {}",
        recipient
    )))
}

impl SignalNotifier {
    pub fn new(
        signal_cli_path: PathBuf,
        recipient: String,
        account: Option<String>,
        notify_on_enter: bool,
        notify_on_exit: bool,
        send_video: bool,
        attachment_timeout: Duration,
    ) -> Result<Self, NotifierError> {
        validate_recipient(&recipient)?;
        Ok(Self {
            signal_cli_path,
            recipient,
            account,
            enabled: true,
            notify_on_enter,
            notify_on_exit,
            timeout: SIGNAL_CLI_TIMEOUT,
            send_video,
            attachment_timeout,
        })
    }

    pub fn disabled() -> Self {
        Self {
            signal_cli_path: PathBuf::new(),
            recipient: String::new(),
            account: None,
            enabled: false,
            notify_on_enter: false,
            notify_on_exit: false,
            timeout: SIGNAL_CLI_TIMEOUT,
            send_video: false,
            attachment_timeout: Duration::from_secs(120),
        }
    }

    /// Verify that signal-cli is installed and working. Returns the version string.
    pub async fn verify_setup(&self) -> Result<String, NotifierError> {
        let path = if self.signal_cli_path.as_os_str().is_empty() {
            PathBuf::from("/usr/local/bin/signal-cli")
        } else {
            self.signal_cli_path.clone()
        };

        if !path.exists() {
            return Err(NotifierError::SignalCliNotFound(format!(
                "signal-cli not found at {:?}",
                path
            )));
        }

        let output = tokio::process::Command::new(&path)
            .arg("--version")
            .output()
            .await
            .map_err(|e| NotifierError::SendError(format!("Failed to run signal-cli: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(NotifierError::SendError(format!(
                "signal-cli --version failed: {}",
                stderr
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Build the signal-cli send arguments and select the appropriate timeout.
    /// Returns (args, timeout).
    fn build_send_args(&self, event: &NotificationEvent) -> (Vec<String>, Duration) {
        let message = event.format_message();

        let video_attachment = if self.send_video {
            if let NotificationEvent::CatExited {
                video_path: Some(ref path),
                ..
            } = event
            {
                if path.exists() {
                    Some(path.clone())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let mut args = Vec::new();

        if let Some(ref account) = self.account {
            args.push("-a".to_string());
            args.push(account.clone());
        }

        args.extend([
            "send".to_string(),
            "-m".to_string(),
            message,
        ]);

        // Recipient must come before --attachment (signal-cli native parsing quirk)
        args.push(self.recipient.clone());

        if let Some(ref attachment_path) = video_attachment {
            args.push("--attachment".to_string());
            args.push(attachment_path.to_string_lossy().to_string());
        }

        let timeout = if video_attachment.is_some() {
            self.attachment_timeout
        } else {
            self.timeout
        };

        (args, timeout)
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
            return Err(NotifierError::SignalCliNotFound(format!(
                "signal-cli not found at {:?}",
                self.signal_cli_path
            )));
        }

        let (args, timeout) = self.build_send_args(&event);

        let output = tokio::time::timeout(
            timeout,
            tokio::process::Command::new(&self.signal_cli_path)
                .args(&args)
                .output(),
        )
            .await
            .map_err(|_| NotifierError::Timeout(timeout.as_secs()))?
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

        notifier
            .notify(NotificationEvent::CatEntered { timestamp })
            .await
            .unwrap();
        assert_eq!(notifier.count(), 1);

        notifier
            .notify(NotificationEvent::CatExited {
                timestamp,
                duration_secs: 120,
                video_path: None,
            })
            .await
            .unwrap();
        assert_eq!(notifier.count(), 2);
    }

    #[tokio::test]
    async fn test_mock_notifier_filters_by_type() {
        let notifier = MockNotifier::new();
        let timestamp = Utc::now();

        notifier
            .notify(NotificationEvent::CatEntered { timestamp })
            .await
            .unwrap();
        notifier
            .notify(NotificationEvent::CatEntered { timestamp })
            .await
            .unwrap();
        notifier
            .notify(NotificationEvent::CatExited {
                timestamp,
                duration_secs: 60,
                video_path: None,
            })
            .await
            .unwrap();

        assert_eq!(notifier.get_enter_notifications().len(), 2);
        assert_eq!(notifier.get_exit_notifications().len(), 1);
    }

    #[tokio::test]
    async fn test_mock_notifier_failure() {
        let notifier = MockNotifier::new().with_failure();

        let result = notifier
            .notify(NotificationEvent::CatEntered {
                timestamp: Utc::now(),
            })
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_disabled_notifier_returns_error() {
        let notifier = MockNotifier::disabled();

        let result = notifier
            .notify(NotificationEvent::CatEntered {
                timestamp: Utc::now(),
            })
            .await;

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
            duration_secs: 3725, // 1h 2m 5s
            video_path: None,
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
            None,
            true,
            false,
            true,
            Duration::from_secs(120),
        )
        .unwrap();

        assert!(notifier.should_notify(&NotificationEvent::CatEntered {
            timestamp: Utc::now()
        }));
        assert!(!notifier.should_notify(&NotificationEvent::CatExited {
            timestamp: Utc::now(),
            duration_secs: 60,
            video_path: None,
        }));
    }

    #[test]
    fn test_disabled_signal_notifier() {
        let notifier = SignalNotifier::disabled();
        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_validate_recipient_valid_phone() {
        assert!(validate_recipient("+1234567890").is_ok());
        assert!(validate_recipient("+44.7911.123456").is_ok());
        assert!(validate_recipient("+1-555-123-4567").is_ok());
    }

    #[test]
    fn test_validate_recipient_valid_group() {
        assert!(validate_recipient("group.abc123").is_ok());
    }

    #[test]
    fn test_validate_recipient_rejects_invalid() {
        assert!(validate_recipient("").is_err());
        assert!(validate_recipient("1234567890").is_err());
        assert!(validate_recipient("not-a-number").is_err());
        assert!(validate_recipient("+").is_err());
        assert!(validate_recipient("+abc").is_err());
    }

    #[test]
    fn test_build_send_args_enter_event_text_only() {
        let notifier = SignalNotifier::new(
            PathBuf::from("/usr/bin/signal-cli"),
            "+1234567890".to_string(),
            None,
            true,
            true,
            true,
            Duration::from_secs(120),
        )
        .unwrap();

        let event = NotificationEvent::CatEntered {
            timestamp: Utc::now(),
        };
        let (args, timeout) = notifier.build_send_args(&event);

        assert_eq!(args[0], "send");
        assert_eq!(args[1], "-m");
        // args[2] is the message text
        assert!(args[2].contains("Cat detected"));
        // No --attachment args
        assert!(!args.iter().any(|a| a == "--attachment"));
        // Last arg is recipient
        assert_eq!(args.last().unwrap(), "+1234567890");
        // Text-only timeout
        assert_eq!(timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_build_send_args_exit_with_video_includes_attachment() {
        let notifier = SignalNotifier::new(
            PathBuf::from("/usr/bin/signal-cli"),
            "+1234567890".to_string(),
            None,
            true,
            true,
            true,
            Duration::from_secs(120),
        )
        .unwrap();

        // Create a real temp file so path.exists() returns true
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let video_path = tmp.path().to_path_buf();

        let event = NotificationEvent::CatExited {
            timestamp: Utc::now(),
            duration_secs: 60,
            video_path: Some(video_path.clone()),
        };
        let (args, timeout) = notifier.build_send_args(&event);

        // Should have --attachment <path>
        let att_idx = args.iter().position(|a| a == "--attachment").unwrap();
        assert_eq!(args[att_idx + 1], video_path.to_string_lossy());
        // Recipient comes before --attachment (signal-cli native parsing quirk)
        let recip_idx = args.iter().position(|a| a == "+1234567890").unwrap();
        assert!(recip_idx < att_idx);
        // Attachment timeout
        assert_eq!(timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_build_send_args_send_video_false_skips_attachment() {
        let notifier = SignalNotifier::new(
            PathBuf::from("/usr/bin/signal-cli"),
            "+1234567890".to_string(),
            None,
            true,
            true,
            false, // send_video = false
            Duration::from_secs(120),
        )
        .unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();

        let event = NotificationEvent::CatExited {
            timestamp: Utc::now(),
            duration_secs: 60,
            video_path: Some(tmp.path().to_path_buf()),
        };
        let (args, timeout) = notifier.build_send_args(&event);

        assert!(!args.iter().any(|a| a == "--attachment"));
        assert_eq!(timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_build_send_args_with_account_includes_account_flag() {
        let notifier = SignalNotifier::new(
            PathBuf::from("/usr/bin/signal-cli"),
            "+1234567890".to_string(),
            Some("+9876543210".to_string()),
            true,
            true,
            true,
            Duration::from_secs(120),
        )
        .unwrap();

        let event = NotificationEvent::CatEntered {
            timestamp: Utc::now(),
        };
        let (args, _timeout) = notifier.build_send_args(&event);

        assert_eq!(args[0], "-a");
        assert_eq!(args[1], "+9876543210");
        assert_eq!(args[2], "send");
    }

    #[test]
    fn test_build_send_args_missing_video_falls_back_to_text() {
        let notifier = SignalNotifier::new(
            PathBuf::from("/usr/bin/signal-cli"),
            "+1234567890".to_string(),
            None,
            true,
            true,
            true,
            Duration::from_secs(120),
        )
        .unwrap();

        let event = NotificationEvent::CatExited {
            timestamp: Utc::now(),
            duration_secs: 60,
            video_path: Some(PathBuf::from("/nonexistent/video.mp4")),
        };
        let (args, timeout) = notifier.build_send_args(&event);

        assert!(!args.iter().any(|a| a == "--attachment"));
        assert_eq!(timeout, Duration::from_secs(30));
    }

    /// Integration test: sends a real Signal notification.
    /// Run with: SIGNAL_CLI_PATH=/path/to/signal-cli SIGNAL_RECIPIENT=+1234567890 \
    ///           cargo test --lib test_real_signal_cli_send -- --ignored
    #[tokio::test]
    #[ignore]
    async fn test_real_signal_cli_send() {
        let cli_path = std::env::var("SIGNAL_CLI_PATH")
            .expect("Set SIGNAL_CLI_PATH to run this test");
        let recipient = std::env::var("SIGNAL_RECIPIENT")
            .expect("Set SIGNAL_RECIPIENT to run this test");

        let notifier = SignalNotifier::new(
            PathBuf::from(&cli_path),
            recipient,
            None,
            true,
            true,
            true,
            Duration::from_secs(120),
        )
        .unwrap();

        // Verify setup
        let version = notifier.verify_setup().await.expect("signal-cli verify failed");
        println!("signal-cli version: {}", version);

        // Send a test notification
        let result = notifier
            .notify(NotificationEvent::CatEntered {
                timestamp: Utc::now(),
            })
            .await;

        assert!(result.is_ok(), "Failed to send: {:?}", result.err());
        println!("Test notification sent successfully");
    }

    #[test]
    fn test_signal_notifier_new_rejects_invalid_recipient() {
        let result = SignalNotifier::new(
            PathBuf::from("/usr/bin/signal-cli"),
            "not-valid".to_string(),
            None,
            true,
            true,
            true,
            Duration::from_secs(120),
        );
        assert!(matches!(result, Err(NotifierError::InvalidRecipient(_))));
    }
}
