use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    ReadError(#[from] std::io::Error),
    #[error("Failed to parse config file: {0}")]
    ParseError(#[from] toml::de::Error),
    #[error("Invalid configuration: {0}")]
    ValidationError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub camera: CameraConfig,
    pub detector: DetectorConfig,
    pub storage: StorageConfig,
    pub notification: NotificationConfig,
    pub tracking: TrackingConfig,
    #[serde(default)]
    pub web: WebConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    #[serde(default = "default_device_path")]
    pub device_path: String,
    #[serde(default = "default_frame_width")]
    pub frame_width: u32,
    #[serde(default = "default_frame_height")]
    pub frame_height: u32,
    #[serde(default = "default_fps")]
    pub fps: u32,
}

fn default_device_path() -> String {
    "auto".to_string()
}

fn default_frame_width() -> u32 {
    640
}

fn default_frame_height() -> u32 {
    480
}

fn default_fps() -> u32 {
    30
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            device_path: default_device_path(),
            frame_width: default_frame_width(),
            frame_height: default_frame_height(),
            fps: default_fps(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    pub model_path: PathBuf,
    #[serde(default = "default_input_size")]
    pub input_size: u32,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default = "default_cat_class_id")]
    pub cat_class_id: u32,
    /// Model format: "auto" (detect from filename), "yolox", "yolo11", or "clip"
    #[serde(default = "default_model_format")]
    pub model_format: String,
    /// Path to CLIP text embeddings file (required when model_format = "clip")
    pub text_embeddings_path: Option<PathBuf>,
}

fn default_input_size() -> u32 {
    640
}

fn default_model_format() -> String {
    "auto".to_string()
}

fn default_confidence_threshold() -> f32 {
    0.5
}

fn default_cat_class_id() -> u32 {
    15 // COCO dataset cat class ID
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/yolo11n.onnx"),
            input_size: default_input_size(),
            confidence_threshold: default_confidence_threshold(),
            cat_class_id: default_cat_class_id(),
            model_format: default_model_format(),
            text_embeddings_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub output_dir: PathBuf,
    #[serde(default = "default_image_format")]
    pub image_format: String,
    #[serde(default = "default_jpeg_quality")]
    pub jpeg_quality: u8,
}

fn default_image_format() -> String {
    "jpg".to_string()
}

fn default_jpeg_quality() -> u8 {
    85
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("captures"),
            image_format: default_image_format(),
            jpeg_quality: default_jpeg_quality(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    #[serde(default)]
    pub enabled: bool,
    pub signal_cli_path: Option<PathBuf>,
    pub recipient: Option<String>,
    #[serde(default)]
    pub notify_on_enter: bool,
    #[serde(default)]
    pub notify_on_exit: bool,
    #[serde(default = "default_send_video")]
    pub send_video: bool,
    #[serde(default = "default_attachment_timeout_secs")]
    pub attachment_timeout_secs: u64,
}

fn default_send_video() -> bool {
    true
}

fn default_attachment_timeout_secs() -> u64 {
    120
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            signal_cli_path: None,
            recipient: None,
            notify_on_enter: true,
            notify_on_exit: true,
            send_video: default_send_video(),
            attachment_timeout_secs: default_attachment_timeout_secs(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingConfig {
    #[serde(default = "default_sample_interval_secs")]
    pub sample_interval_secs: u64,
    #[serde(default = "default_enter_threshold")]
    pub enter_threshold: u32,
    #[serde(default = "default_exit_threshold")]
    pub exit_threshold: u32,
    #[serde(default = "default_detection_interval_ms")]
    pub detection_interval_ms: u64,
}

fn default_sample_interval_secs() -> u64 {
    10
}

fn default_enter_threshold() -> u32 {
    3 // Require 3 consecutive detections to confirm entry
}

fn default_exit_threshold() -> u32 {
    5 // Require 5 consecutive non-detections to confirm exit
}

fn default_detection_interval_ms() -> u64 {
    500 // Check every 500ms
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            sample_interval_secs: default_sample_interval_secs(),
            enter_threshold: default_enter_threshold(),
            exit_threshold: default_exit_threshold(),
            detection_interval_ms: default_detection_interval_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_bind_address")]
    pub bind_address: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_stream_fps")]
    pub stream_fps: u32,
}

fn default_bind_address() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_stream_fps() -> u32 {
    5 // 5 fps for stream to reduce bandwidth
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bind_address: default_bind_address(),
            port: default_port(),
            stream_fps: default_stream_fps(),
        }
    }
}

impl Config {
    pub fn load(path: &std::path::Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.detector.confidence_threshold < 0.0 || self.detector.confidence_threshold > 1.0 {
            return Err(ConfigError::ValidationError(
                "confidence_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.notification.enabled && self.notification.recipient.is_none() {
            return Err(ConfigError::ValidationError(
                "recipient is required when notifications are enabled".to_string(),
            ));
        }

        if self.tracking.sample_interval_secs == 0 {
            return Err(ConfigError::ValidationError(
                "sample_interval_secs must be greater than 0".to_string(),
            ));
        }

        if self.web.enabled && self.web.port == 0 {
            return Err(ConfigError::ValidationError(
                "web port must be greater than 0".to_string(),
            ));
        }

        if self.web.stream_fps == 0 || self.web.stream_fps > 30 {
            return Err(ConfigError::ValidationError(
                "stream_fps must be between 1 and 30".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_valid_config_parses() {
        let config_content = r#"
[camera]
device_path = "/dev/video1"
frame_width = 1280
frame_height = 720
fps = 15

[detector]
model_path = "models/yolov8n.onnx"
confidence_threshold = 0.7
cat_class_id = 15

[storage]
output_dir = "my_captures"
image_format = "png"
jpeg_quality = 90

[notification]
enabled = false

[tracking]
sample_interval_secs = 5
enter_threshold = 2
exit_threshold = 3
detection_interval_ms = 250
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_content.as_bytes()).unwrap();

        let config = Config::load(temp_file.path()).unwrap();

        assert_eq!(config.camera.device_path, "/dev/video1");
        assert_eq!(config.camera.frame_width, 1280);
        assert_eq!(config.camera.frame_height, 720);
        assert_eq!(config.camera.fps, 15);
        assert_eq!(config.detector.confidence_threshold, 0.7);
        assert_eq!(config.storage.output_dir, PathBuf::from("my_captures"));
        assert_eq!(config.tracking.sample_interval_secs, 5);
    }

    #[test]
    fn test_defaults_applied_for_missing_optional_fields() {
        let config_content = r#"
[camera]

[detector]
model_path = "models/yolov8n.onnx"

[storage]
output_dir = "captures"

[notification]
enabled = false

[tracking]
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_content.as_bytes()).unwrap();

        let config = Config::load(temp_file.path()).unwrap();

        // Check defaults are applied
        assert_eq!(config.camera.device_path, "auto");
        assert_eq!(config.camera.frame_width, 640);
        assert_eq!(config.camera.frame_height, 480);
        assert_eq!(config.camera.fps, 30);
        assert_eq!(config.detector.confidence_threshold, 0.5);
        assert_eq!(config.detector.cat_class_id, 15);
        assert_eq!(config.storage.image_format, "jpg");
        assert_eq!(config.storage.jpeg_quality, 85);
        assert_eq!(config.tracking.sample_interval_secs, 10);
        assert_eq!(config.tracking.enter_threshold, 3);
        assert_eq!(config.tracking.exit_threshold, 5);
    }

    #[test]
    fn test_invalid_confidence_threshold_rejected() {
        let config_content = r#"
[camera]

[detector]
model_path = "models/yolov8n.onnx"
confidence_threshold = 1.5

[storage]
output_dir = "captures"

[notification]
enabled = false

[tracking]
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_content.as_bytes()).unwrap();

        let result = Config::load(temp_file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ConfigError::ValidationError(_)));
    }

    #[test]
    fn test_notification_requires_recipient_when_enabled() {
        let config_content = r#"
[camera]

[detector]
model_path = "models/yolov8n.onnx"

[storage]
output_dir = "captures"

[notification]
enabled = true

[tracking]
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_content.as_bytes()).unwrap();

        let result = Config::load(temp_file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ConfigError::ValidationError(_)));
    }

    #[test]
    fn test_invalid_toml_returns_parse_error() {
        let config_content = r#"
[camera
device_path = missing bracket
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_content.as_bytes()).unwrap();

        let result = Config::load(temp_file.path());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ConfigError::ParseError(_)));
    }

    #[test]
    fn test_missing_file_returns_read_error() {
        let result = Config::load(std::path::Path::new("/nonexistent/config.toml"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ConfigError::ReadError(_)));
    }

    #[test]
    fn test_zero_sample_interval_rejected() {
        let config_content = r#"
[camera]

[detector]
model_path = "models/yolov8n.onnx"

[storage]
output_dir = "captures"

[notification]
enabled = false

[tracking]
sample_interval_secs = 0
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_content.as_bytes()).unwrap();

        let result = Config::load(temp_file.path());
        assert!(result.is_err());
    }
}
