use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ServiceError {
    #[error("Failed to create service file: {0}")]
    CreateError(String),
    #[error("Failed to enable service: {0}")]
    EnableError(String),
    #[error("Failed to start service: {0}")]
    StartError(String),
    #[error("Failed to stop service: {0}")]
    StopError(String),
    #[error("Failed to remove service: {0}")]
    RemoveError(String),
    #[error("Service operation requires root privileges")]
    PermissionDenied,
    #[error("systemctl not found")]
    SystemctlNotFound,
}

const SERVICE_NAME: &str = "cat-detector";

pub struct SystemdService {
    config_path: PathBuf,
    binary_path: PathBuf,
}

impl SystemdService {
    pub fn new(config_path: PathBuf) -> Result<Self, ServiceError> {
        let binary_path =
            std::env::current_exe().map_err(|e| ServiceError::CreateError(e.to_string()))?;

        Ok(Self {
            config_path,
            binary_path,
        })
    }

    fn service_file_path() -> PathBuf {
        PathBuf::from(format!("/etc/systemd/system/{}.service", SERVICE_NAME))
    }

    fn generate_service_file(&self) -> String {
        let working_dir = self
            .binary_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("/"));

        format!(
            r#"[Unit]
Description=Cat Detector - Webcam-based cat detection with notifications
After=network.target

[Service]
Type=simple
WorkingDirectory={working_dir}
ExecStart={binary} run --config {config}
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=RUST_LOG=info
Environment=ORT_DYLIB_PATH={working_dir}/onnxruntime/lib/libonnxruntime.so

[Install]
WantedBy=multi-user.target
"#,
            working_dir = working_dir.display(),
            binary = self.binary_path.display(),
            config = self.config_path.display()
        )
    }

    fn check_root() -> Result<(), ServiceError> {
        if !nix_check_root() {
            return Err(ServiceError::PermissionDenied);
        }
        Ok(())
    }

    fn check_systemctl() -> Result<(), ServiceError> {
        if !std::path::Path::new("/usr/bin/systemctl").exists()
            && !std::path::Path::new("/bin/systemctl").exists()
        {
            return Err(ServiceError::SystemctlNotFound);
        }
        Ok(())
    }

    pub async fn install(&self) -> Result<(), ServiceError> {
        Self::check_root()?;
        Self::check_systemctl()?;

        let service_content = self.generate_service_file();
        let service_path = Self::service_file_path();

        std::fs::write(&service_path, service_content)
            .map_err(|e| ServiceError::CreateError(e.to_string()))?;

        // Reload systemd
        let output = std::process::Command::new("systemctl")
            .args(["daemon-reload"])
            .output()
            .map_err(|e| ServiceError::CreateError(e.to_string()))?;

        if !output.status.success() {
            return Err(ServiceError::CreateError(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        // Enable service
        let output = std::process::Command::new("systemctl")
            .args(["enable", SERVICE_NAME])
            .output()
            .map_err(|e| ServiceError::EnableError(e.to_string()))?;

        if !output.status.success() {
            return Err(ServiceError::EnableError(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        Ok(())
    }

    pub async fn uninstall(&self) -> Result<(), ServiceError> {
        Self::check_root()?;
        Self::check_systemctl()?;

        // Stop service if running
        let _ = std::process::Command::new("systemctl")
            .args(["stop", SERVICE_NAME])
            .output();

        // Disable service
        let _ = std::process::Command::new("systemctl")
            .args(["disable", SERVICE_NAME])
            .output();

        // Remove service file
        let service_path = Self::service_file_path();
        if service_path.exists() {
            std::fs::remove_file(&service_path)
                .map_err(|e| ServiceError::RemoveError(e.to_string()))?;
        }

        // Reload systemd
        let output = std::process::Command::new("systemctl")
            .args(["daemon-reload"])
            .output()
            .map_err(|e| ServiceError::RemoveError(e.to_string()))?;

        if !output.status.success() {
            return Err(ServiceError::RemoveError(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        Ok(())
    }

    pub async fn start(&self) -> Result<(), ServiceError> {
        Self::check_root()?;
        Self::check_systemctl()?;

        let output = std::process::Command::new("systemctl")
            .args(["start", SERVICE_NAME])
            .output()
            .map_err(|e| ServiceError::StartError(e.to_string()))?;

        if !output.status.success() {
            return Err(ServiceError::StartError(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        Ok(())
    }

    pub async fn stop(&self) -> Result<(), ServiceError> {
        Self::check_root()?;
        Self::check_systemctl()?;

        let output = std::process::Command::new("systemctl")
            .args(["stop", SERVICE_NAME])
            .output()
            .map_err(|e| ServiceError::StopError(e.to_string()))?;

        if !output.status.success() {
            return Err(ServiceError::StopError(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        Ok(())
    }

    pub fn status() -> Result<String, ServiceError> {
        Self::check_systemctl()?;

        let output = std::process::Command::new("systemctl")
            .args(["status", SERVICE_NAME])
            .output()
            .map_err(|e| ServiceError::CreateError(e.to_string()))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

#[cfg(target_os = "linux")]
fn nix_check_root() -> bool {
    unsafe { libc::geteuid() == 0 }
}

#[cfg(not(target_os = "linux"))]
fn nix_check_root() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_file_generation() {
        let service = SystemdService {
            config_path: PathBuf::from("/etc/cat-detector/config.toml"),
            binary_path: PathBuf::from("/usr/local/bin/cat-detector"),
        };

        let content = service.generate_service_file();

        assert!(content.contains("[Unit]"));
        assert!(content.contains("[Service]"));
        assert!(content.contains("[Install]"));
        assert!(content.contains("/usr/local/bin/cat-detector"));
        assert!(content.contains("/etc/cat-detector/config.toml"));
        assert!(content.contains("Type=simple"));
        assert!(content.contains("Restart=on-failure"));
        assert!(content.contains("WorkingDirectory=/usr/local/bin"));
        assert!(content.contains("ORT_DYLIB_PATH=/usr/local/bin/onnxruntime/lib/libonnxruntime.so"));
    }

    #[test]
    fn test_service_file_path() {
        let path = SystemdService::service_file_path();
        assert_eq!(
            path,
            PathBuf::from("/etc/systemd/system/cat-detector.service")
        );
    }
}
