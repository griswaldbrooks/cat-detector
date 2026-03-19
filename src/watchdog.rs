use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WatchdogError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub enum StorageStatus {
    Ok { used_bytes: u64 },
    Warning { used_bytes: u64 },
    Critical { used_bytes: u64 },
}

pub struct StorageWatchdog {
    warn_threshold_bytes: u64,
    critical_threshold_bytes: u64,
    captures_dir: PathBuf,
    last_status: StorageStatus,
}

impl StorageWatchdog {
    pub fn new(warn_threshold_gb: f64, critical_threshold_gb: f64, captures_dir: PathBuf) -> Self {
        Self::new_bytes(
            (warn_threshold_gb * 1_000_000_000.0) as u64,
            (critical_threshold_gb * 1_000_000_000.0) as u64,
            captures_dir,
        )
    }

    pub fn new_bytes(
        warn_threshold_bytes: u64,
        critical_threshold_bytes: u64,
        captures_dir: PathBuf,
    ) -> Self {
        Self {
            warn_threshold_bytes,
            critical_threshold_bytes,
            captures_dir,
            last_status: StorageStatus::Ok { used_bytes: 0 },
        }
    }

    pub fn check(&mut self) -> StorageStatus {
        let used_bytes = self.calculate_dir_size();
        let status = if used_bytes >= self.critical_threshold_bytes {
            StorageStatus::Critical { used_bytes }
        } else if used_bytes >= self.warn_threshold_bytes {
            StorageStatus::Warning { used_bytes }
        } else {
            StorageStatus::Ok { used_bytes }
        };
        self.last_status = status.clone();
        status
    }

    pub fn is_recording_allowed(&self) -> bool {
        !matches!(self.last_status, StorageStatus::Critical { .. })
    }

    pub fn status(&self) -> &StorageStatus {
        &self.last_status
    }

    fn calculate_dir_size(&self) -> u64 {
        Self::dir_size_recursive(&self.captures_dir)
    }

    fn dir_size_recursive(path: &std::path::Path) -> u64 {
        let mut total = 0u64;
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let metadata = match entry.metadata() {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if metadata.is_dir() {
                    total += Self::dir_size_recursive(&entry.path());
                } else {
                    total += metadata.len();
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_file(dir: &std::path::Path, name: &str, size_bytes: usize) {
        std::fs::write(dir.join(name), vec![0u8; size_bytes]).unwrap();
    }

    #[test]
    fn test_check_returns_ok_when_below_warn_threshold() {
        let tmp = tempdir().unwrap();
        // Write a small file (well below any threshold)
        std::fs::write(tmp.path().join("test.jpg"), vec![0u8; 1024]).unwrap();

        let mut watchdog = StorageWatchdog::new(5.0, 8.0, tmp.path().to_path_buf());

        let status = watchdog.check();
        assert!(matches!(status, StorageStatus::Ok { .. }));
        assert!(watchdog.is_recording_allowed());
    }

    #[test]
    fn test_check_returns_warning_when_between_thresholds() {
        let tmp = tempdir().unwrap();
        // Use byte-scale thresholds: warn at 500 bytes, critical at 1000 bytes
        // Write 600 bytes — between warn and critical
        make_file(tmp.path(), "data.bin", 600);

        let mut watchdog = StorageWatchdog::new_bytes(500, 1000, tmp.path().to_path_buf());

        let status = watchdog.check();
        assert!(matches!(status, StorageStatus::Warning { .. }));
        assert!(watchdog.is_recording_allowed()); // warning still allows recording
    }

    #[test]
    fn test_check_returns_critical_and_blocks_recording() {
        let tmp = tempdir().unwrap();
        // Write 1500 bytes — above critical threshold of 1000
        make_file(tmp.path(), "big.bin", 1500);

        let mut watchdog = StorageWatchdog::new_bytes(500, 1000, tmp.path().to_path_buf());

        let status = watchdog.check();
        assert!(matches!(status, StorageStatus::Critical { .. }));
        assert!(!watchdog.is_recording_allowed());
    }

    #[test]
    fn test_recovery_from_critical_to_ok() {
        let tmp = tempdir().unwrap();
        let file_path = tmp.path().join("big.bin");

        let mut watchdog = StorageWatchdog::new_bytes(500, 1000, tmp.path().to_path_buf());

        // Start critical
        std::fs::write(&file_path, vec![0u8; 1500]).unwrap();
        watchdog.check();
        assert!(!watchdog.is_recording_allowed());

        // Remove file → usage drops
        std::fs::remove_file(&file_path).unwrap();
        watchdog.check();
        assert!(watchdog.is_recording_allowed());
    }

    #[test]
    fn test_counts_files_in_subdirectories() {
        let tmp = tempdir().unwrap();
        let sub = tmp.path().join("sessions");
        std::fs::create_dir(&sub).unwrap();

        // 400 bytes in root + 400 bytes in subdir = 800 total
        make_file(tmp.path(), "root.bin", 400);
        make_file(&sub, "sub.bin", 400);

        // Warn at 500, critical at 1000 → 800 bytes should be Warning
        let mut watchdog = StorageWatchdog::new_bytes(500, 1000, tmp.path().to_path_buf());
        let status = watchdog.check();
        assert!(matches!(status, StorageStatus::Warning { .. }));
        if let StorageStatus::Warning { used_bytes } = status {
            assert_eq!(used_bytes, 800);
        }
    }

    #[test]
    fn test_nonexistent_dir_returns_ok_with_zero() {
        let mut watchdog =
            StorageWatchdog::new_bytes(500, 1000, PathBuf::from("/tmp/does_not_exist_watchdog"));
        let status = watchdog.check();
        assert!(matches!(status, StorageStatus::Ok { used_bytes: 0 }));
    }
}
