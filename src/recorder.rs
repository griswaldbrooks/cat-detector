use image::DynamicImage;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RecorderError {
    #[error("Failed to start recording: {0}")]
    StartError(String),
    #[error("Failed to write frame: {0}")]
    WriteError(String),
    #[error("Failed to finalize recording: {0}")]
    FinalizeError(String),
    #[error("Already recording")]
    AlreadyRecording,
}

/// Trait for recording video from frames
pub trait VideoRecorder: Send + Sync {
    /// Start recording to a new video file.
    fn start_recording(
        &mut self,
        output_dir: &Path,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<(), RecorderError>;

    /// Add a frame to the current recording. No-op if not recording.
    fn add_frame(&mut self, image: &DynamicImage) -> Result<(), RecorderError>;

    /// Stop recording and finalize the video file. Returns the path if recording was active.
    fn stop_recording(&mut self) -> Result<Option<PathBuf>, RecorderError>;

    /// Check if currently recording.
    fn is_recording(&self) -> bool;
}

/// FFmpeg-based video recorder that pipes raw frames to an ffmpeg process
pub struct FfmpegRecorder {
    ffmpeg_path: String,
    width: u32,
    height: u32,
    fps: u32,
    process: Option<std::process::Child>,
    output_path: Option<PathBuf>,
}

impl FfmpegRecorder {
    pub fn new(ffmpeg_path: &str, width: u32, height: u32, fps: u32) -> Self {
        Self {
            ffmpeg_path: ffmpeg_path.to_string(),
            width,
            height,
            fps,
            process: None,
            output_path: None,
        }
    }
}

impl VideoRecorder for FfmpegRecorder {
    fn start_recording(
        &mut self,
        output_dir: &Path,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<(), RecorderError> {
        if self.is_recording() {
            return Err(RecorderError::AlreadyRecording);
        }

        // Create output directory if needed
        std::fs::create_dir_all(output_dir)
            .map_err(|e| RecorderError::StartError(format!("Failed to create dir: {}", e)))?;

        let filename = format!("cat_video_{}.mp4", timestamp.format("%Y%m%d_%H%M%S"));
        let output_path = output_dir.join(&filename);

        // Spawn ffmpeg process
        // Input: raw RGB24 frames piped to stdin
        // Output: H.264 MP4 file
        let child = std::process::Command::new(&self.ffmpeg_path)
            .args([
                "-y", // overwrite
                "-f",
                "rawvideo", // input format
                "-pixel_format",
                "rgb24", // pixel format
                "-video_size",
                &format!("{}x{}", self.width, self.height),
                "-framerate",
                &self.fps.to_string(), // input framerate
                "-i",
                "pipe:0", // read from stdin
                "-c:v",
                "libx264", // H.264 codec
                "-preset",
                "ultrafast", // fast encoding
                "-crf",
                "23", // quality
                "-pix_fmt",
                "yuv420p", // browser-compatible
                "-movflags",
                "+faststart", // streaming-friendly
            ])
            .arg(output_path.to_str().unwrap())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| RecorderError::StartError(format!("Failed to spawn ffmpeg: {}", e)))?;

        tracing::info!("Started recording to {:?}", output_path);
        self.process = Some(child);
        self.output_path = Some(output_path);
        Ok(())
    }

    fn add_frame(&mut self, image: &DynamicImage) -> Result<(), RecorderError> {
        let child = match self.process.as_mut() {
            Some(c) => c,
            None => return Ok(()), // Not recording, no-op
        };

        let rgb = image
            .resize_exact(
                self.width,
                self.height,
                image::imageops::FilterType::Triangle,
            )
            .to_rgb8();

        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| RecorderError::WriteError("FFmpeg stdin not available".to_string()))?;

        use std::io::Write;
        stdin
            .write_all(rgb.as_raw())
            .map_err(|e| RecorderError::WriteError(format!("Failed to write frame: {}", e)))?;

        Ok(())
    }

    fn stop_recording(&mut self) -> Result<Option<PathBuf>, RecorderError> {
        let mut child = match self.process.take() {
            Some(c) => c,
            None => return Ok(None),
        };

        // Close stdin to signal EOF to ffmpeg
        drop(child.stdin.take());

        // Wait for ffmpeg to finish
        let status = child.wait().map_err(|e| {
            RecorderError::FinalizeError(format!("Failed to wait for ffmpeg: {}", e))
        })?;

        let path = self.output_path.take();

        if !status.success() {
            return Err(RecorderError::FinalizeError(format!(
                "FFmpeg exited with status: {}",
                status
            )));
        }

        tracing::info!("Finished recording to {:?}", path);
        Ok(path)
    }

    fn is_recording(&self) -> bool {
        self.process.is_some()
    }
}

/// Mock video recorder for testing
#[cfg(test)]
pub struct MockVideoRecorder {
    recording: bool,
    frame_count: usize,
    recordings: Vec<(usize, PathBuf)>, // (frame_count, path) for each completed recording
    output_dir: Option<PathBuf>,
}

#[cfg(test)]
impl MockVideoRecorder {
    pub fn new() -> Self {
        Self {
            recording: false,
            frame_count: 0,
            recordings: Vec::new(),
            output_dir: None,
        }
    }

    pub fn completed_recordings(&self) -> &[(usize, PathBuf)] {
        &self.recordings
    }
}

#[cfg(test)]
impl VideoRecorder for MockVideoRecorder {
    fn start_recording(
        &mut self,
        output_dir: &Path,
        _timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<(), RecorderError> {
        if self.recording {
            return Err(RecorderError::AlreadyRecording);
        }
        self.recording = true;
        self.frame_count = 0;
        self.output_dir = Some(output_dir.to_path_buf());
        Ok(())
    }

    fn add_frame(&mut self, _image: &DynamicImage) -> Result<(), RecorderError> {
        if !self.recording {
            return Ok(());
        }
        self.frame_count += 1;
        Ok(())
    }

    fn stop_recording(&mut self) -> Result<Option<PathBuf>, RecorderError> {
        if !self.recording {
            return Ok(None);
        }
        self.recording = false;
        let path = self
            .output_dir
            .take()
            .unwrap_or_default()
            .join("mock_video.mp4");
        self.recordings.push((self.frame_count, path.clone()));
        self.frame_count = 0;
        Ok(Some(path))
    }

    fn is_recording(&self) -> bool {
        self.recording
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_recorder_start_stop_lifecycle() {
        let mut recorder = MockVideoRecorder::new();
        assert!(!recorder.is_recording());

        recorder
            .start_recording(Path::new("/tmp"), chrono::Utc::now())
            .unwrap();
        assert!(recorder.is_recording());

        let path = recorder.stop_recording().unwrap();
        assert!(!recorder.is_recording());
        assert!(path.is_some());
        assert!(path.unwrap().to_str().unwrap().ends_with(".mp4"));
    }

    #[test]
    fn test_mock_recorder_add_frames_while_recording() {
        let mut recorder = MockVideoRecorder::new();
        let img = DynamicImage::new_rgb8(640, 480);

        recorder
            .start_recording(Path::new("/tmp"), chrono::Utc::now())
            .unwrap();

        recorder.add_frame(&img).unwrap();
        recorder.add_frame(&img).unwrap();
        recorder.add_frame(&img).unwrap();

        recorder.stop_recording().unwrap();

        assert_eq!(recorder.completed_recordings().len(), 1);
        assert_eq!(recorder.completed_recordings()[0].0, 3); // 3 frames
    }

    #[test]
    fn test_mock_recorder_add_frame_when_not_recording_is_noop() {
        let mut recorder = MockVideoRecorder::new();
        let img = DynamicImage::new_rgb8(640, 480);

        // Should not error
        recorder.add_frame(&img).unwrap();

        assert_eq!(recorder.completed_recordings().len(), 0);
    }

    #[test]
    fn test_mock_recorder_stop_when_not_recording_returns_none() {
        let mut recorder = MockVideoRecorder::new();

        let result = recorder.stop_recording().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_mock_recorder_double_start_errors() {
        let mut recorder = MockVideoRecorder::new();

        recorder
            .start_recording(Path::new("/tmp"), chrono::Utc::now())
            .unwrap();

        let result = recorder.start_recording(Path::new("/tmp"), chrono::Utc::now());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RecorderError::AlreadyRecording
        ));
    }

    #[test]
    fn test_ffmpeg_recorder_start_stop_with_real_ffmpeg() {
        // Skip if ffmpeg not available
        if std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_err()
        {
            eprintln!("Skipping: ffmpeg not available");
            return;
        }

        let tmp = tempfile::tempdir().unwrap();
        let mut recorder = FfmpegRecorder::new("ffmpeg", 640, 480, 30);

        assert!(!recorder.is_recording());

        recorder
            .start_recording(tmp.path(), chrono::Utc::now())
            .unwrap();
        assert!(recorder.is_recording());

        // Write a few frames
        let img = DynamicImage::new_rgb8(640, 480);
        for _ in 0..30 {
            recorder.add_frame(&img).unwrap();
        }

        let path = recorder.stop_recording().unwrap();
        assert!(!recorder.is_recording());
        assert!(path.is_some());

        let video_path = path.unwrap();
        assert!(video_path.exists());
        assert!(std::fs::metadata(&video_path).unwrap().len() > 0);
    }
}
