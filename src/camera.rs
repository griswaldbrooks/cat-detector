use async_trait::async_trait;
use image::{DynamicImage, RgbImage};
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum CameraError {
    #[error("Failed to initialize camera: {0}")]
    InitError(String),
    #[error("Failed to capture frame: {0}")]
    CaptureError(String),
    #[error("Camera not available: {0}")]
    NotAvailable(String),
}

#[async_trait]
pub trait CameraCapture: Send + Sync {
    async fn capture_frame(&mut self) -> Result<DynamicImage, CameraError>;
    #[allow(dead_code)]
    fn is_available(&self) -> bool;
}

#[cfg(feature = "real-camera")]
pub struct NokhwaCamera {
    camera: nokhwa::Camera,
}

#[cfg(feature = "real-camera")]
impl NokhwaCamera {
    pub fn new(device_path: &str, width: u32, height: u32, fps: u32) -> Result<Self, CameraError> {
        use nokhwa::pixel_format::RgbFormat;
        use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};

        let index = CameraIndex::String(device_path.to_string());
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        let camera = nokhwa::Camera::new(index, requested)
            .map_err(|e| CameraError::InitError(e.to_string()))?;

        Ok(Self { camera })
    }
}

#[cfg(feature = "real-camera")]
#[async_trait]
impl CameraCapture for NokhwaCamera {
    async fn capture_frame(&mut self) -> Result<DynamicImage, CameraError> {
        let frame = self.camera.frame()
            .map_err(|e| CameraError::CaptureError(e.to_string()))?;

        let decoded = frame.decode_image::<nokhwa::pixel_format::RgbFormat>()
            .map_err(|e| CameraError::CaptureError(e.to_string()))?;

        Ok(DynamicImage::ImageRgb8(decoded))
    }

    fn is_available(&self) -> bool {
        self.camera.is_stream_open()
    }
}

/// A stub camera for when the real camera feature is not enabled
pub struct StubCamera {
    width: u32,
    height: u32,
}

impl StubCamera {
    pub fn new(_device_path: &str, width: u32, height: u32, _fps: u32) -> Result<Self, CameraError> {
        Ok(Self { width, height })
    }
}

#[async_trait]
impl CameraCapture for StubCamera {
    async fn capture_frame(&mut self) -> Result<DynamicImage, CameraError> {
        // Return a blank image for testing/stub purposes
        let img = RgbImage::new(self.width, self.height);
        Ok(DynamicImage::ImageRgb8(img))
    }

    fn is_available(&self) -> bool {
        true
    }
}

/// Mock camera for testing
#[cfg(test)]
pub struct MockCamera {
    frames: Vec<DynamicImage>,
    current_index: usize,
    available: bool,
}

#[cfg(test)]
impl MockCamera {
    pub fn new(frames: Vec<DynamicImage>) -> Self {
        Self {
            frames,
            current_index: 0,
            available: true,
        }
    }

    pub fn with_availability(mut self, available: bool) -> Self {
        self.available = available;
        self
    }

    pub fn from_solid_colors(colors: Vec<[u8; 3]>, width: u32, height: u32) -> Self {
        let frames = colors
            .into_iter()
            .map(|color| {
                let mut img = RgbImage::new(width, height);
                for pixel in img.pixels_mut() {
                    *pixel = image::Rgb(color);
                }
                DynamicImage::ImageRgb8(img)
            })
            .collect();
        Self::new(frames)
    }
}

#[cfg(test)]
#[async_trait]
impl CameraCapture for MockCamera {
    async fn capture_frame(&mut self) -> Result<DynamicImage, CameraError> {
        if !self.available {
            return Err(CameraError::NotAvailable("Mock camera unavailable".to_string()));
        }

        if self.frames.is_empty() {
            return Err(CameraError::CaptureError("No frames available".to_string()));
        }

        let frame = self.frames[self.current_index].clone();
        self.current_index = (self.current_index + 1) % self.frames.len();
        Ok(frame)
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_camera_cycles_through_frames() {
        let colors = vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let mut camera = MockCamera::from_solid_colors(colors, 100, 100);

        let frame1 = camera.capture_frame().await.unwrap();
        let frame2 = camera.capture_frame().await.unwrap();
        let frame3 = camera.capture_frame().await.unwrap();
        let frame4 = camera.capture_frame().await.unwrap(); // Should cycle back

        // Verify dimensions
        assert_eq!(frame1.width(), 100);
        assert_eq!(frame1.height(), 100);

        // Verify cycling (frame4 should be same color as frame1)
        let rgb1 = frame1.to_rgb8();
        let rgb4 = frame4.to_rgb8();
        assert_eq!(rgb1.get_pixel(0, 0), rgb4.get_pixel(0, 0));
    }

    #[tokio::test]
    async fn test_mock_camera_availability() {
        let camera = MockCamera::from_solid_colors(vec![[0, 0, 0]], 10, 10);
        assert!(camera.is_available());

        let camera = MockCamera::from_solid_colors(vec![[0, 0, 0]], 10, 10)
            .with_availability(false);
        assert!(!camera.is_available());
    }

    #[tokio::test]
    async fn test_mock_camera_unavailable_returns_error() {
        let mut camera = MockCamera::from_solid_colors(vec![[0, 0, 0]], 10, 10)
            .with_availability(false);

        let result = camera.capture_frame().await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CameraError::NotAvailable(_)));
    }

    #[tokio::test]
    async fn test_stub_camera_returns_blank_image() {
        let mut camera = StubCamera::new("/dev/video0", 640, 480, 30).unwrap();

        let frame = camera.capture_frame().await.unwrap();
        assert_eq!(frame.width(), 640);
        assert_eq!(frame.height(), 480);
    }
}
