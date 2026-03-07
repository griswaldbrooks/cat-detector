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

/// Real camera using v4l crate (pure Rust V4L2) - Linux only
#[cfg(feature = "real-camera")]
pub struct V4L2Camera {
    // SAFETY: stream borrows from device via raw pointer (see ensure_stream).
    // Field order matters: Rust drops fields in declaration order,
    // so stream is dropped before device, keeping the borrow valid.
    stream: Option<v4l::io::mmap::Stream<'static>>,
    device: v4l::Device,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
}

#[cfg(feature = "real-camera")]
impl V4L2Camera {
    pub fn new(device_path: &str, width: u32, height: u32, _fps: u32) -> Result<Self, CameraError> {
        use v4l::video::Capture;

        let device = v4l::Device::with_path(device_path)
            .map_err(|e| CameraError::InitError(format!("Failed to open {}: {}", device_path, e)))?;

        // Try MJPG format first (most compatible)
        let mut fmt = device.format().map_err(|e| {
            CameraError::InitError(format!("Failed to get format: {}", e))
        })?;

        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = v4l::FourCC::new(b"MJPG");

        let fmt = device.set_format(&fmt).map_err(|e| {
            CameraError::InitError(format!("Failed to set format: {}", e))
        })?;

        tracing::info!(
            "Camera configured: {}x{} {:?}",
            fmt.width,
            fmt.height,
            fmt.fourcc
        );

        Ok(Self {
            stream: None,
            device,
            width: fmt.width,
            height: fmt.height,
        })
    }

    fn ensure_stream(&mut self) -> Result<(), CameraError> {
        use v4l::io::mmap::Stream;

        if self.stream.is_none() {
            // SAFETY: Stream borrows from Device. This is safe because:
            // 1. Both live in the same struct, so device outlives stream
            // 2. Field order ensures stream is dropped before device
            // 3. capture_frame takes &mut self, so no concurrent access
            // 4. Device is not moved after stream creation (owned by struct)
            let device_ptr = &self.device as *const v4l::Device;
            let stream = unsafe {
                Stream::with_buffers(&*device_ptr, v4l::buffer::Type::VideoCapture, 4)
                    .map_err(|e| CameraError::InitError(format!("Failed to create stream: {}", e)))?
            };

            self.stream = Some(stream);
        }

        Ok(())
    }
}

#[cfg(feature = "real-camera")]
#[async_trait]
impl CameraCapture for V4L2Camera {
    async fn capture_frame(&mut self) -> Result<DynamicImage, CameraError> {
        use v4l::io::traits::CaptureStream;

        self.ensure_stream()?;

        let frame_data = {
            let stream = self.stream.as_mut().ok_or_else(|| {
                CameraError::CaptureError("Stream not initialized".to_string())
            })?;

            let (buf, _meta) = stream.next().map_err(|e| {
                CameraError::CaptureError(format!("Failed to capture frame: {}", e))
            })?;

            buf.to_vec()
        };

        // Decode the MJPG frame
        let img = image::load_from_memory(&frame_data).map_err(|e| {
            CameraError::CaptureError(format!("Failed to decode frame: {}", e))
        })?;

        Ok(img)
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(feature = "nokhwa-camera")]
pub struct NokhwaCamera {
    camera: nokhwa::Camera,
}

#[cfg(feature = "nokhwa-camera")]
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

#[cfg(feature = "nokhwa-camera")]
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
