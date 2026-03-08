use async_trait::async_trait;
use chrono::{DateTime, Utc};
use image::DynamicImage;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum StorageError {
    #[error("Failed to create directory: {0}")]
    DirectoryError(String),
    #[error("Failed to save image: {0}")]
    SaveError(String),
    #[error("Invalid image format: {0}")]
    FormatError(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImageType {
    Entry,
    Exit,
    Sample,
}

impl std::fmt::Display for ImageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageType::Entry => write!(f, "entry"),
            ImageType::Exit => write!(f, "exit"),
            ImageType::Sample => write!(f, "sample"),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SavedImage {
    pub path: PathBuf,
    pub timestamp: DateTime<Utc>,
    pub image_type: ImageType,
}

#[async_trait]
pub trait ImageStorage: Send + Sync {
    async fn save_image(
        &self,
        image: &DynamicImage,
        image_type: ImageType,
        timestamp: DateTime<Utc>,
    ) -> Result<SavedImage, StorageError>;

    #[allow(dead_code)]
    fn get_output_dir(&self) -> &PathBuf;
}

pub struct FileSystemStorage {
    output_dir: PathBuf,
    image_format: String,
    jpeg_quality: u8,
}

impl FileSystemStorage {
    pub fn new(
        output_dir: PathBuf,
        image_format: String,
        jpeg_quality: u8,
    ) -> Result<Self, StorageError> {
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| StorageError::DirectoryError(e.to_string()))?;

        Ok(Self {
            output_dir,
            image_format,
            jpeg_quality,
        })
    }

    fn generate_filename(&self, image_type: &ImageType, timestamp: DateTime<Utc>) -> String {
        let date_str = timestamp.format("%Y%m%d_%H%M%S");
        let millis = timestamp.timestamp_subsec_millis();
        format!(
            "cat_{}_{}.{:03}.{}",
            image_type, date_str, millis, self.image_format
        )
    }
}

#[async_trait]
impl ImageStorage for FileSystemStorage {
    async fn save_image(
        &self,
        image: &DynamicImage,
        image_type: ImageType,
        timestamp: DateTime<Utc>,
    ) -> Result<SavedImage, StorageError> {
        let filename = self.generate_filename(&image_type, timestamp);
        let path = self.output_dir.join(&filename);

        match self.image_format.as_str() {
            "jpg" | "jpeg" => {
                let mut file = std::fs::File::create(&path)
                    .map_err(|e| StorageError::SaveError(e.to_string()))?;
                let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                    &mut file,
                    self.jpeg_quality,
                );
                image
                    .to_rgb8()
                    .write_with_encoder(encoder)
                    .map_err(|e| StorageError::SaveError(e.to_string()))?;
            }
            "png" => {
                image
                    .save(&path)
                    .map_err(|e| StorageError::SaveError(e.to_string()))?;
            }
            other => {
                return Err(StorageError::FormatError(format!(
                    "Unsupported format: {}",
                    other
                )));
            }
        }

        Ok(SavedImage {
            path,
            timestamp,
            image_type,
        })
    }

    fn get_output_dir(&self) -> &PathBuf {
        &self.output_dir
    }
}

/// Mock storage for testing
#[cfg(test)]
pub struct MockStorage {
    saved_images: std::sync::Mutex<Vec<SavedImage>>,
    output_dir: PathBuf,
    should_fail: bool,
}

#[cfg(test)]
impl MockStorage {
    pub fn new() -> Self {
        Self {
            saved_images: std::sync::Mutex::new(Vec::new()),
            output_dir: PathBuf::from("/mock/captures"),
            should_fail: false,
        }
    }

    pub fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }

    pub fn get_saved_images(&self) -> Vec<SavedImage> {
        self.saved_images.lock().unwrap().clone()
    }

    pub fn get_images_by_type(&self, image_type: ImageType) -> Vec<SavedImage> {
        self.saved_images
            .lock()
            .unwrap()
            .iter()
            .filter(|img| img.image_type == image_type)
            .cloned()
            .collect()
    }

    pub fn count(&self) -> usize {
        self.saved_images.lock().unwrap().len()
    }

    pub fn clear(&self) {
        self.saved_images.lock().unwrap().clear();
    }
}

#[cfg(test)]
impl Default for MockStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[async_trait]
impl ImageStorage for MockStorage {
    async fn save_image(
        &self,
        _image: &DynamicImage,
        image_type: ImageType,
        timestamp: DateTime<Utc>,
    ) -> Result<SavedImage, StorageError> {
        if self.should_fail {
            return Err(StorageError::SaveError("Mock failure".to_string()));
        }

        let filename = format!(
            "cat_{}_{}.jpg",
            image_type,
            timestamp.format("%Y%m%d_%H%M%S")
        );
        let path = self.output_dir.join(filename);

        let saved = SavedImage {
            path,
            timestamp,
            image_type,
        };

        self.saved_images.lock().unwrap().push(saved.clone());
        Ok(saved)
    }

    fn get_output_dir(&self) -> &PathBuf {
        &self.output_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_storage_records_saves() {
        let storage = MockStorage::new();
        let image = DynamicImage::new_rgb8(100, 100);
        let timestamp = Utc::now();

        let result = storage
            .save_image(&image, ImageType::Entry, timestamp)
            .await;
        assert!(result.is_ok());
        assert_eq!(storage.count(), 1);

        storage
            .save_image(&image, ImageType::Sample, timestamp)
            .await
            .unwrap();
        assert_eq!(storage.count(), 2);
    }

    #[tokio::test]
    async fn test_mock_storage_filters_by_type() {
        let storage = MockStorage::new();
        let image = DynamicImage::new_rgb8(100, 100);
        let timestamp = Utc::now();

        storage
            .save_image(&image, ImageType::Entry, timestamp)
            .await
            .unwrap();
        storage
            .save_image(&image, ImageType::Sample, timestamp)
            .await
            .unwrap();
        storage
            .save_image(&image, ImageType::Sample, timestamp)
            .await
            .unwrap();
        storage
            .save_image(&image, ImageType::Exit, timestamp)
            .await
            .unwrap();

        assert_eq!(storage.get_images_by_type(ImageType::Entry).len(), 1);
        assert_eq!(storage.get_images_by_type(ImageType::Sample).len(), 2);
        assert_eq!(storage.get_images_by_type(ImageType::Exit).len(), 1);
    }

    #[tokio::test]
    async fn test_mock_storage_failure() {
        let storage = MockStorage::new().with_failure();
        let image = DynamicImage::new_rgb8(100, 100);

        let result = storage
            .save_image(&image, ImageType::Entry, Utc::now())
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_filename_format() {
        let storage = FileSystemStorage {
            output_dir: PathBuf::from("/test"),
            image_format: "jpg".to_string(),
            jpeg_quality: 85,
        };

        let timestamp = DateTime::parse_from_rfc3339("2024-01-15T10:30:45.123Z")
            .unwrap()
            .with_timezone(&Utc);

        let filename = storage.generate_filename(&ImageType::Entry, timestamp);
        assert!(filename.starts_with("cat_entry_20240115_103045"));
        assert!(filename.ends_with(".jpg"));
    }

    #[test]
    fn test_image_type_display() {
        assert_eq!(format!("{}", ImageType::Entry), "entry");
        assert_eq!(format!("{}", ImageType::Exit), "exit");
        assert_eq!(format!("{}", ImageType::Sample), "sample");
    }

    #[tokio::test]
    async fn test_saved_image_has_correct_metadata() {
        let storage = MockStorage::new();
        let image = DynamicImage::new_rgb8(100, 100);
        let timestamp = Utc::now();

        let saved = storage
            .save_image(&image, ImageType::Entry, timestamp)
            .await
            .unwrap();

        assert_eq!(saved.image_type, ImageType::Entry);
        assert_eq!(saved.timestamp, timestamp);
        assert!(saved.path.to_string_lossy().contains("entry"));
    }
}
