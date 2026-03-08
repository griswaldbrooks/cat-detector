use async_trait::async_trait;
use image::DynamicImage;
use ndarray::ArrayView2;
use ort::session::Session;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum DetectorError {
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),
    #[error("Failed to run inference: {0}")]
    InferenceError(String),
    #[allow(dead_code)]
    #[error("Failed to preprocess image: {0}")]
    PreprocessError(String),
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub class_id: u32,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[async_trait]
pub trait CatDetector: Send + Sync {
    async fn detect(&self, image: &DynamicImage) -> Result<Vec<Detection>, DetectorError>;
    fn is_cat(&self, detection: &Detection) -> bool;
}

/// ONNX-based YOLOX detector
pub struct OnnxDetector {
    session: std::sync::Mutex<Session>,
    confidence_threshold: f32,
    cat_class_id: u32,
    input_width: u32,
    input_height: u32,
}

impl OnnxDetector {
    pub fn new(
        model_path: &std::path::Path,
        confidence_threshold: f32,
        cat_class_id: u32,
    ) -> Result<Self, DetectorError> {
        // Default to 416 for yolox_tiny, can be overridden with new_with_size
        Self::new_with_size(model_path, confidence_threshold, cat_class_id, 416)
    }

    pub fn new_with_size(
        model_path: &std::path::Path,
        confidence_threshold: f32,
        cat_class_id: u32,
        input_size: u32,
    ) -> Result<Self, DetectorError> {
        let session = Session::builder()
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            confidence_threshold,
            cat_class_id,
            input_width: input_size,
            input_height: input_size,
        })
    }

    fn preprocess(&self, image: &DynamicImage) -> Result<ndarray::Array4<f32>, DetectorError> {
        let resized = image.resize_exact(
            self.input_width,
            self.input_height,
            image::imageops::FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();

        let mut array = ndarray::Array4::<f32>::zeros((
            1,
            3,
            self.input_height as usize,
            self.input_width as usize,
        ));

        // YOLOX expects pixel values in 0-255 range (not normalized)
        for (x, y, pixel) in rgb.enumerate_pixels() {
            array[[0, 0, y as usize, x as usize]] = pixel[0] as f32;
            array[[0, 1, y as usize, x as usize]] = pixel[1] as f32;
            array[[0, 2, y as usize, x as usize]] = pixel[2] as f32;
        }

        Ok(array)
    }

    fn postprocess(
        &self,
        output: &ArrayView2<f32>,
        original_width: u32,
        original_height: u32,
    ) -> Vec<Detection> {
        let mut detections = Vec::new();

        // YOLOX output format: [num_boxes, 85] where 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
        // The model outputs grid-relative values that need to be decoded with strides
        // YOLOX-tiny @ 416: strides are 8, 16, 32 -> grids are 52x52, 26x26, 13x13

        let strides = [8, 16, 32];
        let mut grids = Vec::new();

        // Generate grid coordinates for each stride
        for &stride in &strides {
            let grid_size = self.input_width / stride;
            for gy in 0..grid_size {
                for gx in 0..grid_size {
                    grids.push((gx as f32, gy as f32, stride as f32));
                }
            }
        }

        for (i, row) in output.rows().into_iter().enumerate() {
            if i >= grids.len() {
                break;
            }

            let (gx, gy, stride) = grids[i];

            // Decode bbox: x/y are offsets from grid cell, w/h need exp()
            let x_center = (row[0] + gx) * stride;
            let y_center = (row[1] + gy) * stride;
            let width = row[2].exp() * stride;
            let height = row[3].exp() * stride;
            let objectness = row[4];

            // Find class with highest score
            let row_slice = match row.as_slice() {
                Some(s) => s,
                None => continue,
            };
            let class_scores = &row_slice[5..];
            let (class_id, &class_score) = match class_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
            {
                Some(result) => result,
                None => continue,
            };

            // YOLOX confidence = objectness * class_score
            let confidence = objectness * class_score;

            if confidence >= self.confidence_threshold {
                // Scale coordinates back to original image size
                let scale_x = original_width as f32 / self.input_width as f32;
                let scale_y = original_height as f32 / self.input_height as f32;

                detections.push(Detection {
                    class_id: class_id as u32,
                    confidence,
                    bbox: BoundingBox {
                        x: (x_center - width / 2.0) * scale_x,
                        y: (y_center - height / 2.0) * scale_y,
                        width: width * scale_x,
                        height: height * scale_y,
                    },
                });
            }
        }

        // Apply NMS (Non-Maximum Suppression)
        self.non_max_suppression(detections, 0.45)
    }

    fn non_max_suppression(
        &self,
        mut detections: Vec<Detection>,
        iou_threshold: f32,
    ) -> Vec<Detection> {
        // Sort ascending so we can pop highest-confidence from the end (O(1))
        detections.sort_by(|a, b| a.confidence.total_cmp(&b.confidence));

        let mut keep = Vec::new();

        while let Some(current) = detections.pop() {
            keep.push(current.clone());

            detections.retain(|d| {
                if d.class_id != current.class_id {
                    return true;
                }
                self.iou(&current.bbox, &d.bbox) < iou_threshold
            });
        }

        keep
    }

    fn iou(&self, a: &BoundingBox, b: &BoundingBox) -> f32 {
        let x1 = a.x.max(b.x);
        let y1 = a.y.max(b.y);
        let x2 = (a.x + a.width).min(b.x + b.width);
        let y2 = (a.y + a.height).min(b.y + b.height);

        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let area_a = a.width * a.height;
        let area_b = b.width * b.height;
        let union = area_a + area_b - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}

#[async_trait]
impl CatDetector for OnnxDetector {
    async fn detect(&self, image: &DynamicImage) -> Result<Vec<Detection>, DetectorError> {
        let original_width = image.width();
        let original_height = image.height();

        let input = self.preprocess(image)?;

        // Create input tensor using the shape and data
        let shape = input.shape().to_vec();
        let data = input.into_raw_vec_and_offset().0;
        let input_tensor = ort::value::Tensor::from_array((shape, data))
            .map_err(|e| DetectorError::InferenceError(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| DetectorError::InferenceError(format!("Failed to lock session: {}", e)))?;

        let outputs = session
            .run(ort::inputs!["images" => input_tensor])
            .map_err(|e| DetectorError::InferenceError(e.to_string()))?;

        let output_value = outputs
            .get("output")
            .ok_or_else(|| DetectorError::InferenceError("Missing output".to_string()))?;

        let (_shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e: ort::Error| DetectorError::InferenceError(e.to_string()))?;

        // Get data from the tensor
        let flat_data: Vec<f32> = data.to_vec();

        // YOLOX output is [1, num_boxes, 85], reshape to [num_boxes, 85]
        let num_boxes = flat_data.len() / 85;
        let array = ndarray::Array2::from_shape_vec((num_boxes, 85), flat_data)
            .map_err(|e| DetectorError::InferenceError(e.to_string()))?;

        let detections = self.postprocess(&array.view(), original_width, original_height);
        Ok(detections)
    }

    fn is_cat(&self, detection: &Detection) -> bool {
        detection.class_id == self.cat_class_id && detection.confidence >= self.confidence_threshold
    }
}

/// Mock detector for testing
#[cfg(test)]
pub struct MockDetector {
    results: std::sync::Mutex<Vec<Vec<Detection>>>,
    call_count: std::sync::atomic::AtomicUsize,
    cat_class_id: u32,
    confidence_threshold: f32,
}

#[cfg(test)]
impl MockDetector {
    pub fn new(results: Vec<Vec<Detection>>) -> Self {
        Self {
            results: std::sync::Mutex::new(results),
            call_count: std::sync::atomic::AtomicUsize::new(0),
            cat_class_id: 15,
            confidence_threshold: 0.5,
        }
    }

    pub fn always_detect_cat() -> Self {
        Self::new(vec![vec![Detection {
            class_id: 15,
            confidence: 0.95,
            bbox: BoundingBox {
                x: 100.0,
                y: 100.0,
                width: 200.0,
                height: 200.0,
            },
        }]])
    }

    pub fn never_detect() -> Self {
        Self::new(vec![vec![]])
    }

    pub fn with_sequence(detections: Vec<bool>) -> Self {
        let results = detections
            .into_iter()
            .map(|detected| {
                if detected {
                    vec![Detection {
                        class_id: 15,
                        confidence: 0.95,
                        bbox: BoundingBox {
                            x: 100.0,
                            y: 100.0,
                            width: 200.0,
                            height: 200.0,
                        },
                    }]
                } else {
                    vec![]
                }
            })
            .collect();
        Self::new(results)
    }

    pub fn call_count(&self) -> usize {
        self.call_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(test)]
#[async_trait]
impl CatDetector for MockDetector {
    async fn detect(&self, _image: &DynamicImage) -> Result<Vec<Detection>, DetectorError> {
        let count = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let results = self.results.lock().unwrap();

        if results.is_empty() {
            return Ok(vec![]);
        }

        let index = count % results.len();
        Ok(results[index].clone())
    }

    fn is_cat(&self, detection: &Detection) -> bool {
        detection.class_id == self.cat_class_id && detection.confidence >= self.confidence_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_detector_always_detect() {
        let detector = MockDetector::always_detect_cat();
        let image = DynamicImage::new_rgb8(640, 480);

        let detections = detector.detect(&image).await.unwrap();
        assert_eq!(detections.len(), 1);
        assert!(detector.is_cat(&detections[0]));
    }

    #[tokio::test]
    async fn test_mock_detector_never_detect() {
        let detector = MockDetector::never_detect();
        let image = DynamicImage::new_rgb8(640, 480);

        let detections = detector.detect(&image).await.unwrap();
        assert!(detections.is_empty());
    }

    #[tokio::test]
    async fn test_mock_detector_sequence() {
        let detector = MockDetector::with_sequence(vec![true, false, true]);
        let image = DynamicImage::new_rgb8(640, 480);

        let d1 = detector.detect(&image).await.unwrap();
        assert_eq!(d1.len(), 1);

        let d2 = detector.detect(&image).await.unwrap();
        assert!(d2.is_empty());

        let d3 = detector.detect(&image).await.unwrap();
        assert_eq!(d3.len(), 1);

        // Should cycle back
        let d4 = detector.detect(&image).await.unwrap();
        assert_eq!(d4.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_detector_tracks_call_count() {
        let detector = MockDetector::always_detect_cat();
        let image = DynamicImage::new_rgb8(640, 480);

        assert_eq!(detector.call_count(), 0);
        detector.detect(&image).await.unwrap();
        assert_eq!(detector.call_count(), 1);
        detector.detect(&image).await.unwrap();
        assert_eq!(detector.call_count(), 2);
    }

    #[test]
    fn test_is_cat_checks_class_and_confidence() {
        let detector = MockDetector::always_detect_cat();

        let cat_detection = Detection {
            class_id: 15,
            confidence: 0.8,
            bbox: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
            },
        };
        assert!(detector.is_cat(&cat_detection));

        let dog_detection = Detection {
            class_id: 16, // Dog class
            confidence: 0.8,
            bbox: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
            },
        };
        assert!(!detector.is_cat(&dog_detection));

        let low_confidence = Detection {
            class_id: 15,
            confidence: 0.3, // Below threshold
            bbox: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
            },
        };
        assert!(!detector.is_cat(&low_confidence));
    }

    #[test]
    fn test_iou_calculation() {
        // Test IOU with helper function
        fn iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
            let x1 = a.x.max(b.x);
            let y1 = a.y.max(b.y);
            let x2 = (a.x + a.width).min(b.x + b.width);
            let y2 = (a.y + a.height).min(b.y + b.height);

            let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
            let area_a = a.width * a.height;
            let area_b = b.width * b.height;
            let union = area_a + area_b - intersection;

            if union > 0.0 {
                intersection / union
            } else {
                0.0
            }
        }

        // Identical boxes should have IOU of 1.0
        let box1 = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        };
        let box2 = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        };
        assert!((iou(&box1, &box2) - 1.0).abs() < 0.001);

        // Non-overlapping boxes should have IOU of 0.0
        let box3 = BoundingBox {
            x: 200.0,
            y: 200.0,
            width: 100.0,
            height: 100.0,
        };
        assert!((iou(&box1, &box3) - 0.0).abs() < 0.001);

        // 50% overlap
        let box4 = BoundingBox {
            x: 50.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        };
        // Intersection = 50*100 = 5000, Union = 10000 + 10000 - 5000 = 15000
        // IOU = 5000/15000 = 0.333...
        assert!((iou(&box1, &box4) - 0.333).abs() < 0.01);
    }
}
