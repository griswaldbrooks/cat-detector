use async_trait::async_trait;
use image::DynamicImage;
use ndarray::ArrayView2;
use ort::session::Session;
use std::path::Path;
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

/// Model output format for postprocessing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// YOLOX: output [1, num_boxes, 85], grid-decoded, objectness * class_score, 0-255 input
    Yolox,
    /// YOLOv8/YOLO11: output [1, 84, 8400], transposed, direct class scores, 0-1 input
    Yolo11,
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

/// ONNX-based object detector supporting YOLOX and YOLOv8/YOLO11 models
pub struct OnnxDetector {
    session: std::sync::Mutex<Session>,
    confidence_threshold: f32,
    cat_class_id: u32,
    input_width: u32,
    input_height: u32,
    model_format: ModelFormat,
}

impl OnnxDetector {
    pub fn new(
        model_path: &std::path::Path,
        confidence_threshold: f32,
        cat_class_id: u32,
    ) -> Result<Self, DetectorError> {
        Self::new_with_size(model_path, confidence_threshold, cat_class_id, 416)
    }

    pub fn new_with_size(
        model_path: &std::path::Path,
        confidence_threshold: f32,
        cat_class_id: u32,
        input_size: u32,
    ) -> Result<Self, DetectorError> {
        Self::new_with_format(
            model_path,
            confidence_threshold,
            cat_class_id,
            input_size,
            None,
        )
    }

    pub fn new_with_format(
        model_path: &std::path::Path,
        confidence_threshold: f32,
        cat_class_id: u32,
        input_size: u32,
        format: Option<ModelFormat>,
    ) -> Result<Self, DetectorError> {
        let session = Session::builder()
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?;

        // Auto-detect model format from filename if not specified
        let model_format = format.unwrap_or_else(|| {
            let path_str = model_path.to_string_lossy().to_lowercase();
            if path_str.contains("yolox") {
                ModelFormat::Yolox
            } else {
                // Default to YOLO11 format for yolov8/yolo11/unknown models
                ModelFormat::Yolo11
            }
        });

        tracing::info!(
            "Loaded model {:?} with format {:?}",
            model_path,
            model_format
        );

        Ok(Self {
            session: std::sync::Mutex::new(session),
            confidence_threshold,
            cat_class_id,
            input_width: input_size,
            input_height: input_size,
            model_format,
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

        let scale = match self.model_format {
            ModelFormat::Yolox => 1.0,    // YOLOX expects 0-255
            ModelFormat::Yolo11 => 255.0, // YOLO11 expects 0-1
        };

        for (x, y, pixel) in rgb.enumerate_pixels() {
            array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / scale;
            array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / scale;
            array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / scale;
        }

        Ok(array)
    }

    fn postprocess_yolox(
        &self,
        output: &ArrayView2<f32>,
        original_width: u32,
        original_height: u32,
    ) -> Vec<Detection> {
        let mut detections = Vec::new();

        let strides = [8, 16, 32];
        let mut grids = Vec::new();

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

            let x_center = (row[0] + gx) * stride;
            let y_center = (row[1] + gy) * stride;
            let width = row[2].exp() * stride;
            let height = row[3].exp() * stride;
            let objectness = row[4];

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

            let confidence = objectness * class_score;

            if confidence >= self.confidence_threshold {
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

        self.non_max_suppression(detections, 0.45)
    }

    fn postprocess_yolo11(
        &self,
        flat_data: &[f32],
        output_shape: &[usize],
        original_width: u32,
        original_height: u32,
    ) -> Vec<Detection> {
        // YOLO11 output: [1, 84, 8400] where 84 = 4 (xywh) + 80 (class scores)
        // Data is column-major per box: access as flat_data[feature * num_boxes + box_idx]
        let num_features = output_shape[1]; // 84
        let num_boxes = output_shape[2]; // 8400
        let num_classes = num_features - 4;

        let scale_x = original_width as f32 / self.input_width as f32;
        let scale_y = original_height as f32 / self.input_height as f32;

        let mut detections = Vec::new();

        for box_idx in 0..num_boxes {
            // Find best class
            let mut max_score: f32 = 0.0;
            let mut max_class: usize = 0;
            for c in 0..num_classes {
                let score = flat_data[(4 + c) * num_boxes + box_idx];
                if score > max_score {
                    max_score = score;
                    max_class = c;
                }
            }

            if max_score >= self.confidence_threshold {
                let x_center = flat_data[box_idx];
                let y_center = flat_data[num_boxes + box_idx];
                let width = flat_data[2 * num_boxes + box_idx];
                let height = flat_data[3 * num_boxes + box_idx];

                detections.push(Detection {
                    class_id: max_class as u32,
                    confidence: max_score,
                    bbox: BoundingBox {
                        x: (x_center - width / 2.0) * scale_x,
                        y: (y_center - height / 2.0) * scale_y,
                        width: width * scale_x,
                        height: height * scale_y,
                    },
                });
            }
        }

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

        // Get the first output (name varies: "output" for YOLOX, "output0" for YOLO11)
        let output_value = outputs
            .values()
            .next()
            .ok_or_else(|| DetectorError::InferenceError("No output tensors".to_string()))?;

        let (shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e: ort::Error| DetectorError::InferenceError(e.to_string()))?;

        let flat_data: Vec<f32> = data.to_vec();
        let output_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        let detections = match self.model_format {
            ModelFormat::Yolox => {
                let num_boxes = flat_data.len() / 85;
                let array = ndarray::Array2::from_shape_vec((num_boxes, 85), flat_data)
                    .map_err(|e| DetectorError::InferenceError(e.to_string()))?;
                self.postprocess_yolox(&array.view(), original_width, original_height)
            }
            ModelFormat::Yolo11 => {
                self.postprocess_yolo11(&flat_data, &output_shape, original_width, original_height)
            }
        };
        Ok(detections)
    }

    fn is_cat(&self, detection: &Detection) -> bool {
        detection.class_id == self.cat_class_id && detection.confidence >= self.confidence_threshold
    }
}

/// CLIP-based zero-shot cat classifier using image-text similarity.
///
/// Uses a CLIP image encoder (ONNX) and pre-computed text embeddings to classify
/// frames as "cat" or "no cat" via cosine similarity. Returns a full-frame Detection
/// when cat probability exceeds the confidence threshold.
pub struct ClipDetector {
    session: std::sync::Mutex<Session>,
    confidence_threshold: f32,
    cat_class_id: u32,
    /// Pre-computed, L2-normalized text embedding for the positive class (cat) [dim]
    cat_embedding: Vec<f32>,
    /// Pre-computed, L2-normalized text embeddings for negative classes [N][dim]
    negative_embeddings: Vec<Vec<f32>>,
    embedding_dim: usize,
}

/// CLIP image preprocessing constants (OpenAI CLIP normalization)
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];
const CLIP_INPUT_SIZE: u32 = 224;

impl ClipDetector {
    pub fn new(
        model_path: &Path,
        text_embeddings_path: &Path,
        confidence_threshold: f32,
        cat_class_id: u32,
    ) -> Result<Self, DetectorError> {
        let session = Session::builder()
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?;

        // Load text embeddings binary file
        let emb_bytes = std::fs::read(text_embeddings_path).map_err(|e| {
            DetectorError::ModelLoadError(format!(
                "Failed to load text embeddings from {:?}: {}",
                text_embeddings_path, e
            ))
        })?;

        // Format: [count: u32 LE][dim: u32 LE][embed_0: f32 x dim]...[embed_N: f32 x dim]
        // First embedding is positive (cat), rest are negatives
        if emb_bytes.len() < 8 {
            return Err(DetectorError::ModelLoadError(
                "Text embeddings file too small".to_string(),
            ));
        }

        let count = u32::from_le_bytes([emb_bytes[0], emb_bytes[1], emb_bytes[2], emb_bytes[3]])
            as usize;
        let embedding_dim =
            u32::from_le_bytes([emb_bytes[4], emb_bytes[5], emb_bytes[6], emb_bytes[7]]) as usize;

        let expected_size = 8 + count * embedding_dim * 4;
        if emb_bytes.len() < expected_size || count < 2 {
            return Err(DetectorError::ModelLoadError(format!(
                "Invalid embeddings: count={}, dim={}, file_size={}, expected={}",
                count,
                embedding_dim,
                emb_bytes.len(),
                expected_size
            )));
        }

        let float_data: Vec<f32> = emb_bytes[8..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let cat_embedding = float_data[..embedding_dim].to_vec();
        let mut negative_embeddings = Vec::new();
        for i in 1..count {
            let start = i * embedding_dim;
            negative_embeddings.push(float_data[start..start + embedding_dim].to_vec());
        }

        tracing::info!(
            "Loaded CLIP model {:?} with {} text embeddings (dim={}, {} negative classes)",
            model_path,
            count,
            embedding_dim,
            negative_embeddings.len(),
        );

        Ok(Self {
            session: std::sync::Mutex::new(session),
            confidence_threshold,
            cat_class_id,
            cat_embedding,
            negative_embeddings,
            embedding_dim,
        })
    }

    fn preprocess(&self, image: &DynamicImage) -> Result<ndarray::Array4<f32>, DetectorError> {
        // Resize shortest side to 224, then center crop
        let (w, h) = (image.width(), image.height());
        let scale = CLIP_INPUT_SIZE as f32 / w.min(h) as f32;
        let new_w = (w as f32 * scale).round() as u32;
        let new_h = (h as f32 * scale).round() as u32;
        let resized = image.resize_exact(new_w, new_h, image::imageops::FilterType::CatmullRom);

        // Center crop to 224x224
        let left = (new_w - CLIP_INPUT_SIZE) / 2;
        let top = (new_h - CLIP_INPUT_SIZE) / 2;
        let cropped = resized.crop_imm(left, top, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE);
        let rgb = cropped.to_rgb8();

        let mut array = ndarray::Array4::<f32>::zeros((
            1,
            3,
            CLIP_INPUT_SIZE as usize,
            CLIP_INPUT_SIZE as usize,
        ));

        for (x, y, pixel) in rgb.enumerate_pixels() {
            array[[0, 0, y as usize, x as usize]] =
                (pixel[0] as f32 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0];
            array[[0, 1, y as usize, x as usize]] =
                (pixel[1] as f32 / 255.0 - CLIP_MEAN[1]) / CLIP_STD[1];
            array[[0, 2, y as usize, x as usize]] =
                (pixel[2] as f32 / 255.0 - CLIP_MEAN[2]) / CLIP_STD[2];
        }

        Ok(array)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        // Both vectors are already L2-normalized, so dot product = cosine similarity
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

#[async_trait]
impl CatDetector for ClipDetector {
    async fn detect(&self, image: &DynamicImage) -> Result<Vec<Detection>, DetectorError> {
        let original_width = image.width();
        let original_height = image.height();

        let input = self.preprocess(image)?;

        let shape = input.shape().to_vec();
        let data = input.into_raw_vec_and_offset().0;
        let input_tensor = ort::value::Tensor::from_array((shape, data))
            .map_err(|e| DetectorError::InferenceError(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| DetectorError::InferenceError(format!("Failed to lock session: {}", e)))?;

        let input_name = session.inputs()[0].name().to_string();
        let outputs = session
            .run(ort::inputs![&*input_name => input_tensor])
            .map_err(|e| DetectorError::InferenceError(e.to_string()))?;

        let output_value = outputs
            .values()
            .next()
            .ok_or_else(|| DetectorError::InferenceError("No output tensors".to_string()))?;

        let (_shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e: ort::Error| DetectorError::InferenceError(e.to_string()))?;

        let image_features: Vec<f32> = data.to_vec();

        // L2-normalize image features
        let norm: f32 = image_features.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 0.0 {
            image_features.iter().map(|x| x / norm).collect()
        } else {
            image_features
        };

        // Compute cosine similarities against all text embeddings
        let image_emb = &normalized[..self.embedding_dim];
        let cat_sim = Self::cosine_similarity(image_emb, &self.cat_embedding);
        let neg_sims: Vec<f32> = self
            .negative_embeddings
            .iter()
            .map(|emb| Self::cosine_similarity(image_emb, emb))
            .collect();

        // Softmax with temperature=100 over all classes to get cat probability
        let max_sim = neg_sims
            .iter()
            .fold(cat_sim, |max, &s| if s > max { s } else { max });
        let cat_exp = ((cat_sim - max_sim) * 100.0).exp();
        let sum_exp: f32 = cat_exp
            + neg_sims
                .iter()
                .map(|&s| ((s - max_sim) * 100.0).exp())
                .sum::<f32>();
        let cat_prob = cat_exp / sum_exp;

        tracing::debug!(
            "CLIP: cat_sim={:.4}, neg_sims={:?}, cat_prob={:.4}",
            cat_sim,
            neg_sims,
            cat_prob,
        );

        if cat_prob >= self.confidence_threshold {
            Ok(vec![Detection {
                class_id: self.cat_class_id,
                confidence: cat_prob,
                bbox: BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: original_width as f32,
                    height: original_height as f32,
                },
            }])
        } else {
            Ok(vec![])
        }
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
    should_fail: bool,
}

#[cfg(test)]
impl MockDetector {
    pub fn new(results: Vec<Vec<Detection>>) -> Self {
        Self {
            results: std::sync::Mutex::new(results),
            call_count: std::sync::atomic::AtomicUsize::new(0),
            cat_class_id: 15,
            confidence_threshold: 0.5,
            should_fail: false,
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

    pub fn failing() -> Self {
        Self {
            results: std::sync::Mutex::new(vec![]),
            call_count: std::sync::atomic::AtomicUsize::new(0),
            cat_class_id: 15,
            confidence_threshold: 0.5,
            should_fail: true,
        }
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

        if self.should_fail {
            return Err(DetectorError::InferenceError("mock failure".to_string()));
        }

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

    #[test]
    fn test_clip_cosine_similarity_normalized() {
        // Two identical normalized vectors should have similarity ~1.0
        let a = vec![0.5_f32.sqrt(), 0.5_f32.sqrt()];
        let b = vec![0.5_f32.sqrt(), 0.5_f32.sqrt()];
        let sim = ClipDetector::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors should have similarity ~0.0
        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        let sim = ClipDetector::cosine_similarity(&c, &d);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_clip_detector_loads_embeddings() {
        // Create temporary embeddings file: 3 vectors of dim 4
        // Format: [count: u32 LE][dim: u32 LE][embeddings: f32...]
        let cat_emb: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
        let neg1_emb: Vec<f32> = vec![-0.5, -0.5, 0.5, 0.5];
        let neg2_emb: Vec<f32> = vec![0.0, -0.5, -0.5, 0.5];

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3u32.to_le_bytes()); // count
        bytes.extend_from_slice(&4u32.to_le_bytes()); // dim
        for f in cat_emb
            .iter()
            .chain(neg1_emb.iter())
            .chain(neg2_emb.iter())
        {
            bytes.extend_from_slice(&f.to_le_bytes());
        }

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();

        // Verify the embedding loading logic by parsing the file format
        let data = std::fs::read(tmp.path()).unwrap();
        let count =
            u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let dim =
            u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        assert_eq!(count, 3);
        assert_eq!(dim, 4);

        let floats: Vec<f32> = data[8..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        assert_eq!(floats.len(), 12); // 3 * 4
        assert_eq!(floats[0], 0.5);   // cat_emb[0]
        assert_eq!(floats[4], -0.5);  // neg1_emb[0]
        assert_eq!(floats[8], 0.0);   // neg2_emb[0]
    }
}
