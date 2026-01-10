//! Integration tests for the cat detector using real ONNX model and test images.
//!
//! These tests require:
//! - ONNX Runtime library (set ORT_DYLIB_PATH environment variable)
//! - The YOLOX model at models/yolox_s.onnx
//! - Test images in test_images/
//!
//! Run with: ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so cargo test --test integration_test

use std::path::Path;

const MODEL_PATH: &str = "models/yolox_s.onnx";
const MODEL_INPUT_SIZE: u32 = 640;

fn model_available() -> bool {
    Path::new(MODEL_PATH).exists() && std::env::var("ORT_DYLIB_PATH").is_ok()
}

mod detector_integration {
    use super::*;

    macro_rules! integration_test {
        ($name:ident, $body:expr) => {
            #[tokio::test]
            async fn $name() {
                if !model_available() {
                    eprintln!(
                        "Skipping {}: model or ORT_DYLIB_PATH not available",
                        stringify!($name)
                    );
                    return;
                }
                $body
            }
        };
    }

    // === Cat images (should detect cats) ===

    integration_test!(test_cat1_detected, {
        let detector = cat_detector::detector::OnnxDetector::new_with_size(
            Path::new(MODEL_PATH),
            0.5,
            15,
            MODEL_INPUT_SIZE,
        )
        .expect("Failed to load model");

        let img = image::open("test_images/cat1.jpg").expect("Failed to load test image");

        use cat_detector::detector::CatDetector;
        let detections = detector.detect(&img).await.expect("Detection failed");

        let cats: Vec<_> = detections.iter().filter(|d| detector.is_cat(d)).collect();
        assert!(!cats.is_empty(), "Should detect cat in cat1.jpg");
        assert!(
            cats[0].confidence > 0.7,
            "Cat confidence should be > 70%, got {:.1}%",
            cats[0].confidence * 100.0
        );
    });

    integration_test!(test_cat2_detected, {
        let detector = cat_detector::detector::OnnxDetector::new_with_size(
            Path::new(MODEL_PATH),
            0.5,
            15,
            MODEL_INPUT_SIZE,
        )
        .expect("Failed to load model");

        let img = image::open("test_images/cat2.jpg").expect("Failed to load test image");

        use cat_detector::detector::CatDetector;
        let detections = detector.detect(&img).await.expect("Detection failed");

        let cats: Vec<_> = detections.iter().filter(|d| detector.is_cat(d)).collect();
        assert!(!cats.is_empty(), "Should detect cat in cat2.jpg");
        assert!(
            cats[0].confidence > 0.7,
            "Cat confidence should be > 70%, got {:.1}%",
            cats[0].confidence * 100.0
        );
    });

    integration_test!(test_cat3_detected, {
        // cat3.jpeg - yolox_s detects this at ~66% confidence
        let detector = cat_detector::detector::OnnxDetector::new_with_size(
            Path::new(MODEL_PATH),
            0.5,
            15,
            MODEL_INPUT_SIZE,
        )
        .expect("Failed to load model");

        let img = image::open("test_images/cat3.jpeg").expect("Failed to load test image");

        use cat_detector::detector::CatDetector;
        let detections = detector.detect(&img).await.expect("Detection failed");

        let cats: Vec<_> = detections.iter().filter(|d| detector.is_cat(d)).collect();
        assert!(!cats.is_empty(), "Should detect cat in cat3.jpeg");
        assert!(
            cats[0].confidence > 0.5,
            "Cat confidence should be > 50%, got {:.1}%",
            cats[0].confidence * 100.0
        );
    });

    integration_test!(test_cat4_detected, {
        // cat4.jpeg - yolox_s detects this at ~77% confidence
        let detector = cat_detector::detector::OnnxDetector::new_with_size(
            Path::new(MODEL_PATH),
            0.5,
            15,
            MODEL_INPUT_SIZE,
        )
        .expect("Failed to load model");

        let img = image::open("test_images/cat4.jpeg").expect("Failed to load test image");

        use cat_detector::detector::CatDetector;
        let detections = detector.detect(&img).await.expect("Detection failed");

        let cats: Vec<_> = detections.iter().filter(|d| detector.is_cat(d)).collect();
        assert!(!cats.is_empty(), "Should detect cat in cat4.jpeg");
        assert!(
            cats[0].confidence > 0.7,
            "Cat confidence should be > 70%, got {:.1}%",
            cats[0].confidence * 100.0
        );
    });

    integration_test!(test_cat5_detected, {
        let detector = cat_detector::detector::OnnxDetector::new_with_size(
            Path::new(MODEL_PATH),
            0.5,
            15,
            MODEL_INPUT_SIZE,
        )
        .expect("Failed to load model");

        let img = image::open("test_images/cat5.jpeg").expect("Failed to load test image");

        use cat_detector::detector::CatDetector;
        let detections = detector.detect(&img).await.expect("Detection failed");

        let cats: Vec<_> = detections.iter().filter(|d| detector.is_cat(d)).collect();
        assert!(!cats.is_empty(), "Should detect cat in cat5.jpeg");
        assert!(
            cats[0].confidence > 0.5,
            "Cat confidence should be > 50%, got {:.1}%",
            cats[0].confidence * 100.0
        );
    });

    // === No-cat images (should NOT detect cats) ===

    integration_test!(test_no_cat2_no_detection, {
        let detector = cat_detector::detector::OnnxDetector::new_with_size(
            Path::new(MODEL_PATH),
            0.3,
            15,
            MODEL_INPUT_SIZE,
        )
        .expect("Failed to load model");

        let img = image::open("test_images/no_cat2.jpeg").expect("Failed to load test image");

        use cat_detector::detector::CatDetector;
        let detections = detector.detect(&img).await.expect("Detection failed");

        let cats: Vec<_> = detections.iter().filter(|d| detector.is_cat(d)).collect();
        assert!(
            cats.is_empty(),
            "Should not detect cats in no_cat2.jpeg (bathroom without cat)"
        );
    });

    integration_test!(test_no_cat3_no_detection, {
        let detector = cat_detector::detector::OnnxDetector::new_with_size(
            Path::new(MODEL_PATH),
            0.3,
            15,
            MODEL_INPUT_SIZE,
        )
        .expect("Failed to load model");

        let img = image::open("test_images/no_cat3.jpeg").expect("Failed to load test image");

        use cat_detector::detector::CatDetector;
        let detections = detector.detect(&img).await.expect("Detection failed");

        let cats: Vec<_> = detections.iter().filter(|d| detector.is_cat(d)).collect();
        assert!(
            cats.is_empty(),
            "Should not detect cats in no_cat3.jpeg (bathroom without cat)"
        );
    });

    // === Performance and sanity tests ===

    integration_test!(test_inference_speed, {
        let detector = cat_detector::detector::OnnxDetector::new_with_size(
            Path::new(MODEL_PATH),
            0.5,
            15,
            MODEL_INPUT_SIZE,
        )
        .expect("Failed to load model");

        let img = image::open("test_images/cat1.jpg").expect("Failed to load test image");

        use cat_detector::detector::CatDetector;

        // Warm up
        let _ = detector.detect(&img).await;

        // Measure
        let start = std::time::Instant::now();
        let iterations = 5;
        for _ in 0..iterations {
            let _ = detector.detect(&img).await;
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;

        println!("Average inference time: {:.1}ms", avg_ms);
        // Debug builds are slower; release builds should be < 50ms
        assert!(
            avg_ms < 1000.0,
            "Inference should be < 1000ms (debug) or < 50ms (release), got {:.1}ms",
            avg_ms
        );
    });

    integration_test!(test_bounding_box_sanity, {
        let detector = cat_detector::detector::OnnxDetector::new_with_size(
            Path::new(MODEL_PATH),
            0.5,
            15,
            MODEL_INPUT_SIZE,
        )
        .expect("Failed to load model");

        let img = image::open("test_images/cat1.jpg").expect("Failed to load test image");
        let img_width = img.width() as f32;
        let img_height = img.height() as f32;

        use cat_detector::detector::CatDetector;
        let detections = detector.detect(&img).await.expect("Detection failed");

        for det in &detections {
            assert!(
                det.bbox.x >= -10.0 && det.bbox.x < img_width,
                "bbox.x out of bounds: {}",
                det.bbox.x
            );
            assert!(
                det.bbox.y >= -10.0 && det.bbox.y < img_height,
                "bbox.y out of bounds: {}",
                det.bbox.y
            );
            assert!(
                det.bbox.width > 0.0 && det.bbox.width <= img_width * 1.1,
                "bbox.width invalid: {}",
                det.bbox.width
            );
            assert!(
                det.bbox.height > 0.0 && det.bbox.height <= img_height * 1.1,
                "bbox.height invalid: {}",
                det.bbox.height
            );
        }
    });
}
