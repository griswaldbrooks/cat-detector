//! Integration tests for CLIP ViT-B/32 cat detection.
//!
//! These tests require:
//! - ONNX Runtime library (set ORT_DYLIB_PATH environment variable)
//! - The CLIP model at models/clip_vitb32_image.onnx
//! - Embeddings files in models/ (text and/or image)
//! - Test images in test_images/
//!
//! Run with: ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so cargo test --test clip_integration_test

use std::path::Path;

const CLIP_MODEL_PATH: &str = "models/clip_vitb32_image.onnx";
const CLIP_TEXT_EMBEDDINGS_PATH: &str = "models/clip_text_embeddings.bin";
const CLIP_IMAGE_EMBEDDINGS_PATH: &str = "models/clip_image_embeddings.bin";

fn clip_available() -> bool {
    Path::new(CLIP_MODEL_PATH).exists()
        && Path::new(CLIP_TEXT_EMBEDDINGS_PATH).exists()
        && std::env::var("ORT_DYLIB_PATH").is_ok()
}

fn image_embeddings_available() -> bool {
    clip_available() && Path::new(CLIP_IMAGE_EMBEDDINGS_PATH).exists()
}

fn make_clip_detector(threshold: f32) -> cat_detector::detector::ClipDetector {
    cat_detector::detector::ClipDetector::new(
        Path::new(CLIP_MODEL_PATH),
        Path::new(CLIP_TEXT_EMBEDDINGS_PATH),
        threshold,
        15,
    )
    .expect("Failed to load CLIP model")
}

fn make_fewshot_detector(threshold: f32) -> cat_detector::detector::ClipDetector {
    cat_detector::detector::ClipDetector::new(
        Path::new(CLIP_MODEL_PATH),
        Path::new(CLIP_IMAGE_EMBEDDINGS_PATH),
        threshold,
        15,
    )
    .expect("Failed to load CLIP model with image embeddings")
}

macro_rules! clip_test {
    ($name:ident, $body:expr) => {
        #[tokio::test]
        async fn $name() {
            if !clip_available() {
                eprintln!(
                    "Skipping {}: CLIP model or ORT_DYLIB_PATH not available",
                    stringify!($name)
                );
                return;
            }
            $body
        }
    };
}

mod clip_cat_detection {
    use super::*;
    use cat_detector::detector::CatDetector;

    // === Stock side-angle cat images ===

    clip_test!(test_clip_cat1_stock, {
        let detector = make_clip_detector(0.5);
        let img = image::open("test_images/cat1.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(!detections.is_empty(), "Should detect cat in cat1.jpg");
        assert!(
            detections[0].confidence > 0.9,
            "Expected >90% confidence, got {:.1}%",
            detections[0].confidence * 100.0
        );
    });

    clip_test!(test_clip_cat2_stock, {
        let detector = make_clip_detector(0.5);
        let img = image::open("test_images/cat2.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(!detections.is_empty(), "Should detect cat in cat2.jpg");
        assert!(
            detections[0].confidence > 0.9,
            "Expected >90% confidence, got {:.1}%",
            detections[0].confidence * 100.0
        );
    });

    // === Overhead light cat (primary deployment scenario) ===

    clip_test!(test_clip_overhead_light_center, {
        let detector = make_clip_detector(0.5);
        let img = image::open("test_images/cat_overhead_center.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            !detections.is_empty(),
            "Should detect cat in overhead center view"
        );
        assert!(
            detections[0].confidence > 0.95,
            "Expected >95% confidence for overhead light cat, got {:.1}%",
            detections[0].confidence * 100.0
        );
    });

    clip_test!(test_clip_overhead_light_walking, {
        let detector = make_clip_detector(0.5);
        let img =
            image::open("test_images/cat_overhead_walking1.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            !detections.is_empty(),
            "Should detect walking cat from overhead"
        );
        assert!(
            detections[0].confidence > 0.95,
            "Expected >95% confidence, got {:.1}%",
            detections[0].confidence * 100.0
        );
    });

    clip_test!(test_clip_overhead_light_at_litterbot, {
        let detector = make_clip_detector(0.5);
        let img =
            image::open("test_images/cat_overhead_litterbot1.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            !detections.is_empty(),
            "Should detect cat at litter robot from overhead"
        );
        assert!(
            detections[0].confidence > 0.95,
            "Expected >95% confidence, got {:.1}%",
            detections[0].confidence * 100.0
        );
    });

    // === Overhead tabby cat ===

    clip_test!(test_clip_overhead_tabby, {
        let detector = make_clip_detector(0.5);
        let img = image::open("test_images/cat_overhead_tabby1.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            !detections.is_empty(),
            "Should detect tabby cat from overhead"
        );
        assert!(
            detections[0].confidence > 0.95,
            "Expected >95% confidence for tabby, got {:.1}%",
            detections[0].confidence * 100.0
        );
    });

    clip_test!(test_clip_tabby_entering_litterbot, {
        let detector = make_clip_detector(0.5);
        let img = image::open("test_images/cat_tabby_entering_litterbot1.jpg")
            .expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            !detections.is_empty(),
            "Should detect tabby entering litter robot"
        );
        // This image scores ~80% — cat is partially occluded entering the litter robot
        assert!(
            detections[0].confidence > 0.7,
            "Expected >70% confidence, got {:.1}%",
            detections[0].confidence * 100.0
        );
    });

    // === Cat inside litter robot (harder, partially visible) ===

    clip_test!(test_clip_inside_litterbot, {
        let detector = make_clip_detector(0.5);
        let img =
            image::open("test_images/cat_inside_litterbot1.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            !detections.is_empty(),
            "Should detect cat inside litter robot"
        );
        assert!(
            detections[0].confidence > 0.9,
            "Expected >90% confidence for cat inside litterbot, got {:.1}%",
            detections[0].confidence * 100.0
        );
    });
}

mod clip_negative_detection {
    use super::*;
    use cat_detector::detector::CatDetector;

    // === Empty rooms (should NOT detect cats) ===

    clip_test!(test_clip_no_cat_overhead, {
        let detector = make_clip_detector(0.3);
        let img = image::open("test_images/no_cat_overhead.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            detections.is_empty(),
            "Should not detect cat in empty overhead view"
        );
    });

    clip_test!(test_clip_no_cat_room, {
        let detector = make_clip_detector(0.3);
        let img = image::open("test_images/no_cat_room.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(detections.is_empty(), "Should not detect cat in empty room");
    });

    clip_test!(test_clip_no_cat2, {
        let detector = make_clip_detector(0.3);
        let img = image::open("test_images/no_cat2.jpeg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            detections.is_empty(),
            "Should not detect cat in no_cat2.jpeg"
        );
    });

    // === Person rejection ===

    clip_test!(test_clip_person_not_detected_as_cat, {
        let detector = make_clip_detector(0.3);
        let img = image::open("test_images/person_images/person_overhead_1.jpg")
            .expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(detections.is_empty(), "Should not detect person as cat");
    });

    clip_test!(test_clip_person2_not_detected_as_cat, {
        let detector = make_clip_detector(0.3);
        let img = image::open("test_images/person_images/person_overhead_2.jpg")
            .expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(detections.is_empty(), "Should not detect person as cat");
    });
}

// === Few-shot image prototype embedding tests ===
// These use image embeddings (models/clip_image_embeddings.bin) instead of text embeddings.
// The 3 false positive edge cases are fixed; cat+litter box remains a known limitation.

mod clip_fewshot_detection {
    use super::*;
    use cat_detector::detector::CatDetector;

    // Verify few-shot still detects cats correctly

    #[tokio::test]
    async fn test_fewshot_detects_overhead_cat() {
        if !image_embeddings_available() {
            eprintln!("Skipping: image embeddings not available");
            return;
        }
        let detector = make_fewshot_detector(0.5);
        let img = image::open("test_images/cat_overhead_center.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            !detections.is_empty(),
            "Few-shot should detect cat in overhead view"
        );
        assert!(
            detections[0].confidence > 0.95,
            "Expected >95% confidence, got {:.1}%",
            detections[0].confidence * 100.0
        );
    }

    #[tokio::test]
    async fn test_fewshot_detects_overhead_tabby() {
        if !image_embeddings_available() {
            eprintln!("Skipping: image embeddings not available");
            return;
        }
        let detector = make_fewshot_detector(0.5);
        let img = image::open("test_images/cat_overhead_tabby1.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            !detections.is_empty(),
            "Few-shot should detect tabby from overhead"
        );
    }

    #[tokio::test]
    async fn test_fewshot_detects_stock_cat() {
        if !image_embeddings_available() {
            eprintln!("Skipping: image embeddings not available");
            return;
        }
        let detector = make_fewshot_detector(0.5);
        let img = image::open("test_images/cat1.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            !detections.is_empty(),
            "Few-shot should still detect stock cat photo"
        );
    }

    #[tokio::test]
    async fn test_fewshot_rejects_empty_room() {
        if !image_embeddings_available() {
            eprintln!("Skipping: image embeddings not available");
            return;
        }
        let detector = make_fewshot_detector(0.5);
        let img = image::open("test_images/no_cat_overhead.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            detections.is_empty(),
            "Few-shot should not detect cat in empty room"
        );
    }

    // === Fixed edge cases ===

    #[tokio::test]
    async fn test_fewshot_rejects_dirty_litter_box() {
        if !image_embeddings_available() {
            eprintln!("Skipping: image embeddings not available");
            return;
        }
        let detector = make_fewshot_detector(0.5);
        let img = image::open("test_images/litter_box_dirty_overhead_1.jpg")
            .expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            detections.is_empty(),
            "Few-shot should NOT detect dirty litter box as cat"
        );
    }

    #[tokio::test]
    async fn test_fewshot_rejects_litter_robot_moving() {
        if !image_embeddings_available() {
            eprintln!("Skipping: image embeddings not available");
            return;
        }
        let detector = make_fewshot_detector(0.5);
        let img = image::open("test_images/litter_robot_moving_overhead_1.jpg")
            .expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            detections.is_empty(),
            "Few-shot should NOT detect moving litter robot as cat"
        );
    }

    #[tokio::test]
    async fn test_fewshot_rejects_person_overhead_catbox() {
        if !image_embeddings_available() {
            eprintln!("Skipping: image embeddings not available");
            return;
        }
        let detector = make_fewshot_detector(0.5);
        let img =
            image::open("test_images/person_overhead_catbox_1.jpg").expect("Failed to load image");
        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(
            detections.is_empty(),
            "Few-shot should NOT detect overhead person as cat"
        );
    }

    // Note: test_images/cat_with_litter_box_overhead_1.jpg was mislabeled — it contains
    // only a litter box and robot, no cat. Renamed to litter_box_and_robot_overhead_1.jpg.
    // The "known limitation" test that was here has been removed.
}

mod clip_performance {
    use super::*;
    use cat_detector::detector::CatDetector;

    clip_test!(test_clip_inference_speed, {
        let detector = make_clip_detector(0.5);
        let img = image::open("test_images/cat1.jpg").expect("Failed to load image");

        // Warm up
        let _ = detector.detect(&img).await;

        // Measure
        let start = std::time::Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = detector.detect(&img).await;
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;

        println!("CLIP average inference time: {:.1}ms", avg_ms);
        // Debug builds are slower; release builds should be ~20ms on i3-6100T
        assert!(
            avg_ms < 1000.0,
            "Inference should be < 1000ms (debug), got {:.1}ms",
            avg_ms
        );
    });

    clip_test!(test_clip_detection_returns_full_frame_bbox, {
        let detector = make_clip_detector(0.5);
        let img = image::open("test_images/cat1.jpg").expect("Failed to load image");
        let img_width = img.width() as f32;
        let img_height = img.height() as f32;

        let detections = detector.detect(&img).await.expect("Detection failed");
        assert!(!detections.is_empty(), "Should detect cat");

        let det = &detections[0];
        assert_eq!(det.bbox.x, 0.0, "CLIP bbox should be full-frame (x=0)");
        assert_eq!(det.bbox.y, 0.0, "CLIP bbox should be full-frame (y=0)");
        assert_eq!(
            det.bbox.width, img_width,
            "CLIP bbox width should match image"
        );
        assert_eq!(
            det.bbox.height, img_height,
            "CLIP bbox height should match image"
        );
        assert_eq!(det.class_id, 15, "Cat class_id should be 15 (COCO)");
    });
}
