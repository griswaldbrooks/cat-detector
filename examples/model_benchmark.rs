//! Benchmark multiple YOLO models against test images.
//!
//! Run with:
//!   ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so cargo run --example model_benchmark --release

use image::DynamicImage;
use ndarray::{Array2, Array4};
use ort::session::Session;
use std::path::Path;
use std::time::Instant;

const CAT_CLASS_ID: usize = 15;
const INPUT_SIZE: u32 = 640;
const CONF_THRESHOLD: f32 = 0.1; // Low threshold to see all detections

struct ModelResult {
    model_name: String,
    image_name: String,
    cat_confidence: Option<f32>,
    all_detections: usize,
    inference_ms: f64,
}

fn preprocess_yolox(image: &DynamicImage) -> Array4<f32> {
    let resized = image.resize_exact(
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();
    let mut array = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    // YOLOX: pixel values 0-255
    for (x, y, pixel) in rgb.enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = pixel[0] as f32;
        array[[0, 1, y as usize, x as usize]] = pixel[1] as f32;
        array[[0, 2, y as usize, x as usize]] = pixel[2] as f32;
    }
    array
}

fn preprocess_yolo11(image: &DynamicImage) -> Array4<f32> {
    let resized = image.resize_exact(
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();
    let mut array = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    // YOLO11: pixel values 0-1 (normalized)
    for (x, y, pixel) in rgb.enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
        array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
        array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
    }
    array
}

fn postprocess_yolox(flat_data: &[f32]) -> (Option<f32>, usize) {
    let num_boxes = flat_data.len() / 85;
    let array = Array2::from_shape_vec((num_boxes, 85), flat_data.to_vec()).unwrap();

    let strides = [8u32, 16, 32];
    let mut grids = Vec::new();
    for &stride in &strides {
        let grid_size = INPUT_SIZE / stride;
        for gy in 0..grid_size {
            for gx in 0..grid_size {
                grids.push((gx as f32, gy as f32, stride as f32));
            }
        }
    }

    let mut best_cat_conf: Option<f32> = None;
    let mut total_detections = 0;

    for (i, row) in array.rows().into_iter().enumerate() {
        if i >= grids.len() {
            break;
        }
        let (_gx, _gy, _stride) = grids[i];
        let objectness = row[4];
        let row_slice = row.as_slice().unwrap();
        let class_scores = &row_slice[5..];
        let (class_id, &class_score) = class_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();

        let confidence = objectness * class_score;
        if confidence >= CONF_THRESHOLD {
            total_detections += 1;
            if class_id == CAT_CLASS_ID {
                best_cat_conf = Some(best_cat_conf.map_or(confidence, |c: f32| c.max(confidence)));
            }
        }
    }
    (best_cat_conf, total_detections)
}

fn postprocess_yolo11(flat_data: &[f32], output_shape: &[usize]) -> (Option<f32>, usize) {
    // YOLO11 output: [1, 84, 8400] -> transpose to [8400, 84]
    // Each column: [x_center, y_center, width, height, class0..class79]
    let num_classes = output_shape[1] - 4; // 84 - 4 = 80
    let num_boxes = output_shape[2]; // 8400

    let mut best_cat_conf: Option<f32> = None;
    let mut total_detections = 0;

    for box_idx in 0..num_boxes {
        // Find max class score for this box
        let mut max_score: f32 = 0.0;
        let mut max_class: usize = 0;
        for c in 0..num_classes {
            let score = flat_data[(4 + c) * num_boxes + box_idx];
            if score > max_score {
                max_score = score;
                max_class = c;
            }
        }

        if max_score >= CONF_THRESHOLD {
            total_detections += 1;
            if max_class == CAT_CLASS_ID {
                best_cat_conf = Some(best_cat_conf.map_or(max_score, |c: f32| c.max(max_score)));
            }
        }
    }
    (best_cat_conf, total_detections)
}

fn run_model(
    session: &mut Session,
    image: &DynamicImage,
    model_type: &str,
    input_name: &str,
) -> (Option<f32>, usize, f64) {
    let input = if model_type == "yolox" {
        preprocess_yolox(image)
    } else {
        preprocess_yolo11(image)
    };

    let shape = input.shape().to_vec();
    let data = input.into_raw_vec_and_offset().0;
    let input_tensor = ort::value::Tensor::from_array((shape, data)).unwrap();

    // Warm up
    let _ = session
        .run(ort::inputs![input_name => input_tensor])
        .unwrap();

    // Rerun for measurement
    let input2 = if model_type == "yolox" {
        preprocess_yolox(image)
    } else {
        preprocess_yolo11(image)
    };
    let shape2 = input2.shape().to_vec();
    let data2 = input2.into_raw_vec_and_offset().0;
    let input_tensor2 = ort::value::Tensor::from_array((shape2, data2)).unwrap();

    let start = Instant::now();
    let outputs = session
        .run(ort::inputs![input_name => input_tensor2])
        .unwrap();
    let inference_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Get first output regardless of name
    let output_value = outputs.values().next().unwrap();
    let (shape_info, data) = output_value.try_extract_tensor::<f32>().unwrap();
    let flat: Vec<f32> = data.to_vec();

    let shape: Vec<usize> = shape_info.iter().map(|&d| d as usize).collect();

    let (cat_conf, det_count) = if model_type == "yolox" {
        postprocess_yolox(&flat)
    } else {
        postprocess_yolo11(&flat, &shape)
    };

    (cat_conf, det_count, inference_ms)
}

fn main() {
    let models: Vec<(&str, &str, &str)> = vec![
        ("YOLOX-S", "models/yolox_s.onnx", "yolox"),
        ("YOLO11n", "models/yolo11n.onnx", "yolo11"),
        ("YOLO11s", "models/yolo11s.onnx", "yolo11"),
    ];

    let test_dir = Path::new("test_images");
    let mut images: Vec<String> = std::fs::read_dir(test_dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            if name.ends_with(".jpg") || name.ends_with(".jpeg") || name.ends_with(".png") {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    images.sort();

    let mut results: Vec<ModelResult> = Vec::new();

    for (model_name, model_path, model_type) in &models {
        if !Path::new(model_path).exists() {
            eprintln!("Skipping {}: model not found at {}", model_name, model_path);
            continue;
        }

        println!("Loading {}...", model_name);
        let mut session = Session::builder()
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        // Detect input name
        let input_name = session.inputs()[0].name().to_string();
        println!("  Input: {}", input_name);
        for output in session.outputs() {
            println!("  Output: {}", output.name());
        }

        for img_name in &images {
            let img_path = test_dir.join(img_name);
            let image = image::open(&img_path).expect("Failed to load image");

            let (cat_conf, det_count, inference_ms) =
                run_model(&mut session, &image, model_type, &input_name);

            results.push(ModelResult {
                model_name: model_name.to_string(),
                image_name: img_name.clone(),
                cat_confidence: cat_conf,
                all_detections: det_count,
                inference_ms,
            });
        }
    }

    // Print comparison table
    println!("\n{:=<100}", "");
    println!(
        "{:<25} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "Image", "YOLOX-S", "YOLO11n", "YOLO11s", "ms(YX-S)", "ms(11n)", "ms(11s)"
    );
    println!("{:-<100}", "");

    for img_name in &images {
        let yolox = results
            .iter()
            .find(|r| r.model_name == "YOLOX-S" && r.image_name == *img_name);
        let y11n = results
            .iter()
            .find(|r| r.model_name == "YOLO11n" && r.image_name == *img_name);
        let y11s = results
            .iter()
            .find(|r| r.model_name == "YOLO11s" && r.image_name == *img_name);

        let fmt_conf = |r: Option<&ModelResult>| -> String {
            match r {
                Some(r) => match r.cat_confidence {
                    Some(c) => format!("{:.1}%", c * 100.0),
                    None => "---".to_string(),
                },
                None => "N/A".to_string(),
            }
        };

        let fmt_ms = |r: Option<&ModelResult>| -> String {
            match r {
                Some(r) => format!("{:.0}", r.inference_ms),
                None => "N/A".to_string(),
            }
        };

        println!(
            "{:<25} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10}",
            img_name,
            fmt_conf(yolox),
            fmt_conf(y11n),
            fmt_conf(y11s),
            fmt_ms(yolox),
            fmt_ms(y11n),
            fmt_ms(y11s),
        );
    }
    println!("{:=<100}", "");
}
