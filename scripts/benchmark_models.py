#!/usr/bin/env python3
"""Benchmark detection/classification models on overhead cat test images.

Compares YOLO detectors, CLIP zero-shot classification, and DINOv2+linear probe
against the overhead camera test images.

Usage:
    python3 scripts/benchmark_models.py
    python3 scripts/benchmark_models.py --overhead-only
    python3 scripts/benchmark_models.py --train-dinov2-probe
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE_YOLO = 640
INPUT_SIZE_CLIP = 224
INPUT_SIZE_DINO = 224
CAT_CLASS_ID = 15  # COCO class for cat
CONF_THRESHOLD = 0.1  # Low to see all detections

MODELS_DIR = Path("models")
TEST_DIR = Path("test_images")


# ---------------------------------------------------------------------------
# Image categorization
# ---------------------------------------------------------------------------
def categorize_image(name: str) -> str:
    """Categorize test image by type."""
    if "no_cat" in name:
        return "no_cat"
    elif "overhead" in name or "litterbot" in name or "scratcher" in name or "eating" in name:
        if "tabby" in name:
            return "overhead_tabby"
        elif "inside" in name:
            return "overhead_inside"
        else:
            return "overhead_light"
    else:
        return "stock"


# ---------------------------------------------------------------------------
# YOLO preprocessing / postprocessing
# ---------------------------------------------------------------------------
def preprocess_yolo(image: Image.Image, normalize: bool = True) -> np.ndarray:
    """Preprocess for YOLO models. normalize=True for YOLO11/v8, False for YOLOX."""
    img = image.resize((INPUT_SIZE_YOLO, INPUT_SIZE_YOLO), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    if not normalize:
        # YOLOX: 0-255
        arr = arr.transpose(2, 0, 1)[np.newaxis]
    else:
        # YOLO11/v8: 0-1
        arr = (arr / 255.0).transpose(2, 0, 1)[np.newaxis]
    return arr


def postprocess_yolo11(output: np.ndarray) -> tuple[float | None, int]:
    """Postprocess YOLO11/v8 output [1, 84, 8400]."""
    data = output[0]  # [84, 8400]
    scores = data[4:]  # [80, 8400] - class scores
    max_scores = scores.max(axis=0)  # [8400]
    max_classes = scores.argmax(axis=0)  # [8400]

    mask = max_scores >= CONF_THRESHOLD
    total = int(mask.sum())

    cat_mask = mask & (max_classes == CAT_CLASS_ID)
    if cat_mask.any():
        best_cat = float(max_scores[cat_mask].max())
    else:
        best_cat = None

    return best_cat, total


def postprocess_yolox(output: np.ndarray) -> tuple[float | None, int]:
    """Postprocess YOLOX output [1, N, 85]."""
    data = output[0]  # [N, 85]
    objectness = data[:, 4]
    class_scores = data[:, 5:]
    max_class_scores = class_scores.max(axis=1)
    max_classes = class_scores.argmax(axis=1)
    confidence = objectness * max_class_scores

    mask = confidence >= CONF_THRESHOLD
    total = int(mask.sum())

    cat_mask = mask & (max_classes == CAT_CLASS_ID)
    if cat_mask.any():
        best_cat = float(confidence[cat_mask].max())
    else:
        best_cat = None

    return best_cat, total


# ---------------------------------------------------------------------------
# CLIP preprocessing
# ---------------------------------------------------------------------------
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_clip(image: Image.Image) -> np.ndarray:
    """Preprocess for CLIP: resize, center crop 224, normalize."""
    # Resize shortest side to 224, then center crop
    w, h = image.size
    scale = INPUT_SIZE_CLIP / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = image.resize((new_w, new_h), Image.BICUBIC)

    # Center crop
    left = (new_w - INPUT_SIZE_CLIP) // 2
    top = (new_h - INPUT_SIZE_CLIP) // 2
    img = img.crop((left, top, left + INPUT_SIZE_CLIP, top + INPUT_SIZE_CLIP))

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr


# ---------------------------------------------------------------------------
# DINOv2 preprocessing
# ---------------------------------------------------------------------------
DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_dino(image: Image.Image) -> np.ndarray:
    """Preprocess for DINOv2: resize to 224x224, ImageNet normalize."""
    img = image.resize((INPUT_SIZE_DINO, INPUT_SIZE_DINO), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - DINO_MEAN) / DINO_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------
def run_yolo_benchmark(model_name: str, model_path: str, model_type: str,
                       images: list[tuple[str, Image.Image]]) -> list[dict]:
    """Run YOLO model on all images, return results."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    normalize = model_type != "yolox"
    postprocess = postprocess_yolo11 if model_type != "yolox" else postprocess_yolox

    results = []
    for img_name, img in images:
        inp = preprocess_yolo(img, normalize=normalize)

        # Warmup
        session.run(None, {input_name: inp})

        # Timed run
        start = time.perf_counter()
        output = session.run(None, {input_name: inp})
        elapsed_ms = (time.perf_counter() - start) * 1000

        cat_conf, total_det = postprocess(output[0])
        results.append({
            "model": model_name,
            "image": img_name,
            "cat_confidence": cat_conf,
            "total_detections": total_det,
            "inference_ms": elapsed_ms,
            "category": categorize_image(img_name),
        })

    return results


def run_clip_benchmark(images: list[tuple[str, Image.Image]]) -> list[dict]:
    """Run CLIP ViT-B/32 zero-shot classification on all images."""
    model_path = str(MODELS_DIR / "clip_vitb32_image.onnx")
    if not os.path.exists(model_path):
        print("  CLIP model not found, skipping")
        return []

    import torch
    text_emb = torch.load(str(MODELS_DIR / "clip_text_embeddings.pt"),
                          map_location="cpu", weights_only=True).numpy()
    # text_emb[0] = "a photo of a cat", text_emb[1] = "a photo of an empty room"

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    results = []
    for img_name, img in images:
        inp = preprocess_clip(img)

        # Warmup
        session.run(None, {input_name: inp})

        # Timed run
        start = time.perf_counter()
        output = session.run(None, {input_name: inp})
        elapsed_ms = (time.perf_counter() - start) * 1000

        image_features = output[0]  # [1, 512]
        # Cosine similarity with text embeddings (already normalized)
        similarities = (image_features @ text_emb.T)[0]  # [2]

        # Softmax to get probabilities
        exp_sim = np.exp((similarities - similarities.max()) * 100.0)  # temperature scaling
        probs = exp_sim / exp_sim.sum()

        cat_prob = float(probs[0])  # probability of "a photo of a cat"

        results.append({
            "model": "CLIP-ViT-B/32",
            "image": img_name,
            "cat_confidence": cat_prob if cat_prob > 0.5 else None,
            "cat_score": float(similarities[0]),
            "nocat_score": float(similarities[1]),
            "cat_probability": cat_prob,
            "total_detections": 1 if cat_prob > 0.5 else 0,
            "inference_ms": elapsed_ms,
            "category": categorize_image(img_name),
        })

    return results


def run_dinov2_benchmark(images: list[tuple[str, Image.Image]],
                         train_probe: bool = False) -> list[dict]:
    """Run DINOv2-Small feature extraction. Optionally train a linear probe."""
    model_path = str(MODELS_DIR / "dinov2_vits14.onnx")
    if not os.path.exists(model_path):
        print("  DINOv2 model not found, skipping")
        return []

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]

    # Extract features for all images
    features = {}
    timings = {}
    for img_name, img in images:
        inp = preprocess_dino(img)

        # Build feed dict, handling optional 'masks' input
        feed = {"pixel_values": inp}
        if "masks" in input_names:
            feed["masks"] = np.array(False)

        # Warmup
        session.run(None, feed)

        # Timed run
        start = time.perf_counter()
        output = session.run(None, feed)
        elapsed_ms = (time.perf_counter() - start) * 1000

        features[img_name] = output[0][0]  # [384]
        timings[img_name] = elapsed_ms

    if not train_probe:
        # Without a probe, just report features and timing
        results = []
        for img_name, _ in images:
            results.append({
                "model": "DINOv2-S (features only)",
                "image": img_name,
                "cat_confidence": None,
                "total_detections": 0,
                "inference_ms": timings[img_name],
                "category": categorize_image(img_name),
            })
        return results

    # Train a simple linear probe using the labeled images
    print("  Training linear probe on labeled images...")
    X_train = []
    y_train = []
    for img_name in features:
        cat = categorize_image(img_name)
        if cat == "no_cat":
            X_train.append(features[img_name])
            y_train.append(0)
        else:
            X_train.append(features[img_name])
            y_train.append(1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Simple logistic regression (no sklearn dependency)
    # Use numpy-based gradient descent
    n_features = X_train.shape[1]
    weights = np.zeros(n_features, dtype=np.float32)
    bias = 0.0
    lr = 0.01
    n_epochs = 1000

    for _ in range(n_epochs):
        logits = X_train @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        grad_w = X_train.T @ (probs - y_train) / len(y_train)
        grad_b = (probs - y_train).mean()
        weights -= lr * grad_w
        bias -= lr * grad_b

    print(f"  Probe trained. Weights norm: {np.linalg.norm(weights):.3f}, bias: {bias:.3f}")

    # Evaluate on all images
    results = []
    for img_name, _ in images:
        feat = features[img_name]
        logit = float(feat @ weights + bias)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))

        results.append({
            "model": "DINOv2-S + probe",
            "image": img_name,
            "cat_confidence": float(prob) if prob > 0.5 else None,
            "cat_probability": float(prob),
            "total_detections": 1 if prob > 0.5 else 0,
            "inference_ms": timings[img_name],
            "category": categorize_image(img_name),
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_results(all_results: list[dict], overhead_only: bool = False):
    """Print formatted comparison table."""
    if not all_results:
        print("No results to display.")
        return

    # Get unique models and images
    models = list(dict.fromkeys(r["model"] for r in all_results))
    all_images = list(dict.fromkeys(r["image"] for r in all_results))

    if overhead_only:
        images = [img for img in all_images if categorize_image(img) != "stock"]
    else:
        images = all_images

    # Build lookup
    lookup = {}
    for r in all_results:
        lookup[(r["model"], r["image"])] = r

    # Print header
    model_cols = [m[:15] for m in models]
    header = f"{'Image':<40} {'Cat?':>4}"
    for mc in model_cols:
        header += f" {mc:>15}"
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")

    # Group by category
    categories = ["overhead_light", "overhead_tabby", "overhead_inside", "no_cat", "stock"]
    cat_labels = {
        "overhead_light": "--- OVERHEAD (LIGHT CAT) ---",
        "overhead_tabby": "--- OVERHEAD (TABBY) ---",
        "overhead_inside": "--- OVERHEAD (INSIDE LITTERBOT) ---",
        "no_cat": "--- NO CAT ---",
        "stock": "--- STOCK PHOTOS ---",
    }

    for cat in categories:
        cat_images = [img for img in images if categorize_image(img) == cat]
        if not cat_images:
            continue

        print(f"\n{cat_labels[cat]}")
        for img in sorted(cat_images):
            has_cat = "Y" if categorize_image(img) != "no_cat" else "N"
            row = f"{img:<40} {has_cat:>4}"
            for model in models:
                r = lookup.get((model, img))
                if r is None:
                    row += f" {'N/A':>15}"
                elif "cat_probability" in r:
                    # Classification model - show probability
                    prob = r["cat_probability"]
                    row += f" {prob * 100:>6.1f}%({r['inference_ms']:>4.0f}ms)"
                elif r["cat_confidence"] is not None:
                    row += f" {r['cat_confidence'] * 100:>6.1f}%({r['inference_ms']:>4.0f}ms)"
                else:
                    row += f" {'---':>6} ({r['inference_ms']:>4.0f}ms)"
            print(row)

    print(f"\n{'=' * len(header)}")

    # Summary per model per category
    print("\n\nSUMMARY: Detection rate by category (confidence > threshold)")
    print(f"{'Model':<25} {'Cat threshold':>14}", end="")
    for cat in categories:
        if cat == "stock" and overhead_only:
            continue
        print(f" {cat:>20}", end="")
    print(f" {'Avg ms':>8}")
    print("-" * 130)

    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        is_classifier = any("cat_probability" in r for r in model_results)

        row = f"{model:<25} {'prob > 0.5' if is_classifier else f'conf > {CONF_THRESHOLD}':>14}"
        for cat in categories:
            if cat == "stock" and overhead_only:
                continue
            cat_results = [r for r in model_results if r["category"] == cat and r["image"] in images]
            if not cat_results:
                row += f" {'N/A':>20}"
                continue

            if cat == "no_cat":
                # For no_cat images, we want 0 false positives
                if is_classifier:
                    fps = sum(1 for r in cat_results if r.get("cat_probability", 0) > 0.5)
                else:
                    fps = sum(1 for r in cat_results if r["cat_confidence"] is not None)
                row += f" {fps}/{len(cat_results)} FP".rjust(20)
            else:
                # For cat images, we want high detection rate
                if is_classifier:
                    detected = sum(1 for r in cat_results if r.get("cat_probability", 0) > 0.5)
                else:
                    detected = sum(1 for r in cat_results if r["cat_confidence"] is not None)
                rate = detected / len(cat_results) * 100
                row += f" {detected}/{len(cat_results)} ({rate:.0f}%)".rjust(20)

        avg_ms = np.mean([r["inference_ms"] for r in model_results if r["image"] in images])
        row += f" {avg_ms:>7.1f}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark models on overhead cat images")
    parser.add_argument("--overhead-only", action="store_true",
                        help="Only show overhead camera images (not stock photos)")
    parser.add_argument("--train-dinov2-probe", action="store_true",
                        help="Train a linear probe on DINOv2 features")
    args = parser.parse_args()

    # Load all test images
    print("Loading test images...")
    images = []
    for f in sorted(TEST_DIR.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            img = Image.open(f).convert("RGB")
            images.append((f.name, img))
    print(f"  Loaded {len(images)} images")

    all_results = []

    # YOLO models
    yolo_models = [
        ("YOLO11n", "models/yolo11n.onnx", "yolo11"),
        ("YOLO11s", "models/yolo11s.onnx", "yolo11"),
        ("YOLO11m", "models/yolo11m.onnx", "yolo11"),
        ("YOLOv8s", "models/yolov8s.onnx", "yolo11"),  # Same postprocess as YOLO11
        ("YOLOv8m", "models/yolov8m.onnx", "yolo11"),
        ("YOLOX-S", "models/yolox_s.onnx", "yolox"),
    ]

    for model_name, model_path, model_type in yolo_models:
        if not os.path.exists(model_path):
            print(f"  Skipping {model_name}: {model_path} not found")
            continue
        print(f"Running {model_name}...")
        results = run_yolo_benchmark(model_name, model_path, model_type, images)
        all_results.extend(results)

    # CLIP
    print("Running CLIP ViT-B/32...")
    all_results.extend(run_clip_benchmark(images))

    # DINOv2
    print("Running DINOv2-Small...")
    all_results.extend(run_dinov2_benchmark(images, train_probe=args.train_dinov2_probe))

    # Print results
    print_results(all_results, overhead_only=args.overhead_only)

    # Save raw results as JSON
    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {output_path}")


if __name__ == "__main__":
    main()
