---
sidebar_position: 4
title: How CLIP Detection Works
---

# How CLIP Zero-Shot Detection Works

cat-detector uses OpenAI's CLIP ViT-B/32 model for zero-shot cat classification. Unlike traditional object detectors (YOLO, SSD) that are trained on labeled bounding boxes, CLIP classifies entire frames by comparing image content against text descriptions. This document explains the full pipeline.

## Why CLIP Instead of YOLO?

Traditional object detectors like YOLO are trained on COCO dataset images — mostly side-angle photos. The cat-detector uses an overhead USB webcam pointed at a litter robot. YOLO models consistently fail at this viewpoint (best result: 8/19 overhead detections). CLIP's zero-shot approach generalizes to any camera angle because it matches images against natural language descriptions rather than learned bounding box patterns.

See the [Model Evaluation](./model-evaluation.md) for detailed benchmark comparisons.

## Pipeline Overview

```
                    ┌──────────────────────┐
  Camera frame      │  1. Preprocess       │
  (640x480 RGB) ──> │  Resize + crop +     │
                    │  normalize to        │
                    │  224x224 NCHW tensor  │
                    └─────────┬────────────┘
                              │
                              v
                    ┌──────────────────────┐
                    │  2. CLIP Image        │
                    │  Encoder (ONNX)       │     Pre-computed at build time:
                    │  ViT-B/32, 88M params │     ┌──────────────────────────┐
                    └─────────┬────────────┘     │  Text embeddings (6KB)   │
                              │                   │  "a photo of a cat"      │
                    512-dim image embedding        │  "a photo of an empty    │
                              │                   │   room"                  │
                              v                   │  "a photo of a person"   │
                    ┌──────────────────────┐     └────────────┬─────────────┘
                    │  3. Cosine Similarity │                  │
                    │  image vs each text   │ <───────────────┘
                    │  embedding            │   3 x 512-dim text embeddings
                    └─────────┬────────────┘
                              │
                       3 similarity scores
                              │
                              v
                    ┌──────────────────────┐
                    │  4. Softmax (τ=100)   │
                    │  Convert similarities │
                    │  to probabilities     │
                    └─────────┬────────────┘
                              │
                       cat probability
                              │
                              v
                    ┌──────────────────────┐
                    │  5. Threshold (0.5)   │
                    │  cat_prob >= 0.5?     │──> Detection or no detection
                    └──────────────────────┘
```

## Step by Step

### 1. Image Preprocessing

CLIP expects a specific input format matching OpenAI's original training pipeline:

1. **Resize** the shortest side to 224 pixels (maintaining aspect ratio) using Catmull-Rom interpolation
2. **Center crop** to 224x224
3. **Convert** to float32, scale to [0, 1]
4. **Normalize** with CLIP's channel means and standard deviations:
   - Mean: `[0.48145466, 0.4578275, 0.40821073]`
   - Std: `[0.26862954, 0.26130258, 0.27577711]`
5. **Reshape** to NCHW format: `[1, 3, 224, 224]`

This is implemented in `ClipDetector::preprocess()` in `src/detector.rs`.

### 2. Image Encoding

The preprocessed tensor is fed through the CLIP ViT-B/32 image encoder, a Vision Transformer that splits the image into 32x32 pixel patches and processes them through transformer layers.

- **Model**: `clip_vitb32_image.onnx` (352 MB), exported from HuggingFace `openai/clip-vit-base-patch32`
- **Runtime**: ONNX Runtime loaded dynamically via `ORT_DYLIB_PATH`
- **Output**: 512-dimensional embedding vector
- **Inference**: ~21ms on i3-6100T CPU (4 intra-op threads)

The ONNX model includes an L2 normalization layer, so the output is always unit-norm. The Rust code also applies L2 normalization defensively.

### 3. Cosine Similarity

The image embedding is compared against each pre-computed text embedding using cosine similarity (dot product of L2-normalized vectors):

```
similarity = Σ(image_i × text_i)
```

This produces one similarity score per class:
- `cat_sim` — similarity to "a photo of a cat"
- `room_sim` — similarity to "a photo of an empty room"
- `person_sim` — similarity to "a photo of a person"

Typical values:
- Cat present: `cat_sim ≈ 0.28`, `room_sim ≈ 0.20`, `person_sim ≈ 0.18`
- Empty room: `cat_sim ≈ 0.19`, `room_sim ≈ 0.27`, `person_sim ≈ 0.20`
- Person present: `cat_sim ≈ 0.19`, `room_sim ≈ 0.20`, `person_sim ≈ 0.26`

The differences are small in absolute terms, but the softmax step amplifies them.

### 4. Softmax with Temperature

The raw similarity scores are converted to probabilities using softmax with a high temperature (τ = 100):

```
cat_prob = exp((cat_sim - max_sim) × 100) / Σ exp((sim_i - max_sim) × 100)
```

The temperature of 100 (matching CLIP's learned `logit_scale`) amplifies small differences in cosine similarity into confident probability distributions. A 0.08 difference in similarity becomes a 97%+ probability.

The `max_sim` subtraction is for numerical stability (prevents overflow in `exp()`).

### 5. Thresholding

If `cat_prob >= confidence_threshold` (default 0.5), a `Detection` is returned with:
- `class_id`: 15 (COCO cat class, for compatibility with the `CatDetector` trait)
- `confidence`: the cat probability
- `bbox`: full frame (CLIP classifies the whole image, not a region)

If below threshold, an empty detection list is returned.

## Text Embeddings

The text side of CLIP is pre-computed offline, not run at inference time. This is what makes it efficient — only the image encoder runs per frame.

### Three-Class System

| Index | Prompt | Role |
|---|---|---|
| 0 | "a photo of a cat" | Positive class |
| 1 | "a photo of an empty room" | Negative — baseline for no activity |
| 2 | "a photo of a person" | Negative — prevents people triggering cat detection |

The person class was added after deployment showed that people walking by the camera could occasionally produce elevated cat similarity scores. With the 3-class system, people are correctly classified with 73-95% person probability.

### Binary Format

Text embeddings are stored in `models/clip_text_embeddings.bin` (6 KB, tracked in git):

```
[count: u32 LE]     # number of embeddings (3)
[dim: u32 LE]       # embedding dimension (512)
[embed_0: f32 × dim]  # "a photo of a cat" (positive, always first)
[embed_1: f32 × dim]  # "a photo of an empty room" (negative)
[embed_2: f32 × dim]  # "a photo of a person" (negative)
```

### Regenerating Embeddings

To change the classification classes (add new prompts, adjust wording):

```bash
# Edit PROMPTS list in scripts/generate_clip_embeddings.py
# First prompt is always the positive class
pixi run -e clip generate-embeddings
```

This uses the original OpenAI CLIP Python package (via pixi's `clip` environment) to encode text prompts through the CLIP text encoder. The text encoder produces embeddings in the same 512-dimensional space as the image encoder — this shared space is what makes zero-shot classification work.

## How It Differs from Object Detection

| | CLIP (cat-detector) | YOLO/SSD |
|---|---|---|
| **Task** | Whole-frame classification | Object detection with bounding boxes |
| **Training** | Zero-shot (no task-specific training) | Supervised on labeled bounding boxes |
| **Output** | Probability per class | Bounding boxes with class + confidence |
| **Viewpoint** | Generalizes to any angle | Biased toward training data angles |
| **Classes** | Any text description | Fixed set from training |
| **Localization** | None (full frame) | Per-object bounding boxes |
| **Adding classes** | Change text prompts, regenerate embeddings | Retrain the model |

The key tradeoff: CLIP can't tell you *where* the cat is in the frame, only that a cat is present. For cat-detector's use case (entrance/exit monitoring), this is sufficient.
