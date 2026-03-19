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

![CLIP Detection Pipeline](/img/clip-pipeline.png)

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

### 4. Softmax with Temperature

The raw similarity scores are converted to probabilities using softmax with a high temperature (τ = 100):

```
cat_prob = exp((cat_sim - max_sim) × τ) / Σ exp((sim_i - max_sim) × τ)
```

The temperature of 100 (matching CLIP's learned `logit_scale`) amplifies small differences in cosine similarity into confident probability distributions. The `max_sim` subtraction is for numerical stability (prevents overflow in `exp()`).

**How temperature affects confidence:**

| Scenario | cat_sim | room_sim | Δ | τ=1 cat_prob | τ=100 cat_prob |
|---|---|---|---|---|---|
| Clear cat | 0.247 | 0.209 | 0.038 | 51.0% | 96.8% |
| Dirty litter box | 0.212 | 0.207 | 0.005 | 50.1% | 55.7% |
| Empty room | 0.213 | 0.234 | -0.021 | 49.5% | 9.7% |

At τ=1, all predictions are near 50% — useless. At τ=100, clear cases become very confident (96.8%), but borderline cases like the dirty litter box also get pushed just over the threshold (55.7%).

### 5. Thresholding

If `cat_prob >= confidence_threshold` (default 0.5), a `Detection` is returned with:
- `class_id`: 15 (COCO cat class, for compatibility with the `CatDetector` trait)
- `confidence`: the cat probability
- `bbox`: full frame (CLIP classifies the whole image, not a region)

If below threshold, an empty detection list is returned.

## Benchmark Results

*All values computed on 2026-03-18 using the current 3-class text embeddings and CLIP ViT-B/32.*

### Per-Image Classification

| Image | cat_sim | room_sim | person_sim | cat_prob | Result | Correct? |
|---|---|---|---|---|---|---|
| cat1.jpg (stock, side-angle) | 0.2997 | 0.1637 | 0.2521 | 99.2% | CAT | ✅ |
| cat_overhead_litterbot1.jpg | 0.2463 | 0.1845 | 0.2002 | 98.8% | CAT | ✅ |
| cat_overhead_tabby1.jpg | 0.2471 | 0.2089 | 0.2016 | 96.8% | CAT | ✅ |
| no_cat_overhead.jpg | 0.2125 | 0.2343 | 0.2026 | 9.7% | — | ✅ |
| person_overhead_1.jpg | 0.1965 | 0.1985 | 0.2280 | 3.9% | — | ✅ |
| **litter_box_dirty_overhead_1.jpg** | **0.2123** | **0.2070** | **0.1964** | **55.7%** | **CAT** | ❌ FP |
| **litter_robot_moving_overhead_1.jpg** | **0.2350** | **0.2145** | **0.1953** | **87.2%** | **CAT** | ❌ FP |
| **person_overhead_catbox_1.jpg** | **0.2216** | **0.1777** | **0.2100** | **75.3%** | **CAT** | ❌ FP |
| **litter_box_and_robot_overhead_1.jpg** | **0.2131** | **0.2103** | **0.2025** | **47.6%** | **—** | ✅ |

**Summary**: 6/9 correct, 3 false positives. The false positives occur because there is no text embedding for "litter box" — scenes containing litter boxes are slightly more similar to "a photo of a cat" than to "a photo of an empty room", and the high temperature amplifies this tiny difference into a confident (wrong) prediction. (Note: this image was previously mislabeled as "cat_with_litter_box" but contains no cat — the 47.6% is actually below threshold, so it's correctly rejected.)

### Known Limitations

1. **No litter box class**: The 3-class system has no representation for litter boxes, litter robots, or other objects common in the deployment environment. Scenes with these objects default to whichever existing class is closest.

2. **Temperature sensitivity**: With τ=100, a cosine similarity difference as small as 0.005 (dirty litter box) becomes a 55.7% confident prediction — just enough to cross the threshold.

3. **Overhead person from catbox**: The person class was trained with the text "a photo of a person" which may not match the overhead perspective well. The catbox person image scores higher on cat (0.222) than person (0.210).

4. **Cat with litter box**: When a cat is present alongside a litter box, the embeddings are nearly equal across all classes (0.213/0.210/0.203), resulting in a 47.6% cat probability — just below the 0.5 threshold.

These limitations motivate the [few-shot prototype approach](./few-shot-detection.md) which uses representative images from the actual deployment environment instead of generic text descriptions.

## Text Embeddings

The text side of CLIP is pre-computed offline, not run at inference time. This is what makes it efficient — only the image encoder runs per frame.

### Three-Class System

| Index | Prompt | Role |
|---|---|---|
| 0 | "a photo of a cat" | Positive class |
| 1 | "a photo of an empty room" | Negative — baseline for no activity |
| 2 | "a photo of a person" | Negative — prevents people triggering cat detection |

The person class was added after deployment showed that people walking by the camera could occasionally produce elevated cat similarity scores. With the 3-class system, people are correctly classified with 73-95% person probability (though overhead angles from catbox can still cause misclassification — see benchmarks above).

### Text Embedding Similarity Matrix

The three text embeddings are fairly similar to each other (all above 0.81), which means they occupy a relatively small region of the 512-dimensional embedding space:

| | cat | room | person |
|---|---|---|---|
| **cat** | 1.0000 | 0.8149 | 0.8947 |
| **room** | 0.8149 | 1.0000 | 0.8368 |
| **person** | 0.8947 | 0.8368 | 1.0000 |

**L2 distances** (lower = more similar):

| | cat | room | person |
|---|---|---|---|
| **cat** | 0 | 0.6085 | 0.4588 |
| **room** | 0.6085 | 0 | 0.5713 |
| **person** | 0.4588 | 0.5713 | 0 |

Note that "cat" and "person" are the closest pair (L2=0.4588), while "cat" and "room" are the most distant (L2=0.6085). This makes intuitive sense — cats and people are both living beings, while an empty room is semantically different from both.

### Binary Format

Text embeddings are stored in `models/clip_text_embeddings.bin` (6 KB, tracked in git):

```
[count: u32 LE]     # number of embeddings (3)
[dim: u32 LE]       # embedding dimension (512)
[embed_0: f32 × dim]  # "a photo of a cat" (positive, always first)
[embed_1: f32 × dim]  # "a photo of an empty room" (negative)
[embed_2: f32 × dim]  # "a photo of a person" (negative)
```

<details>
<summary>Raw text embedding vectors (512 dimensions each)</summary>

**"a photo of a cat"** (index 0):
```
 0.014839  0.006996 -0.023371 -0.021110 -0.018464  0.019585 -0.030305 -0.074684
-0.017338  0.018549 -0.007272 -0.057772  0.018357 -0.009701  0.013113  0.013050
 0.023123 -0.017145 -0.008487  0.027411  0.060565  0.015679  0.043362 -0.002796
-0.023998  0.018364  0.023734  0.061723 -0.022451 -0.022118  0.021570  0.018090
 0.014357  0.046030 -0.020915 -0.017416  0.024183 -0.002612  0.022596  0.022664
-0.009123 -0.011028  0.026837  0.011385  0.015528  0.008279 -0.024035  0.003048
... (512 values total)
```

**"a photo of an empty room"** (index 1):
```
-0.035613  0.026705 -0.008820  0.009752 -0.003528 -0.006282  0.016475 -0.081894
-0.023054 -0.003041 -0.007691 -0.051163  0.005598 -0.002510  0.028033 -0.041102
-0.038936  0.006916  0.026500  0.048828  0.043612  0.005480  0.034588 -0.032941
-0.016236  0.028811  0.025801  0.007362 -0.007014 -0.046107 -0.001501  0.015215
-0.014104  0.028420 -0.071122 -0.028660  0.019562 -0.010183 -0.021407 -0.014782
-0.001861 -0.033285  0.027487 -0.007759  0.054028  0.018422 -0.012138  0.005854
... (512 values total)
```

**"a photo of a person"** (index 2):
```
-0.000751  0.024512 -0.006577  0.012351  0.006939 -0.024575 -0.011932 -0.078067
-0.005109  0.016693 -0.021322 -0.034124  0.019928 -0.001946  0.014360 -0.009200
 0.055922  0.024275 -0.012183 -0.003026  0.082108  0.028157  0.017819 -0.032328
-0.021158  0.000695  0.006157  0.053323 -0.028469 -0.001919  0.022747  0.009145
-0.023504  0.004503 -0.006197 -0.041314  0.007477 -0.017493 -0.004094  0.016229
-0.025926 -0.007087  0.007618  0.041891  0.019979  0.033560 -0.024059  0.022099
... (512 values total)
```

</details>

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

## Reproducing These Results

The benchmark values on this page were computed with:

```bash
pixi run python scripts/compute_doc_values.py
```

This script runs the CLIP image encoder on representative test images and prints cosine similarities, softmax probabilities, and raw embedding vectors. Output is saved to `scripts/doc_values_output.txt`.
