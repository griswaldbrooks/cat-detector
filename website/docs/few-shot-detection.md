---
sidebar_position: 5
title: Few-Shot Prototype Detection
---

# Few-Shot Prototype Detection

:::info Not Yet Deployed
The embedding pipeline and benchmarks are complete. The [zero-shot text approach](./clip-detection.md) is still the deployed pipeline — switching to image embeddings requires only a config change pointing to `clip_image_embeddings.bin`.
:::

## Motivation

The current zero-shot detector compares camera frames against three **text** embeddings: "a photo of a cat", "a photo of an empty room", and "a photo of a person". This works well for clear cases but fails on objects common in the catbox environment:

| Image | Cat Probability | Result | Problem |
|---|---|---|---|
| Dirty litter box (no cat) | 55.7% | **False positive** | Litter box texture resembles "cat" more than "room" |
| Litter robot moving (no cat) | 87.2% | **False positive** | Moving robot parts trigger high cat similarity |
| Person from overhead | 75.3% | **False positive** | Overhead person silhouette matches "cat" > "person" |
| Cat next to litter box | 47.6% | **False negative** | Litter box pulls similarity toward "room" |

The root cause is that text embeddings like "a photo of a cat" are too generic — they encode a typical internet photo of a cat, not the specific overhead view from our camera. With the high softmax temperature (τ=100) amplifying small cosine similarity differences, even a 0.005 delta becomes a confident classification.

## How Prototype Networks Work

[Prototype networks](https://arxiv.org/abs/1703.05175) (Snell et al. 2017) represent each class by the **mean embedding** of a small set of representative examples (the "support set"). Classification assigns a query to the class whose prototype is nearest in embedding space.

This maps naturally onto CLIP: instead of encoding class descriptions as text, we encode representative **images** through the same CLIP image encoder and average the resulting embeddings. The averaged embedding becomes a prototype that captures the visual characteristics of that class from the deployment camera's perspective.

Why this works with CLIP:

1. **CLIP's embedding space is shared** — image and text embeddings occupy the same 512-dimensional space, so image prototypes are drop-in replacements for text embeddings
2. **Averaging preserves class signal** — Snell et al. showed that class means in a learned embedding space are effective nearest-centroid classifiers
3. **Environment-specific** — prototypes from overhead catbox photos capture the exact camera angle, lighting, and scene context that text descriptions miss

## Pipeline Overview

The runtime pipeline is identical to zero-shot — only the embeddings source changes:

![Few-Shot Pipeline](/img/fewshot-pipeline.png)

The left side (steps 1-5) is unchanged from the [zero-shot pipeline](./clip-detection.md). The right side shows the offline prototype generation process that replaces text embedding computation.

## What Changes vs Zero-Shot

| Aspect | Zero-Shot (current) | Few-Shot (proposed) |
|---|---|---|
| **Embedding source** | Text encoder + prompts | Image encoder + reference photos |
| **Classes** | 3 (cat, room, person) | 4 (cat, room, person, litter box) |
| **Binary format** | `[count:u32][dim:u32][f32...]` | Same — identical format |
| **File** | `clip_text_embeddings.bin` (6KB) | `clip_image_embeddings.bin` (~8KB) |
| **Rust code changes** | — | Config: point to new .bin file |
| **Preprocessing** | CLIP resize/crop/normalize | Same |
| **Similarity** | Cosine dot product | Same |
| **Softmax** | τ=100 | Same (may tune) |
| **Threshold** | 0.5 | Same (may tune) |

The key insight is that the Rust detection code is **embedding-agnostic** — it loads a binary file of `count × dim` vectors and computes cosine similarity against each. Swapping text embeddings for image prototypes requires zero Rust code changes, only a different `.bin` file.

## Proposed 4-Class System

| Index | Class | Role | Source Images |
|---|---|---|---|
| 0 | **Cat overhead** | Positive (detection target) | Overhead photos of both cats at catbox |
| 1 | **Empty room overhead** | Negative | Empty catbox with various lighting |
| 2 | **Person overhead** | Negative | Overhead person images at catbox |
| 3 | **Litter box / robot** | Negative | Dirty litter box, moving robot, no cat |

Adding the litter box class is the critical fix — it gives the softmax denominator an explicit "litter box" option, preventing the model from being forced to choose between "cat" and "room" when it sees litter box textures.

## Prototype Computation

For each class with reference images:

1. **Encode** each image through CLIP's image encoder
2. **L2-normalize** each embedding to unit length
3. **Average** the normalized embeddings for the class
4. **Re-normalize** the averaged vector back to unit length

In pseudocode:

```
for each class c:
    embeddings = [CLIP_encode(img) for img in reference_images[c]]
    embeddings = [e / norm(e) for e in embeddings]       # L2 normalize
    prototype = mean(embeddings)                          # average
    prototype = prototype / norm(prototype)               # re-normalize
```

Re-normalization after averaging is important because the mean of unit vectors is not itself a unit vector. Without it, prototype norms would vary by class (more diverse classes produce a shorter mean vector), biasing cosine similarities.

## Image Curation Guidelines

### Selection criteria

- **Overhead perspective**: All images should be from the catbox camera's viewpoint (or similar overhead angle)
- **Variety within class**: Different lighting conditions, times of day, cat positions
- **Clean class boundaries**: No ambiguous images (e.g., a cat partially visible should go in "cat", not "litter box")
- **Representative of deployment**: Use actual catbox captures where possible

### Recommended counts

- **Cat overhead**: 10-15 images (both cats, various positions — walking, sitting, at litter robot)
- **Empty room**: 5-10 images (different lighting, with/without litter robot visible)
- **Person overhead**: 5 images (the existing `person_images/` set covers this)
- **Litter box / robot**: 5-10 images (dirty litter box, robot mid-cycle, robot with drawer open)

### Source directories

Reference images are curated from:
- `test_images/cat_overhead_*.jpg` — overhead cat photos
- `test_images/catbox_captures/` — production captures from catbox
- `test_images/no_cat_overhead.jpg` — empty room
- `test_images/person_images/` — overhead person photos
- `test_images/litter_*.jpg` — litter box edge cases

Selected images are copied into `prototype_images/<class>/` for reproducible embedding generation.

## Benchmark Results

| Image | Zero-Shot P(cat) | Few-Shot P(cat) | Status |
|---|---|---|---|
| Cat overhead (tabby) | 97.0% | **99.96%** | Maintained |
| Cat overhead (at litter robot) | 99.0% | **99.96%** | Maintained |
| Stock cat photo | 99.0% | **90.1%** | Maintained |
| Empty room | 9.7% | **0.02%** | Improved |
| Person overhead (test set) | 3.9% | **0.40%** | Maintained |
| **Dirty litter box** | **55.7%** (FP) | **0.00%** | **Fixed** |
| **Litter robot moving** | **87.2%** (FP) | **0.20%** | **Fixed** |
| **Overhead person at catbox** | **75.3%** (FP) | **0.40%** | **Fixed** |
| **Cat with litter box** | **47.6%** (FN) | **0.00%** (FN) | Remains — see below |

:::note Remaining false negative
The "cat with litter box" image remains a false negative because the litter box dominates the visual features (litter similarity 0.972 vs cat 0.842). With only 2 litter box prototype images, the litter class is very narrowly tuned. This can likely be improved by curating more diverse litter box prototype images or tuning the softmax temperature.
:::

## Embedding Generation

The prototype embeddings are generated by a Python script using the same ONNX model as the Rust runtime:

```bash
pixi run generate-image-embeddings \
  --classes cat_overhead:prototype_images/cat_overhead \
            empty_room:prototype_images/empty_room_overhead \
            person_overhead:prototype_images/person_overhead \
            litter_box:prototype_images/litter_box_overhead \
  --output models/clip_image_embeddings.bin
```

The script prints verification info: per-prototype norms (should all be 1.0) and pairwise cosine similarities (should show clear class separation).

## Future Direction: Binary Threshold

The multi-class softmax approach requires enumerating all negative classes — anything not explicitly listed can cause false positives (the original litter box problem). An alternative is **binary thresholding** on cosine similarity to the cat prototype alone:

- Compute `cos(query, cat_prototype)` — if above threshold, it's a cat
- No need to enumerate what else it might be
- Simpler, no "forgot a class" failure mode

With text embeddings this doesn't work well because absolute cosine similarities are poorly separated ("a photo of a cat" scores ~0.25 for everything). But image prototypes from the actual deployment camera produce much more discriminative similarities — a real overhead cat scores very differently from an empty room.

The current multi-class system works well with sufficient prototype images per class, but binary thresholding is worth exploring as a simpler and more robust long-term approach.

## References

1. Snell, J., Swersky, K., & Zemel, R. (2017). [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175). *NeurIPS 2017*.
2. Radford, A., et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). *ICML 2021*.
