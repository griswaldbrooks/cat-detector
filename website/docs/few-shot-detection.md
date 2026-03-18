---
sidebar_position: 5
title: Few-Shot Prototype Detection
---

# Few-Shot Prototype Detection

:::caution Work in Progress
This feature is under active development. See the [current zero-shot approach](./clip-detection.md) for the deployed detection pipeline.
:::

## Motivation

The current zero-shot text embedding approach has known false positive issues with objects common in the deployment environment (litter boxes, litter robots) and false negatives when cats appear alongside these objects. See [Known Limitations](./clip-detection.md#known-limitations) for details.

## Approach

Instead of comparing frames against **text** embeddings ("a photo of a cat"), few-shot prototype detection compares frames against **image** embeddings computed from representative photos of the actual deployment environment:

- **Cat prototypes**: Averaged CLIP image embeddings from overhead photos of the cats at the catbox
- **Negative prototypes**: Averaged embeddings from empty room shots, litter box scenes, and person images

This approach leverages [prototype networks](https://arxiv.org/abs/1703.05175) — each class is represented by the mean embedding of its support examples, and classification uses nearest-centroid in CLIP's embedding space.

## Expected Benefits

1. **Environment-specific**: Prototypes capture the actual camera angle, lighting, and objects
2. **Litter box discrimination**: Explicit negative examples of litter boxes prevent false positives
3. **Better overhead person handling**: Person prototypes from the overhead perspective
4. **Easy to update**: Adding new reference images and regenerating embeddings is straightforward

## Status

Implementation is tracked across several beads. Check `bd ready` for current progress.
