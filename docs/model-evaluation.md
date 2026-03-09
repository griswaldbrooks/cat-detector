# Model Evaluation Results

Historical benchmark comparing 8 models for overhead cat detection on a Dell Optiplex 3040M (i3-6100T, no GPU). Tests run in March 2025.

## Why CLIP Won

The cat detector uses an overhead USB webcam pointed at a litter robot. YOLO models are trained on side-angle photos from COCO — they struggle with top-down views of cats. CLIP's zero-shot classification ("a photo of a cat" vs "a photo of an empty room" vs "a photo of a person") generalizes to any viewpoint.

## Summary

| Model | Overhead Light (19) | Overhead Tabby (26) | Inside Litter (3) | Stock Side-Angle (15) | No Cat (5) | Avg Inference |
|---|---|---|---|---|---|---|
| **CLIP ViT-B/32** | **19/19 (100%)** | **26/26 (100%)** | **3/3 (97%)** | 14/15 (92%) | **0/5 (0 FP)** | **21ms** |
| DINOv2-S + probe | 19/19 (100%) | 26/26 (100%) | 3/3 (99%) | **15/15 (100%)** | 0/5 (0 FP) | 24ms |
| YOLO11n | 7/19 (23%) | 5/26 (38%) | 0/3 (0%) | 10/15 (69%) | 1/5 (1 FP) | 17ms |
| YOLO11s | 0/19 (0%) | 4/26 (33%) | 0/3 (0%) | 10/15 (66%) | 0/5 (0 FP) | 36ms |
| YOLO11m | 8/19 (23%) | 5/26 (52%) | 0/3 (0%) | 11/15 (67%) | 0/5 (0 FP) | 95ms |
| YOLOv8s | 1/19 (31%) | 8/26 (44%) | 0/3 (0%) | 8/15 (74%) | 0/5 (0 FP) | 44ms |
| YOLOv8m | 6/19 (45%) | 4/26 (64%) | 0/3 (0%) | 9/15 (81%) | 1/5 (1 FP) | 120ms |
| YOLOX-S | 3/19 (35%) | 8/26 (27%) | 0/3 (0%) | 11/15 (61%) | 0/5 (0 FP) | 48ms |

Confidence shown is average across detected images only. "Overhead Light" is the light-colored Ragdoll/Birman cat, "Overhead Tabby" is the tabby cat — both from the overhead webcam. "Stock" images are standard side-angle cat photos.

## Key Findings

- **YOLO models fail at overhead detection.** Best YOLO result was 8/19 on overhead light cat (YOLO11m), vs CLIP's 19/19. No YOLO model detected cats inside the litter robot.
- **CLIP and DINOv2 both excel.** DINOv2-S with a linear probe scored slightly higher on stock images but required training a probe layer. CLIP works zero-shot with just text prompts.
- **CLIP chosen for simplicity.** No training needed, ~21ms inference, 100% overhead detection, zero false positives on empty rooms. Text embeddings can be updated by changing prompts.
- **YOLO11n had false positives.** Detected a "cat" in an empty room image (19% confidence) — the only fast model with false positives.
- **Bigger YOLO models don't help overhead.** YOLO11m (95ms) barely outperformed YOLO11n (17ms) on overhead images while being 5.5x slower.

## Test Image Categories

| Category | Count | Description |
|---|---|---|
| overhead_light | 19 | Light-colored cat from overhead webcam (walking, sitting, at scratcher, at litter robot) |
| overhead_tabby | 26 | Tabby cat from overhead webcam (eating, entering litter robot, at scratcher, etc.) |
| overhead_inside | 3 | Cat inside the litter robot (partially visible from above) |
| stock | 15 | Standard side-angle cat photos (various poses, backgrounds) |
| no_cat | 5 | Empty rooms, overhead views without cats |

## Models Tested

| Model | Type | Input Size | Parameters | Source |
|---|---|---|---|---|
| CLIP ViT-B/32 | Zero-shot classification | 224x224 | 88M (image encoder) | OpenAI via HuggingFace |
| DINOv2-S + probe | Feature extractor + linear probe | 224x224 | 22M | Meta |
| YOLO11n | Object detection | 640x640 | 2.6M | Ultralytics |
| YOLO11s | Object detection | 640x640 | 9.4M | Ultralytics |
| YOLO11m | Object detection | 640x640 | 20.1M | Ultralytics |
| YOLOv8s | Object detection | 640x640 | 11.2M | Ultralytics |
| YOLOv8m | Object detection | 640x640 | 25.9M | Ultralytics |
| YOLOX-S | Object detection | 640x640 | 9.0M | Megvii |

## Decision

CLIP ViT-B/32 was selected as the production model. It provides perfect overhead detection accuracy with zero false positives, fast inference (~21ms on CPU), and requires no training — only text prompts describing the classes.

The raw benchmark data is preserved in `benchmark_results.json` (untracked, 544 entries across 68 test images).
