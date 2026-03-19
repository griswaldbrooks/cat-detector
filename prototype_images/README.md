# Prototype Images

Reference images for few-shot prototype embedding generation. Each subdirectory
contains curated images for one class.

## Directory Structure

```
prototype_images/
  cat_overhead/          # Positive class (index 0): overhead photos of cats at catbox
  empty_room_overhead/   # Negative: empty catbox area, various lighting
  person_overhead/       # Negative: overhead person images
  litter_box_overhead/   # Negative: dirty litter box, litter robot mid-cycle (no cat)
```

## Selection Criteria

- **Overhead perspective**: All images should be from the catbox camera's viewpoint
- **Variety within class**: Different lighting, times of day, positions
- **Clean class boundaries**: No ambiguous images across classes
- **Representative of deployment**: Use actual catbox captures where possible

## Recommended Counts

- **Cat overhead**: 10-15 images (both cats, various positions)
- **Empty room**: 5-10 images (different lighting conditions)
- **Person overhead**: 5 images
- **Litter box / robot**: 5-10 images (dirty box, robot cycling, no cat visible)

## Generating Embeddings

```bash
pixi run generate-image-embeddings
```

## Image Sources

Prototype images are pulled directly from catbox production captures. They must
**never** overlap with test images — run `pixi run check-leakage` to verify.

Sources:
- `catbox` SSH captures (cat entry/sample/exit frames from production sessions)
- Person and litter box images captured specifically for prototype training
