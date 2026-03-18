#!/usr/bin/env python3
"""Generate few-shot image prototype embeddings for CLIP detection.

Computes per-class prototype embeddings by averaging L2-normalized CLIP image
embeddings from reference photos. Output format is identical to text embeddings
(count, dim, f32 data), so the Rust code needs zero changes.

Usage:
    pixi run generate-image-embeddings

    # Or manually:
    python scripts/generate_image_embeddings.py \
        --classes cat_overhead:prototype_images/cat_overhead \
                  empty_room:prototype_images/empty_room_overhead \
                  person_overhead:prototype_images/person_overhead \
                  litter_box:prototype_images/litter_box_overhead \
        --output models/clip_image_embeddings.bin
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from clip_utils import CLIP_MODEL, MODELS_DIR, encode_image, save_embeddings


def compute_prototype(session, image_dir):
    """Compute prototype embedding for a class by averaging normalized image embeddings.

    Returns (prototype, count) where prototype is L2-normalized.
    """
    image_paths = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    embeddings = []
    for img_path in image_paths:
        emb = encode_image(session, img_path)
        embeddings.append(emb)

    # Average and re-normalize
    mean_emb = np.mean(embeddings, axis=0)
    prototype = mean_emb / np.linalg.norm(mean_emb)

    return prototype, len(embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Generate few-shot image prototype embeddings"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        metavar="NAME:DIR",
        help="Class definitions as name:directory pairs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "clip_image_embeddings.bin",
        help="Output binary file path",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=CLIP_MODEL,
        help="CLIP ONNX model path",
    )
    args = parser.parse_args()

    # Parse class definitions
    classes = []
    for cls_def in args.classes:
        if ":" not in cls_def:
            print(f"Error: class definition must be NAME:DIR, got '{cls_def}'", file=sys.stderr)
            sys.exit(1)
        name, dir_path = cls_def.split(":", 1)
        dir_path = Path(dir_path)
        if not dir_path.exists():
            print(f"Error: directory not found: {dir_path}", file=sys.stderr)
            sys.exit(1)
        classes.append((name, dir_path))

    print(f"Loading CLIP model: {args.model}")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    prototypes = []
    print(f"\nComputing prototypes for {len(classes)} classes:")
    for name, dir_path in classes:
        proto, count = compute_prototype(session, dir_path)
        prototypes.append(proto)
        norm = np.linalg.norm(proto)
        print(f"  [{len(prototypes)-1}] {name}: {count} images, norm={norm:.6f}")

    prototypes = np.array(prototypes, dtype=np.float32)

    # Save
    save_embeddings(prototypes, args.output)
    print(f"\nSaved {len(prototypes)} prototypes (dim={prototypes.shape[1]}) to {args.output}")
    print(f"File size: {args.output.stat().st_size} bytes")

    # Verification: pairwise cosine similarities
    print("\nPairwise cosine similarities:")
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            sim = float(np.dot(prototypes[i], prototypes[j]))
            print(f"  {classes[i][0]} <-> {classes[j][0]}: {sim:.4f}")


if __name__ == "__main__":
    main()
