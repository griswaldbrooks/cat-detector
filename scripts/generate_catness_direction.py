#!/usr/bin/env python3
"""Generate catness direction binary file for linear probe detection.

Computes the catness direction as normalize(cat_mean - noncat_mean) from prototype
images. The direction vector isolates the ~46 dimensions that carry cat signal,
producing clean separation even when scene backgrounds are nearly identical.

Output format: [dim: u32 LE][direction: f32 x dim]

Usage:
    pixi run -e clip generate-catness-direction

    # Or manually:
    python scripts/generate_catness_direction.py \
        --cat-dir prototype_images/cat_overhead \
        --noncat-dirs prototype_images/empty_room_overhead \
                      prototype_images/person_overhead \
                      prototype_images/litter_box_overhead
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from clip_utils import (
    CLIP_MODEL,
    MODELS_DIR,
    TEST_IMAGES_DIR,
    encode_image,
)
from generate_image_embeddings import compute_prototype


def save_catness_direction(direction, path):
    """Save direction vector in binary format: [dim: u32 LE][direction: f32 x dim]"""
    dim = len(direction)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", dim))
        f.write(np.array(direction, dtype=np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Generate catness direction binary file"
    )
    parser.add_argument(
        "--cat-dir",
        type=Path,
        default=Path("prototype_images/cat_overhead"),
        help="Directory of cat prototype images",
    )
    parser.add_argument(
        "--noncat-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("prototype_images/empty_room_overhead"),
            Path("prototype_images/person_overhead"),
            Path("prototype_images/litter_box_overhead"),
        ],
        help="Directories of non-cat prototype images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "catness_direction.bin",
        help="Output binary file path",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=CLIP_MODEL,
        help="CLIP ONNX model path",
    )
    args = parser.parse_args()

    # Validate directories
    if not args.cat_dir.exists():
        print(f"Error: cat directory not found: {args.cat_dir}", file=sys.stderr)
        sys.exit(1)
    for d in args.noncat_dirs:
        if not d.exists():
            print(f"Error: non-cat directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading CLIP model: {args.model}")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    # Compute cat prototype (mean of all cat image embeddings)
    print(f"\nCat prototypes from: {args.cat_dir}")
    cat_proto, cat_count = compute_prototype(session, args.cat_dir)
    print(f"  {cat_count} images")

    # Compute non-cat mean (mean of all non-cat image embeddings across all dirs)
    print(f"\nNon-cat prototypes from:")
    all_noncat_embs = []
    for d in args.noncat_dirs:
        image_paths = sorted(
            list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
        )
        for img_path in image_paths:
            emb = encode_image(session, img_path)
            all_noncat_embs.append(emb)
        print(f"  {d}: {len(image_paths)} images")

    if not all_noncat_embs:
        print("Error: no non-cat images found", file=sys.stderr)
        sys.exit(1)

    noncat_mean = np.mean(all_noncat_embs, axis=0)
    noncat_mean = noncat_mean / np.linalg.norm(noncat_mean)
    print(f"  Total: {len(all_noncat_embs)} non-cat images")

    # Compute catness direction: normalize(cat_mean - noncat_mean)
    raw_direction = cat_proto - noncat_mean
    direction = raw_direction / np.linalg.norm(raw_direction)

    print(f"\nCatness direction: dim={len(direction)}, norm={np.linalg.norm(direction):.6f}")
    print(f"Raw diff norm: {np.linalg.norm(raw_direction):.6f}")

    # Save
    save_catness_direction(direction, args.output)
    print(f"Saved to {args.output} ({args.output.stat().st_size} bytes)")

    # Validate against test images
    print("\n--- Validation against test images ---")
    test_dirs = {
        "cat_overhead": TEST_IMAGES_DIR / "cat_overhead",
        "cat_tabby": TEST_IMAGES_DIR / "cat_tabby",
        "empty_room": TEST_IMAGES_DIR / "empty_room",
        "person_overhead": TEST_IMAGES_DIR / "person_overhead",
        "litter_box": TEST_IMAGES_DIR / "litter_box",
    }

    for label, test_dir in test_dirs.items():
        if not test_dir.exists():
            continue
        image_paths = sorted(
            list(test_dir.glob("*.jpg"))
            + list(test_dir.glob("*.jpeg"))
            + list(test_dir.glob("*.png"))
        )
        if not image_paths:
            continue

        scores = []
        for img_path in image_paths:
            emb = encode_image(session, img_path)
            score = float(np.dot(emb, direction))
            scores.append(score)

        min_s, max_s, mean_s = min(scores), max(scores), np.mean(scores)
        print(f"  {label:20s}: n={len(scores):2d}, min={min_s:+.4f}, max={max_s:+.4f}, mean={mean_s:+.4f}")


if __name__ == "__main__":
    main()
