#!/usr/bin/env python3
"""Compute fresh values for the CLIP detection documentation.

Runs the CLIP pipeline on representative test images and prints:
- Text embedding pairwise cosine similarities and L2 distances
- Per-image cosine similarities, softmax probabilities
- Raw 512-dim embedding vectors for collapsible doc sections

Usage: pixi run python scripts/compute_doc_values.py
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from clip_utils import (
    CLIP_MODEL,
    TEMPERATURE,
    TEST_IMAGES_DIR,
    TEXT_EMBEDDINGS_FILE,
    encode_image,
    load_embeddings,
    preprocess_clip,
    softmax_with_temp,
)

PROMPTS = [
    "a photo of a cat",
    "a photo of an empty room",
    "a photo of a person",
]

# Representative images for documentation
DOC_IMAGES = [
    "cat1.jpg",
    "cat_overhead_litterbot1.jpg",
    "cat_overhead_tabby1.jpg",
    "no_cat_overhead.jpg",
    "litter_box_dirty_overhead_1.jpg",
    "litter_robot_moving_overhead_1.jpg",
    "cat_with_litter_box_overhead_1.jpg",
]

PERSON_IMAGES = [
    "person_images/person_overhead_1.jpg",
    "person_overhead_catbox_1.jpg",
]


def main():
    embeddings = load_embeddings(TEXT_EMBEDDINGS_FILE)
    n_embeds, dim = embeddings.shape

    # === Text embedding analysis ===
    print("=" * 80)
    print("TEXT EMBEDDING ANALYSIS")
    print("=" * 80)

    print(f"\nEmbedding count: {n_embeds}, dimension: {dim}")
    for i in range(n_embeds):
        norm = np.linalg.norm(embeddings[i])
        print(f"  [{i}] \"{PROMPTS[i]}\" norm={norm:.6f}")

    print("\nPairwise cosine similarities:")
    for i in range(n_embeds):
        for j in range(i + 1, n_embeds):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            print(f"  \"{PROMPTS[i]}\" <-> \"{PROMPTS[j]}\": {sim:.4f}")

    print("\nPairwise L2 distances:")
    for i in range(n_embeds):
        for j in range(i + 1, n_embeds):
            dist = float(np.linalg.norm(embeddings[i] - embeddings[j]))
            print(f"  \"{PROMPTS[i]}\" <-> \"{PROMPTS[j]}\": {dist:.4f}")

    print("\nCosine similarity matrix:")
    print(f"{'':>30}", end="")
    for j in range(n_embeds):
        print(f"  {PROMPTS[j]:>20}", end="")
    print()
    for i in range(n_embeds):
        print(f"{PROMPTS[i]:>30}", end="")
        for j in range(n_embeds):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            print(f"  {sim:>20.4f}", end="")
        print()

    # === Raw text embedding vectors ===
    print("\n" + "=" * 80)
    print("RAW TEXT EMBEDDING VECTORS (for collapsible sections)")
    print("=" * 80)
    for i in range(n_embeds):
        print(f"\n--- \"{PROMPTS[i]}\" ---")
        vec = embeddings[i]
        # Print in rows of 16
        for row_start in range(0, dim, 16):
            vals = vec[row_start:row_start + 16]
            print(" ".join(f"{v:>9.6f}" for v in vals))

    # === Image inference ===
    print("\n" + "=" * 80)
    print("IMAGE INFERENCE RESULTS")
    print("=" * 80)

    session = ort.InferenceSession(str(CLIP_MODEL), providers=["CPUExecutionProvider"])

    all_images = DOC_IMAGES + PERSON_IMAGES
    for img_name in all_images:
        img_path = TEST_IMAGES_DIR / img_name
        if not img_path.exists():
            print(f"\nSKIPPED (not found): {img_name}")
            continue

        features = encode_image(session, img_path)

        sims = [float(np.dot(features[:dim], embeddings[i])) for i in range(n_embeds)]
        probs = softmax_with_temp(sims, TEMPERATURE)

        print(f"\n--- {img_name} ---")
        print(f"  Cosine similarities:")
        for i in range(n_embeds):
            print(f"    {PROMPTS[i]:>30}: {sims[i]:.4f}")
        print(f"  Softmax probabilities (tau=100):")
        for i in range(n_embeds):
            print(f"    {PROMPTS[i]:>30}: {probs[i]:.4f} ({probs[i]*100:.1f}%)")
        detected = probs[0] >= 0.5
        print(f"  Detection: {'CAT DETECTED' if detected else 'No detection'} (cat_prob={probs[0]:.4f})")

        # Raw image embedding
        print(f"  Raw 512-dim image embedding (first 32 values):")
        print("   ", " ".join(f"{v:>9.6f}" for v in features[:32]))

    print("\n" + "=" * 80)
    print("SUMMARY TABLE FOR DOCUMENTATION")
    print("=" * 80)
    print(f"\n{'Image':<45} {'cat_sim':>8} {'room_sim':>8} {'person_sim':>10} {'cat_prob':>9} {'Result':>8}")
    print("-" * 98)
    for img_name in all_images:
        img_path = TEST_IMAGES_DIR / img_name
        if not img_path.exists():
            continue
        features = encode_image(session, img_path)
        sims = [float(np.dot(features[:dim], embeddings[i])) for i in range(n_embeds)]
        probs = softmax_with_temp(sims, TEMPERATURE)
        detected = probs[0] >= 0.5
        print(f"{img_name:<45} {sims[0]:>8.4f} {sims[1]:>8.4f} {sims[2]:>10.4f} {probs[0]:>8.4f}  {'CAT' if detected else '---':>6}")


if __name__ == "__main__":
    main()
