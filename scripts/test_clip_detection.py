#!/usr/bin/env python3
"""Test CLIP detection pipeline against test images using the same embeddings file.

Loads the binary embeddings file and CLIP image encoder to replicate
the Rust detection pipeline exactly.

Usage: pixi run test-clip
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from clip_utils import (
    CLIP_MODEL,
    CONFIDENCE_THRESHOLD,
    TEMPERATURE,
    TEST_IMAGES_DIR,
    TEXT_EMBEDDINGS_FILE,
    load_embeddings,
    preprocess_clip,
)

PROMPTS = [
    "a photo of a cat",
    "a photo of an empty room",
    "a photo of a person",
]


def main():
    embeddings = load_embeddings(TEXT_EMBEDDINGS_FILE)
    n_embeds, dim = embeddings.shape
    print(f"Loaded {n_embeds} embeddings (dim={dim})")
    for i, emb in enumerate(embeddings):
        label = PROMPTS[i] if i < len(PROMPTS) else f"class_{i}"
        print(f"  [{i}] {label}")

    session = ort.InferenceSession(str(CLIP_MODEL), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Collect all test images recursively (skip catbox_captures)
    image_paths = sorted(
        p for p in TEST_IMAGES_DIR.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and "catbox_captures" not in p.parts
    )

    print(f"\nTesting {len(image_paths)} images (threshold={CONFIDENCE_THRESHOLD})")
    print(f"{'Image':<55} {'cat_sim':>8} {'room_sim':>8} {'person_sim':>10} {'cat_prob':>8} {'2-class':>8} {'Result':>8}")
    print("-" * 115)

    cat_images = [p for p in image_paths if "cat" in p.stem.lower() and "no_cat" not in p.stem.lower()]
    nocat_images = [p for p in image_paths if "no_cat" in p.stem.lower()]
    other_images = [p for p in image_paths if p not in cat_images and p not in nocat_images]

    results = {"cat_tp": 0, "cat_fn": 0, "nocat_tn": 0, "nocat_fp": 0}

    for group_name, group, expect_cat in [("CAT IMAGES", cat_images, True),
                                           ("NO-CAT IMAGES", nocat_images, False),
                                           ("OTHER IMAGES", other_images, None)]:
        if not group:
            continue
        print(f"\n--- {group_name} ---")
        for img_path in group:
            inp = preprocess_clip(img_path)
            output = session.run(None, {input_name: inp})[0]
            features = output.flatten()
            features = features / np.linalg.norm(features)

            # Cosine similarities
            sims = [float(np.dot(features[:dim], embeddings[i])) for i in range(n_embeds)]

            # 3-class softmax
            max_sim = max(sims)
            exps = [np.exp((s - max_sim) * TEMPERATURE) for s in sims]
            cat_prob_3 = exps[0] / sum(exps)

            # 2-class softmax (cat vs room only, for comparison)
            max_sim_2 = max(sims[0], sims[1])
            cat_exp_2 = np.exp((sims[0] - max_sim_2) * TEMPERATURE)
            room_exp_2 = np.exp((sims[1] - max_sim_2) * TEMPERATURE)
            cat_prob_2 = cat_exp_2 / (cat_exp_2 + room_exp_2)

            detected = cat_prob_3 >= CONFIDENCE_THRESHOLD
            name = img_path.relative_to(TEST_IMAGES_DIR) if str(img_path).startswith(str(TEST_IMAGES_DIR)) else img_path.name

            marker = ""
            if expect_cat is True:
                if detected:
                    results["cat_tp"] += 1
                else:
                    results["cat_fn"] += 1
                    marker = " *** MISSED"
            elif expect_cat is False:
                if not detected:
                    results["nocat_tn"] += 1
                else:
                    results["nocat_fp"] += 1
                    marker = " *** FALSE POS"

            sim_strs = f"{sims[0]:>8.4f} {sims[1]:>8.4f} {sims[2]:>10.4f}"
            print(f"{str(name):<55} {sim_strs} {cat_prob_3:>8.4f} {cat_prob_2:>8.4f} {'CAT' if detected else '---':>8}{marker}")

    print(f"\n{'='*115}")
    print(f"Cat images:   {results['cat_tp']}/{results['cat_tp']+results['cat_fn']} detected "
          f"({results['cat_fn']} missed)")
    print(f"No-cat images: {results['nocat_tn']}/{results['nocat_tn']+results['nocat_fp']} correct "
          f"({results['nocat_fp']} false positives)")


if __name__ == "__main__":
    main()
