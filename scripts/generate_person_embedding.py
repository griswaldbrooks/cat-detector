#!/usr/bin/env python3
"""Generate a person embedding compatible with the existing CLIP ONNX image encoder.

Since we can't reproduce the exact text encoder that generated the original
text embeddings, we use person images run through the image encoder to create
a person embedding in the same vector space.

Then we combine it with the existing cat/room embeddings into the new format.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS_DIR = Path(__file__).parent.parent / "models"
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"
CLIP_MODEL = MODELS_DIR / "clip_vitb32_image.onnx"
OLD_EMBEDDINGS = MODELS_DIR / "clip_text_embeddings.bin"
OUTPUT_PATH = MODELS_DIR / "clip_text_embeddings.bin"

# CLIP preprocessing constants
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CLIP_SIZE = 224

TEMPERATURE = 100.0
THRESHOLD = 0.5


def preprocess_clip(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = CLIP_SIZE / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - CLIP_SIZE) // 2
    top = (new_h - CLIP_SIZE) // 2
    img = img.crop((left, top, left + CLIP_SIZE, top + CLIP_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr


def get_image_embedding(session, input_name, image_path):
    inp = preprocess_clip(image_path)
    output = session.run(None, {input_name: inp})[0]
    features = output.flatten().astype(np.float32)
    features = features / np.linalg.norm(features)
    return features


def load_old_embeddings(path):
    """Load old format: raw f32 LE, [512 cat][512 nocat]"""
    data = np.fromfile(str(path), dtype=np.float32)
    dim = len(data) // 2
    return data[:dim], data[dim:]


def softmax_probs(sims, temperature=100.0):
    """Compute softmax probabilities over similarity scores."""
    sims = np.array(sims)
    max_s = sims.max()
    exps = np.exp((sims - max_s) * temperature)
    return exps / exps.sum()


def main():
    # Load old embeddings
    cat_emb, room_emb = load_old_embeddings(OLD_EMBEDDINGS)
    dim = len(cat_emb)
    print(f"Loaded old embeddings: dim={dim}")
    print(f"  cat norm: {np.linalg.norm(cat_emb):.6f}")
    print(f"  room norm: {np.linalg.norm(room_emb):.6f}")
    print(f"  cat-room similarity: {np.dot(cat_emb, room_emb):.4f}")

    session = ort.InferenceSession(str(CLIP_MODEL), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # First verify old embeddings work on test images
    print("\n--- Verifying old 2-class embeddings ---")
    cat_images = sorted(TEST_IMAGES_DIR.glob("cat_overhead_*.jpg"))[:5]
    nocat_images = sorted(TEST_IMAGES_DIR.glob("no_cat_*.jpg"))

    for img_path in cat_images + nocat_images:
        emb = get_image_embedding(session, input_name, img_path)
        cat_sim = float(np.dot(emb, cat_emb))
        room_sim = float(np.dot(emb, room_emb))
        probs = softmax_probs([cat_sim, room_sim])
        label = "CAT" if probs[0] >= THRESHOLD else "---"
        print(f"  {img_path.name:<40} cat={cat_sim:.4f} room={room_sim:.4f} prob={probs[0]:.4f} {label}")

    # Now generate person embedding from person images
    # We'll download a few person images or use a synthetic approach
    person_dir = TEST_IMAGES_DIR / "person_images"
    if not person_dir.exists():
        print(f"\n--- Need person images in {person_dir} ---")
        print("Creating directory. Please add person images and re-run.")
        print("Or we can try a different approach...")

        # Alternative: compute the person embedding as a direction in embedding space
        # We know where "cat" and "room" are. A "person" should be:
        # - Different from "cat" (not a cat)
        # - Different from "room" (not empty)
        # We can estimate this from the catbox captures that triggered on people
        print("\n--- Trying to find person-triggered captures from catbox ---")
        catbox_dir = TEST_IMAGES_DIR / "catbox_captures"
        if catbox_dir.exists():
            # Look for captures where detection triggered but there's no actual cat
            # These are likely person false positives
            all_captures = sorted(catbox_dir.glob("*.jpg")) + sorted(catbox_dir.glob("*.png"))
            print(f"  Found {len(all_captures)} catbox captures")

            # Show all capture embeddings with their cat/room similarities
            embeddings = []
            for img_path in all_captures:
                emb = get_image_embedding(session, input_name, img_path)
                cat_sim = float(np.dot(emb, cat_emb))
                room_sim = float(np.dot(emb, room_emb))
                embeddings.append((img_path, emb, cat_sim, room_sim))

            # Sort by cat probability (highest first) to find potential false positives
            embeddings.sort(key=lambda x: x[2] - x[3], reverse=True)

            print("\n  Top 20 captures by cat probability (potential false positives = people?):")
            for img_path, emb, cat_sim, room_sim in embeddings[:20]:
                probs = softmax_probs([cat_sim, room_sim])
                print(f"    {img_path.name:<50} cat={cat_sim:.4f} room={room_sim:.4f} prob={probs[0]:.4f}")

            print("\n  Bottom 10 (most room-like):")
            for img_path, emb, cat_sim, room_sim in embeddings[-10:]:
                probs = softmax_probs([cat_sim, room_sim])
                print(f"    {img_path.name:<50} cat={cat_sim:.4f} room={room_sim:.4f} prob={probs[0]:.4f}")

        person_dir.mkdir(exist_ok=True)
        print(f"\nPlease add person images to {person_dir}/ and re-run.")
        print("Alternatively, manually select captures that are people (not cats) to use as reference.")
        return

    person_images = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png")) + sorted(person_dir.glob("*.jpeg"))
    if not person_images:
        print(f"No images found in {person_dir}/")
        return

    print(f"\n--- Computing person embedding from {len(person_images)} images ---")
    person_embeddings = []
    for img_path in person_images:
        emb = get_image_embedding(session, input_name, img_path)
        cat_sim = float(np.dot(emb, cat_emb))
        room_sim = float(np.dot(emb, room_emb))
        print(f"  {img_path.name:<40} cat={cat_sim:.4f} room={room_sim:.4f}")
        person_embeddings.append(emb)

    # Average and normalize
    person_emb = np.mean(person_embeddings, axis=0).astype(np.float32)
    person_emb = person_emb / np.linalg.norm(person_emb)

    print(f"\nPerson embedding computed (norm={np.linalg.norm(person_emb):.6f})")
    print(f"  cat-person similarity: {np.dot(cat_emb, person_emb):.4f}")
    print(f"  room-person similarity: {np.dot(room_emb, person_emb):.4f}")

    # Test 3-class detection on test images
    print("\n--- Testing 3-class detection ---")
    all_test = sorted(TEST_IMAGES_DIR.glob("cat_overhead_*.jpg"))[:5] + nocat_images

    for img_path in all_test:
        emb = get_image_embedding(session, input_name, img_path)
        sims = [
            float(np.dot(emb, cat_emb)),
            float(np.dot(emb, room_emb)),
            float(np.dot(emb, person_emb)),
        ]
        probs = softmax_probs(sims)
        label = "CAT" if probs[0] >= THRESHOLD else "---"
        print(f"  {img_path.name:<40} cat={probs[0]:.4f} room={probs[1]:.4f} person={probs[2]:.4f} {label}")

    # Write new format embeddings
    all_embeddings = np.stack([cat_emb, room_emb, person_emb])
    n_embeds = all_embeddings.shape[0]

    with open(OUTPUT_PATH, "wb") as f:
        f.write(struct.pack("<II", n_embeds, dim))
        f.write(all_embeddings.astype(np.float32).tobytes())

    print(f"\nWritten {n_embeds} embeddings to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
