#!/usr/bin/env python3
"""Generate CLIP text embeddings for cat detection.

Uses the original OpenAI CLIP package to ensure text embeddings are in the
same space as the ONNX image encoder (clip_vitb32_image.onnx).

Output format: [count: u32 LE][dim: u32 LE][embed_0: f32 x dim]...[embed_N: f32 x dim]
First embedding is the positive class (cat), rest are negatives.
"""

import struct
from pathlib import Path

import clip
import numpy as np
import torch

PROMPTS = [
    "a photo of a cat",           # positive class (index 0)
    "a photo of an empty room",   # negative
    "a photo of a person",        # negative
]

OUTPUT_PATH = Path(__file__).parent.parent / "models" / "clip_text_embeddings.bin"


def main():
    print("Loading OpenAI CLIP ViT-B/32...")
    model, _ = clip.load("ViT-B/32", device="cpu")
    model.eval()

    print(f"Encoding {len(PROMPTS)} text prompts:")
    for i, p in enumerate(PROMPTS):
        print(f"  [{i}] {p}")

    tokens = clip.tokenize(PROMPTS)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        # L2 normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    embeddings = text_features.cpu().float().numpy()
    n_embeds, dim = embeddings.shape
    print(f"Embedding shape: {n_embeds} x {dim}")

    # Write binary: [count: u32][dim: u32][embeddings: f32...]
    with open(OUTPUT_PATH, "wb") as f:
        f.write(struct.pack("<II", n_embeds, dim))
        f.write(embeddings.tobytes())

    file_size = OUTPUT_PATH.stat().st_size
    print(f"Written to {OUTPUT_PATH} ({file_size} bytes)")

    # Verify
    with open(OUTPUT_PATH, "rb") as f:
        count, dim_check = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(count, dim_check)

    print(f"Verified: {count} embeddings, dim={dim_check}")
    for i in range(count):
        norm = np.linalg.norm(data[i])
        print(f"  [{i}] norm={norm:.6f} ({PROMPTS[i]})")

    # Also verify the old 2-class embeddings still match
    old_path = OUTPUT_PATH.with_suffix(".bin.bak")
    if old_path.exists():
        old_data = np.fromfile(str(old_path), dtype=np.float32).reshape(2, -1)
        for i in range(min(2, count)):
            cos_sim = np.dot(data[i], old_data[i]) / (np.linalg.norm(data[i]) * np.linalg.norm(old_data[i]))
            print(f"  [{i}] similarity to old embedding: {cos_sim:.6f}")

    # Show pairwise similarities
    print("\nPairwise cosine similarities:")
    for i in range(count):
        for j in range(i + 1, count):
            sim = np.dot(data[i], data[j])
            print(f"  {PROMPTS[i]} <-> {PROMPTS[j]}: {sim:.4f}")


if __name__ == "__main__":
    main()
