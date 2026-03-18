"""Shared CLIP utilities for preprocessing and embedding I/O.

Used by test_clip_detection.py, compute_doc_values.py, and generate_image_embeddings.py.
"""

import struct
from pathlib import Path

import numpy as np
from PIL import Image

# CLIP preprocessing constants (OpenAI ViT-B/32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CLIP_SIZE = 224

# Detection parameters
TEMPERATURE = 100.0
CONFIDENCE_THRESHOLD = 0.5

# Default paths
MODELS_DIR = Path(__file__).parent.parent / "models"
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"
CLIP_MODEL = MODELS_DIR / "clip_vitb32_image.onnx"
TEXT_EMBEDDINGS_FILE = MODELS_DIR / "clip_text_embeddings.bin"


def preprocess_clip(image_path):
    """Replicate the Rust CLIP preprocessing: resize shortest side to 224, center crop, normalize."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = CLIP_SIZE / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Center crop
    left = (new_w - CLIP_SIZE) // 2
    top = (new_h - CLIP_SIZE) // 2
    img = img.crop((left, top, left + CLIP_SIZE, top + CLIP_SIZE))

    # To float32, normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    # HWC -> CHW, add batch dim
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr


def load_embeddings(path):
    """Load embeddings in the binary format: [count:u32 LE][dim:u32 LE][data:f32 LE...]"""
    with open(path, "rb") as f:
        data = f.read()

    count, dim = struct.unpack("<II", data[:8])
    floats = np.frombuffer(data[8:], dtype=np.float32).reshape(count, dim)
    return floats


def save_embeddings(embeddings, path):
    """Save embeddings in the binary format: [count:u32 LE][dim:u32 LE][data:f32 LE...]"""
    count, dim = embeddings.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", count, dim))
        f.write(embeddings.astype(np.float32).tobytes())


def softmax_with_temp(sims, temp=TEMPERATURE):
    """Compute softmax with temperature scaling."""
    max_sim = max(sims)
    exps = [np.exp((s - max_sim) * temp) for s in sims]
    total = sum(exps)
    return [e / total for e in exps]


def encode_image(session, image_path):
    """Encode a single image through CLIP and return L2-normalized 512-dim embedding."""
    input_name = session.get_inputs()[0].name
    inp = preprocess_clip(image_path)
    output = session.run(None, {input_name: inp})[0]
    features = output.flatten()
    features = features / np.linalg.norm(features)
    return features
