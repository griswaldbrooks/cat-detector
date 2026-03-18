"""Tests for few-shot image prototype embedding generation.

Requires CLIP ONNX model and test images.
Run with: pixi run python -m pytest scripts/tests/test_generate_image_embeddings.py -v
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

# Add scripts dir to path so we can import
sys.path.insert(0, str(Path(__file__).parent.parent))

from clip_utils import CLIP_MODEL, TEST_IMAGES_DIR, encode_image, load_embeddings, save_embeddings
from generate_image_embeddings import compute_prototype

PROTOTYPE_IMAGES_DIR = Path(__file__).parent.parent.parent / "prototype_images"


@pytest.fixture(scope="module")
def onnx_session():
    """Shared ONNX session for all tests."""
    if not CLIP_MODEL.exists():
        pytest.skip(f"CLIP model not found: {CLIP_MODEL}")
    return ort.InferenceSession(str(CLIP_MODEL), providers=["CPUExecutionProvider"])


class TestEncodeImage:
    def test_encode_single_image_shape_and_norm(self, onnx_session):
        """Encoding a single image produces a 512-dim L2-normalized vector."""
        test_img = TEST_IMAGES_DIR / "cat1.jpg"
        if not test_img.exists():
            pytest.skip("test image not found")

        emb = encode_image(onnx_session, test_img)

        assert emb.shape == (512,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_encode_different_images_differ(self, onnx_session):
        """Different images produce different embeddings."""
        img1 = TEST_IMAGES_DIR / "cat1.jpg"
        img2 = TEST_IMAGES_DIR / "no_cat_overhead.jpg"
        if not img1.exists() or not img2.exists():
            pytest.skip("test images not found")

        emb1 = encode_image(onnx_session, img1)
        emb2 = encode_image(onnx_session, img2)

        sim = float(np.dot(emb1, emb2))
        assert sim < 0.99, f"Expected different embeddings, got sim={sim}"


class TestComputePrototype:
    def test_prototype_is_normalized(self, onnx_session):
        """Prototype embedding from averaging should be L2-normalized."""
        cat_dir = PROTOTYPE_IMAGES_DIR / "cat_overhead"
        if not cat_dir.exists() or not list(cat_dir.glob("*.jpg")):
            pytest.skip("prototype images not curated yet")

        proto, count = compute_prototype(onnx_session, cat_dir)

        assert proto.shape == (512,)
        assert abs(np.linalg.norm(proto) - 1.0) < 1e-5
        assert count > 0

    def test_prototype_mean_of_normalized(self, onnx_session):
        """Prototype should be the re-normalized mean of normalized embeddings."""
        cat_dir = PROTOTYPE_IMAGES_DIR / "cat_overhead"
        if not cat_dir.exists() or not list(cat_dir.glob("*.jpg")):
            pytest.skip("prototype images not curated yet")

        proto, count = compute_prototype(onnx_session, cat_dir)

        # Manually compute for verification
        imgs = sorted(cat_dir.glob("*.jpg"))
        embs = [encode_image(onnx_session, p) for p in imgs]
        manual_mean = np.mean(embs, axis=0)
        manual_proto = manual_mean / np.linalg.norm(manual_mean)

        np.testing.assert_allclose(proto, manual_proto, atol=1e-6)

    def test_empty_directory_raises(self, onnx_session):
        """Should raise ValueError for directory with no images."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="No images found"):
                compute_prototype(onnx_session, Path(tmp))


class TestBinaryFormat:
    def test_save_load_roundtrip(self):
        """Saved embeddings should load back identically."""
        rng = np.random.default_rng(42)
        original = rng.standard_normal((4, 512)).astype(np.float32)
        original = original / np.linalg.norm(original, axis=1, keepdims=True)

        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            save_embeddings(original, f.name)
            loaded = load_embeddings(f.name)

        np.testing.assert_array_equal(original, loaded)

    def test_binary_format_matches_text_embeddings(self):
        """Image embeddings file should use the same binary format as text embeddings."""
        if not Path(TEST_IMAGES_DIR).parent.joinpath("models", "clip_text_embeddings.bin").exists():
            pytest.skip("text embeddings not found")

        text_emb = load_embeddings(Path(TEST_IMAGES_DIR).parent / "models" / "clip_text_embeddings.bin")

        # Verify format: create a 4-class file and check it's parseable
        rng = np.random.default_rng(42)
        fake_protos = rng.standard_normal((4, text_emb.shape[1])).astype(np.float32)
        fake_protos = fake_protos / np.linalg.norm(fake_protos, axis=1, keepdims=True)

        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            save_embeddings(fake_protos, f.name)

            # Read raw header
            with open(f.name, "rb") as fh:
                count, dim = struct.unpack("<II", fh.read(8))
            assert count == 4
            assert dim == text_emb.shape[1]

            loaded = load_embeddings(f.name)
            assert loaded.shape == (4, text_emb.shape[1])


class TestAcceptance:
    """Acceptance tests — these require curated prototype images and will fail until images are selected."""

    def test_4class_fixes_litter_box_false_positive(self, onnx_session):
        """Dirty litter box should NOT be detected as cat with image prototypes."""
        img = TEST_IMAGES_DIR / "litter_box_dirty_overhead_1.jpg"
        emb_file = Path(__file__).parent.parent.parent / "models" / "clip_image_embeddings.bin"
        if not img.exists():
            pytest.skip("litter box test image not found")
        if not emb_file.exists():
            pytest.skip("image embeddings not generated yet")

        embeddings = load_embeddings(emb_file)
        features = encode_image(onnx_session, img)
        sims = [float(np.dot(features, embeddings[i])) for i in range(len(embeddings))]
        max_sim = max(sims)
        exps = [np.exp((s - max_sim) * 100.0) for s in sims]
        cat_prob = exps[0] / sum(exps)

        assert cat_prob < 0.5, f"Litter box should not be cat, got {cat_prob:.4f}"

    def test_4class_fixes_litter_robot_false_positive(self, onnx_session):
        """Moving litter robot should NOT be detected as cat with image prototypes."""
        img = TEST_IMAGES_DIR / "litter_robot_moving_overhead_1.jpg"
        emb_file = Path(__file__).parent.parent.parent / "models" / "clip_image_embeddings.bin"
        if not img.exists():
            pytest.skip("litter robot test image not found")
        if not emb_file.exists():
            pytest.skip("image embeddings not generated yet")

        embeddings = load_embeddings(emb_file)
        features = encode_image(onnx_session, img)
        sims = [float(np.dot(features, embeddings[i])) for i in range(len(embeddings))]
        max_sim = max(sims)
        exps = [np.exp((s - max_sim) * 100.0) for s in sims]
        cat_prob = exps[0] / sum(exps)

        assert cat_prob < 0.5, f"Litter robot should not be cat, got {cat_prob:.4f}"

    def test_4class_fixes_person_false_positive(self, onnx_session):
        """Overhead person at catbox should NOT be detected as cat."""
        img = TEST_IMAGES_DIR / "person_overhead_catbox_1.jpg"
        emb_file = Path(__file__).parent.parent.parent / "models" / "clip_image_embeddings.bin"
        if not img.exists():
            pytest.skip("person test image not found")
        if not emb_file.exists():
            pytest.skip("image embeddings not generated yet")

        embeddings = load_embeddings(emb_file)
        features = encode_image(onnx_session, img)
        sims = [float(np.dot(features, embeddings[i])) for i in range(len(embeddings))]
        max_sim = max(sims)
        exps = [np.exp((s - max_sim) * 100.0) for s in sims]
        cat_prob = exps[0] / sum(exps)

        assert cat_prob < 0.5, f"Person should not be cat, got {cat_prob:.4f}"

    @pytest.mark.xfail(reason="Known limitation: litter box dominates when cat and litter box coexist. Needs more litter box prototype variety or temperature tuning.")
    def test_4class_fixes_cat_with_litter_box_false_negative(self, onnx_session):
        """Cat next to litter box should still be detected as cat."""
        img = TEST_IMAGES_DIR / "cat_with_litter_box_overhead_1.jpg"
        emb_file = Path(__file__).parent.parent.parent / "models" / "clip_image_embeddings.bin"
        if not img.exists():
            pytest.skip("cat+litter test image not found")
        if not emb_file.exists():
            pytest.skip("image embeddings not generated yet")

        embeddings = load_embeddings(emb_file)
        features = encode_image(onnx_session, img)
        sims = [float(np.dot(features, embeddings[i])) for i in range(len(embeddings))]
        max_sim = max(sims)
        exps = [np.exp((s - max_sim) * 100.0) for s in sims]
        cat_prob = exps[0] / sum(exps)

        assert cat_prob >= 0.5, f"Cat with litter box should be detected, got {cat_prob:.4f}"


class TestCLI:
    def test_cli_runs_with_help(self):
        """Script should accept --help without error."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "generate_image_embeddings.py"), "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "Generate few-shot" in result.stdout

    def test_cli_missing_directory_exits(self):
        """Script should exit with error for nonexistent directory."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "generate_image_embeddings.py"),
             "--classes", "test:/nonexistent/path"],
            capture_output=True, text=True
        )
        assert result.returncode != 0
