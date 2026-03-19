#!/usr/bin/env python3
"""Check for train/test leakage between prototype and test images.

Hashes all files in prototype_images/ and test_images/, fails if any
hash appears in both sets.

Usage: pixi run check-leakage
"""

import hashlib
import sys
from pathlib import Path

PROTO_DIR = Path(__file__).parent.parent / "prototype_images"
TEST_DIR = Path(__file__).parent.parent / "test_images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def hash_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect_hashes(root: Path, skip_dirs: set[str] | None = None) -> dict[str, Path]:
    """Return {hash: path} for all images under root, skipping named dirs."""
    skip_dirs = skip_dirs or set()
    hashes = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file():
            if not any(part in skip_dirs for part in p.parts):
                hashes[hash_file(p)] = p
    return hashes


def main() -> int:
    proto_hashes = collect_hashes(PROTO_DIR)
    test_hashes = collect_hashes(TEST_DIR, skip_dirs={"catbox_captures"})

    overlap = set(proto_hashes) & set(test_hashes)
    if overlap:
        print(f"FAIL: {len(overlap)} duplicate image(s) found between prototype and test sets:\n")
        for h in sorted(overlap):
            print(f"  proto: {proto_hashes[h]}")
            print(f"  test:  {test_hashes[h]}")
            print()
        return 1

    print(f"OK: {len(proto_hashes)} prototype images, {len(test_hashes)} test images, 0 duplicates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
