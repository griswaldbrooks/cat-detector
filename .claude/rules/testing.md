---
description: Testing conventions for cat-detector
globs:
  - "src/**/*.rs"
  - "tests/**/*.rs"
---

# Testing

## Unit Tests
- Place in `#[cfg(test)] mod tests` at the bottom of each source file
- Run with `cargo test --lib` (no external dependencies required)
- Use `Mock*` structs from the same module
- Use `tempfile` crate for filesystem tests
- Use `#[tokio::test]` for async tests

## Integration Tests
- Located in `tests/clip_integration_test.rs`
- Require CLIP model, text embeddings, and ORT runtime: `ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so cargo test --test clip_integration_test`
- Test real ClipDetector with actual model inference

## Naming Convention
- `test_<what>_<condition>_<expected>`
- Examples: `test_detect_cat_image_returns_detection`, `test_tracker_absent_to_present_on_threshold`

## What to Test
- All trait implementations (real and mock)
- State machine transitions in tracker
- Config parsing with valid/invalid/missing values
- Error paths and edge cases

## What Not to Test
- Private helper functions (test through public API)
- Third-party crate behavior
