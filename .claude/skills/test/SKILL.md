---
name: test
description: Run cat-detector tests (unit and integration)
user_invocable: true
---

# Test Skill

## Unit Tests (no external deps)
```bash
cargo test --lib
```

### Single module
```bash
cargo test --lib tracker
cargo test --lib detector
cargo test --lib config
```

### Single test
```bash
cargo test --lib test_name_here
```

## Integration Tests (needs ONNX runtime + model)
```bash
ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so cargo test --test integration_test
```

### Single integration test
```bash
ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so cargo test --test integration_test test_name_here
```

## Full Suite
```bash
cargo fmt --check && cargo clippy --all-features -- -D warnings && cargo test --lib && ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so cargo test --test integration_test
```
