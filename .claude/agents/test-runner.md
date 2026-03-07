---
model: sonnet
tools:
  - Bash
  - Read
skills:
  - build
  - test
---

# Test Runner Agent

Runs the full build and test suite, reports results.

## Steps

1. `cargo fmt --check` - Check formatting
2. `cargo clippy --all-features -- -D warnings` - Lint check
3. `cargo test --lib` - Unit tests
4. `ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so cargo test --test integration_test` - Integration tests (if runtime available)

## Output Format

```
## Results
- fmt: PASS/FAIL
- clippy: PASS/FAIL (N warnings)
- unit tests: PASS/FAIL (N passed, M failed)
- integration tests: PASS/FAIL/SKIPPED
```

If any step fails, include the relevant error output.
