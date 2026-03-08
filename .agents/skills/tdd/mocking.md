# When to Mock

## Trait-Based Mocking

This project uses **trait-based dependency injection**. Mock at trait boundaries:

- `CameraCapture` → `MockCamera`, `StubCamera`
- `CatDetector` → `MockDetector`
- `ImageStorage` → `MockStorage`
- `Notifier` → `MockNotifier`

These are system boundaries — each trait abstracts over hardware (camera), ML runtime (ONNX), filesystem (storage), or external processes (signal-cli).

## When to Mock

- External hardware: camera via `MockCamera`
- ML inference: detector via `MockDetector`
- Filesystem operations: storage via `MockStorage`
- External processes: notifier via `MockNotifier`
- Time: use fixed timestamps in tests

## When NOT to Mock

- `CatTracker` — pure logic, test directly with real inputs
- `Config` — parse real TOML strings, don't mock
- Postprocessing (NMS, bbox scaling) — pure functions, test with real data
- Anything that's fast and deterministic

## Mock Conventions

Place `Mock*` structs in the same file as the trait, inside `#[cfg(test)]`:

```rust
#[cfg(test)]
pub struct MockStorage {
    saved: std::sync::Mutex<Vec<SavedImage>>,
    should_fail: bool,
}

#[cfg(test)]
#[async_trait]
impl ImageStorage for MockStorage {
    async fn save_image(&self, ...) -> Result<SavedImage, StorageError> {
        if self.should_fail {
            return Err(StorageError::WriteError("mock failure".into()));
        }
        // Record the call and return success
    }
}
```

Keep mocks minimal — only implement what tests need.
