# Good and Bad Tests

## Good Tests

**Integration-style**: Test through real trait interfaces, not internals.

```rust
// GOOD: Tests observable behavior through the trait
#[tokio::test]
async fn test_cat_enter_saves_image_and_notifies() {
    let camera = MockCamera::from_solid_colors(vec![[255, 0, 0]], 100, 100);
    let detector = MockDetector::always_detect_cat();
    let storage = MockStorage::new();
    let notifier = MockNotifier::new();

    let mut app = App::new(camera, detector, storage.clone(), notifier.clone(), config);
    app.run_once().await.unwrap();

    assert_eq!(storage.saved_images().len(), 1);
    assert_eq!(notifier.notifications().len(), 1);
}
```

Characteristics:

- Tests behavior through public trait interfaces
- Uses `Mock*` structs that implement the real traits
- Survives internal refactors
- Describes WHAT, not HOW
- One logical assertion per test
- Named `test_<what>_<condition>_<expected>`

## Bad Tests

**Implementation-detail tests**: Coupled to internal structure.

```rust
// BAD: Tests that a specific internal method was called
#[test]
fn test_postprocess_calls_non_max_suppression() {
    // Testing private method ordering, not behavior
}

// BAD: Reaching into private state
#[test]
fn test_tracker_internal_counter_value() {
    let tracker = CatTracker::new(config);
    tracker.update(true);
    assert_eq!(tracker.detection_count, 1); // Testing internals!
}
```

Red flags:

- Testing private methods or fields
- Test breaks when refactoring without behavior change
- Test name describes HOW not WHAT
- Asserting on internal counters/state instead of observable output

```rust
// BAD: Bypasses interface to verify
#[tokio::test]
async fn test_storage_writes_to_disk() {
    storage.save_image(&img, ImageType::Entry, timestamp).await.unwrap();
    assert!(std::fs::read_dir(dir).unwrap().count() > 0); // Checking fs directly
}

// GOOD: Verifies through interface
#[tokio::test]
async fn test_saved_image_has_correct_metadata() {
    let result = storage.save_image(&img, ImageType::Entry, timestamp).await.unwrap();
    assert_eq!(result.image_type, ImageType::Entry);
    assert!(result.path.exists());
}
```
