---
description: Rust coding conventions for cat-detector
globs:
  - "src/**/*.rs"
---

# Rust Style

## Error Handling
- `thiserror` for module-level error enums (e.g., `CameraError`, `DetectorError`)
- `anyhow` only in `main.rs` for top-level error propagation
- Use `?` operator; avoid `.unwrap()` outside tests

## Traits & DI
- Define traits for all major components (camera, detector, storage, notifier)
- Use `#[async_trait]` with `Send + Sync` bounds
- Generic `App<C, D, S, N>` accepts any implementation

## Concurrency
- `Arc` for shared ownership across async tasks
- `std::sync::Mutex` (not tokio) for ONNX session (not held across await)
- `tokio::sync::RwLock` for shared state accessed from web handlers

## Mock Implementations
- Place `Mock*` structs in the same file as the trait, inside `#[cfg(test)] mod tests`
- Keep mocks minimal - only implement what tests need

## Feature Gates
- `#[cfg(feature = "real-camera")]` for V4L2 camera code
- `#[cfg(feature = "web")]` for Axum web dashboard
- Stub/mock implementations available without features

## General
- `tracing` crate for logging (`info!`, `warn!`, `error!`, `debug!`)
- `cargo fmt` defaults (no rustfmt.toml overrides)
- Prefer `impl Trait` in function args when the concrete type doesn't matter
