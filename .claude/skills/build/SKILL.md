---
name: build
description: Build the cat-detector project with various configurations
user_invocable: true
---

# Build Skill

## Commands

### Full release build (all features)
```bash
cargo build --release --features real-camera,web
```

### Debug build (fast iteration)
```bash
cargo build --features real-camera,web
```

### Check only (fastest, no codegen)
```bash
cargo check --all-features
```

### Clippy lint check
```bash
cargo clippy --all-features -- -D warnings
```

### Format check
```bash
cargo fmt --check
```

### Build .deb packages
```bash
./scripts/build-deb.sh
```
Produces `cat-detector_VERSION_amd64.deb` and `cat-detector-models_VERSION_all.deb`.
Requires `fpm` (`gem install fpm`).

## Notes
- The `real-camera` feature requires v4l2 dev libraries (`libv4l-dev`)
- The `web` feature pulls in axum and tower dependencies
- Use `--release` for performance testing (CLIP inference is ~10x slower in debug)
