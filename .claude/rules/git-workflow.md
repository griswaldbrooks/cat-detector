---
description: Git workflow conventions for the cat-detector project
---

# Git Workflow

## Commit Messages
- Use imperative mood: "Add feature" not "Added feature"
- First line under 72 characters
- Add `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>` on AI-assisted commits

## Before Committing
1. `cargo fmt --check` - formatting
2. `cargo clippy --all-features -- -D warnings` - lints
3. `cargo test --lib` - unit tests pass
4. Review staged changes with `git diff --cached`

## Never Commit
- `config.toml` (user-specific config; `config.example.toml` is tracked)
- `test_images/` (local test data)
- `models/*.onnx` (large binary files)
- `onnxruntime/` (local runtime installation)
- `.env` or files containing secrets

## Branching
- `main` is the primary branch
- Create feature branches for non-trivial changes
