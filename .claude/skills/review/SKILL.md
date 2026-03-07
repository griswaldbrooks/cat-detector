---
name: review
description: Review recent changes for quality, correctness, and style
user_invocable: true
---

# Review Skill

## View Changes

### Unstaged changes
```bash
git diff
```

### Staged changes
```bash
git diff --cached
```

### Changes since last commit
```bash
git diff HEAD~1
```

### Changes on current branch vs main
```bash
git diff main...HEAD
```

## Checklist

- [ ] `cargo clippy --all-features -- -D warnings` passes
- [ ] `cargo test --lib` passes
- [ ] `cargo fmt --check` passes
- [ ] No `.unwrap()` outside tests
- [ ] Error types use `thiserror`
- [ ] New traits have mock implementations
- [ ] Feature gates are correct (`real-camera`, `web`)
- [ ] No secrets, `config.toml`, or binary files staged
