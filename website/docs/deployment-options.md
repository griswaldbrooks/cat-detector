---
sidebar_position: 3
title: Deployment Options
---

# Deployment Options Evaluation

Evaluated deployment strategies for cat-detector to achieve one-command deploy to any fresh x86_64 Linux machine without rebuilding from source. This should become a reusable pattern for future Rust+ML apps.

## Current State (v1.0.0)

`scripts/deploy.sh` builds locally and rsync's to catbox. System deps (ffmpeg, signal-cli, video group, systemd service user) are all manual. See the [Deployment Guide](./deployment.md) for the full guide.

## Runtime Dependencies

| Dependency | How it's used | Currently | Bundleable? |
|---|---|---|---|
| ONNX Runtime | Dynamic .so via `ORT_DYLIB_PATH` | Already bundled in `onnxruntime/` | Already done |
| CLIP model | File read at startup | Already bundled in `models/` | Already done |
| ffmpeg | Spawned as subprocess, frames piped to stdin | `apt install ffmpeg` | Yes (static builds available) |
| signal-cli | Spawned as subprocess (optional) | User installs to `~/bin/` | Yes, but requires interactive setup |
| V4L2 / camera | Kernel interface | Always present on Linux | Can't bundle (kernel level) |
| glibc | Linked dynamically | System libc | Effectively universal on Linux |

The only system dep requiring `apt install` is ffmpeg. Everything else is already bundled or kernel-level.

## Options Considered

### 1. Release Tarball + Setup Script

Ship `cat-detector-{version}-x86_64-linux.tar.gz` (~400MB) via GitHub releases. Contains binary, CLIP model, ORT runtime, embeddings, config template. A `setup` CLI subcommand guides users through remaining system-level requirements.

**Pros:** Simple, no tooling dependencies, works offline, fits within GitHub's 2GB release asset limit.

**Cons:** Doesn't automate system dep installation. Users run a few commands manually. Doesn't track what it installed for cleanup.

### 2. Static Bundling (tarball variant)

Same as Option 1, but bundle a static ffmpeg binary (~75MB) in the tarball. Eliminates all `apt install` commands. The only system-level requirements left are video group membership and systemd.

```
cat-detector-1.0.0/
├── bin/
│   ├── cat-detector
│   └── ffmpeg            # static build
├── lib/
│   └── libonnxruntime.so*
├── models/
│   ├── clip_vitb32_image.onnx
│   └── clip_text_embeddings.bin
├── config.example.toml
└── INSTALL.md
```

**Pros:** Zero system deps. No package manager involvement. No orphaned deps when upgrading. Simple upgrade path (replace the directory). ~450MB total.

**Cons:** Security updates for ffmpeg require a new release (can't `apt upgrade`). Licensing awareness needed (ffmpeg is LGPL, distributed as separate binary alongside MIT app). Larger download.

### 3. Ansible Playbook

SSH into target, install system deps, copy tarball, configure service, set permissions.

**Pros:** Fully automated end-to-end. Idempotent. Good for managing multiple targets.

**Cons:** Requires Ansible on the control machine. Overkill for a single target. More to maintain.

### 4. Container (Docker/Podman)

Ship a container image with everything baked in.

**Pros:** Fully self-contained, no host dep conflicts.

**Cons:** Camera device passthrough (`--device /dev/video0`) works but is less seamless. signal-cli needs volume mounts for user's Signal data. More indirection for debugging. Container runtime is itself a dependency.

### 5. Nix Flake + NixOS Module

Define the entire application as a Nix expression: package (binary + deps), NixOS module (systemd service, user config, video group), and dev shell.

On NixOS: `services.cat-detector.enable = true;` in system config, then `nixos-rebuild switch`. Atomic upgrades and rollbacks for free.

On non-NixOS (Ubuntu): `nix run github:griswaldbrooks/cat-detector` gets you the binary with deps, but not the systemd/permissions management.

**Pros:** Full dep isolation (`/nix/store` paths never conflict). Reproducible builds pinned via `flake.lock`. Atomic upgrades/rollbacks. Garbage collection removes unused deps. Dev environment matches production.

**Cons:** Steep learning curve for Nix language and concepts. Non-NixOS targets lose the declarative system config. 352MB model in Nix store is unusual. Build times on first run. Anyone you share with also needs Nix.

**Best if:** You're planning to adopt NixOS for catbox or future machines. The investment pays off across all apps, not just cat-detector.

### 6. AppImage

Single executable file containing the full application.

**Pros:** Single file, no extraction. No root to run. Familiar Linux format.

**Cons:** Designed for desktop GUI apps, not headless daemons. Awkward fit for systemd services, disk-writing apps (captures/sessions), and multi-subcommand CLIs. The format fights you more than it helps for this use case.

## Decision

Tabled for post-1.0.x. The leading candidates are:

- **Static bundling (Option 2)** for near-term: lowest friction, zero system deps, directly reusable as a pattern for future apps.
- **Nix (Option 5)** for longer-term: if adopting the Nix ecosystem broadly, it subsumes the deployment, dev environment, and CI story for all projects.

The key concern with tarball-based approaches is dependency lifecycle management — as the app evolves and deps change, removing old deps can get messy, and they might conflict with other apps on the same machine. Static bundling sidesteps this entirely (deps live in the app directory, not system-wide). Nix solves it structurally via content-addressed store paths.

## Related

- [Deployment Guide](./deployment.md) — current manual deployment guide
- `ROADMAP.md` — deployment tooling listed as long-term feature
- `scripts/deploy.sh` — current build+rsync deploy script
