---
name: deploy
description: Build, deploy to catbox, and restart the cat-detector service
user_invocable: true
---

# Deploy Skill

Two deployment methods available. The .deb method is preferred for production.

## Method 1: .deb Package (preferred)

1. **Build** the .deb packages:
```bash
./scripts/build-deb.sh
```

2. **Copy** packages to catbox:
```bash
scp cat-detector-models_*_all.deb cat-detector_*_amd64.deb catbox:/tmp/
```

3. **Install** on catbox (installs binary, models, config, service):
```bash
ssh catbox "sudo -n dpkg -i /tmp/cat-detector-models_*_all.deb /tmp/cat-detector_*_amd64.deb"
```

4. **Verify** the service is running:
```bash
ssh catbox "sudo -n systemctl status cat-detector"
```

### Notes
- First install creates config at `/etc/cat-detector/config.toml` from template
- Upgrades preserve config, restart service automatically
- Binary installs to `/usr/local/bin/cat-detector`
- Models install to `/opt/cat-detector/models/`
- Captures/sessions stored at `/var/lib/cat-detector/captures/`
- Requires `dpkg -i` in catbox NOPASSWD sudoers

## Method 2: rsync (legacy)

1. **Build** the release binary:
```bash
cargo build --release --features real-camera,web
```

2. **Deploy** via rsync:
```bash
./scripts/deploy.sh
```

3. **Restart** the systemd service:
```bash
ssh catbox "sudo -n systemctl restart cat-detector"
```

4. **Verify** the service is running:
```bash
ssh catbox "sudo -n systemctl status cat-detector"
```

## Common for both methods

- **Check logs** for errors:
```bash
ssh catbox "journalctl -u cat-detector --no-pager -n 20"
```

- Dashboard available at `http://catbox.local:8080` after deploy
- Camera auto-detects (`device_path = "auto"` in config)
- If the service fails, check logs with `journalctl -u cat-detector -f`
