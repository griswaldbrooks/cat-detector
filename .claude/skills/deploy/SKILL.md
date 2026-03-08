---
name: deploy
description: Build, deploy to catbox, and restart the cat-detector service
user_invocable: true
---

# Deploy Skill

Build locally, deploy to catbox (Optiplex 3040M), and restart the service.

## Steps

1. **Build** the release binary:
```bash
cargo build --release --features real-camera,web
```

2. **Deploy** binary, config, models, and ONNX runtime:
```bash
./scripts/deploy.sh
```

3. **Restart** the systemd service on catbox:
```bash
ssh catbox "sudo -n systemctl restart cat-detector"
```

4. **Verify** the service is running:
```bash
ssh catbox "sudo -n systemctl status cat-detector"
```

5. **Check logs** for errors:
```bash
ssh catbox "journalctl -u cat-detector --no-pager -n 20"
```

## Notes
- Deploy target is `~/cat-detector` on catbox (SSH config alias)
- Dashboard available at `http://catbox.local:8080` after deploy
- Camera auto-detects (`device_path = "auto"` in config)
- If the service fails, check logs with `journalctl -u cat-detector -f`
- To deploy without restarting, just run steps 1-2
