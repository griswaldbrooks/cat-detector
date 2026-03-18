---
name: health
description: Check catbox service health — status, disk, logs, camera, notifications
user_invocable: true
---

# Health Check Skill

Quick health check of the cat-detector service running on catbox.

## Arguments

- No args: run all checks
- `service`: just service status
- `disk`: just disk usage
- `logs`: just recent logs
- `camera`: just camera status
- `notifications`: just notification delivery test

## Steps

### 1. Service status

```bash
ssh catbox "systemctl status cat-detector --no-pager -l"
```

Report: running/stopped/failed, uptime, memory usage.

### 2. Disk usage

```bash
ssh catbox "df -h /var/lib/cat-detector && echo '---' && du -sh /var/lib/cat-detector/captures/ 2>/dev/null && echo '---' && ls /var/lib/cat-detector/captures/sessions/ 2>/dev/null | wc -l"
```

Report: filesystem usage %, captures directory size, session count.

### 3. Recent logs

```bash
ssh catbox "journalctl -u cat-detector --no-pager -n 30 --since '10 minutes ago'"
```

Report: any errors or warnings, detection activity, recent sessions.

### 4. Camera status

```bash
curl -s -o /dev/null -w "%{http_code}" http://catbox.local:8080/api/frame
```

Report: 200 = camera working, 503 = no frame yet, connection refused = service down.

### 5. Dashboard reachability

```bash
curl -s -o /dev/null -w "%{http_code}" http://catbox.local:8080/
```

### 6. Notification status (only if explicitly requested)

Only run this if the user asks for `notifications` — it sends a real Signal message:

```bash
ssh catbox "ORT_DYLIB_PATH=/opt/cat-detector/lib/libonnxruntime.so /usr/local/bin/cat-detector test-notification"
```

## Output Format

Summarize results as a compact table:

| Check | Status | Details |
|-------|--------|---------|
| Service | ✅ running | uptime 3d, 45MB RSS |
| Disk | ✅ 23% used | 1.2GB captures, 47 sessions |
| Logs | ✅ healthy | 12 detections last 10min, no errors |
| Camera | ✅ streaming | HTTP 200 |
| Dashboard | ✅ reachable | HTTP 200 |

Use ⚠️ for warnings (disk >80%, errors in logs) and ❌ for failures.
