---
name: capture
description: Capture a frame from catbox camera and display it
user_invocable: true
---

# Capture Skill

Grab the latest camera frame from catbox and display it.

## Steps

1. **Fetch frame** with a timestamped filename:
```bash
curl -s -o /tmp/catbox-$(date +%Y%m%d-%H%M%S).jpg http://catbox.local:8080/api/frame
```

2. **Display** the image using the Read tool on the saved file

## Notes
- Requires cat-detector service running on catbox
- Returns 503 if no frame is available yet (service just started)
