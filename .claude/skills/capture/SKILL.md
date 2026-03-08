---
name: capture
description: Capture a frame from catbox camera and display it
user_invocable: true
---

# Capture Skill

Grab camera frames from catbox, display them, and optionally save to the test set.

## Arguments

- No args: capture and display a single frame
- `N` (number): capture N frames, 2 seconds apart, and display all
- `save <name>`: capture a frame and save to `test_images/<name>.jpg`
- `batch <name> [N]`: capture N frames (default 3), 2s apart, save to `test_images/<name>1.jpg` etc., and display all
- `test <name>`: capture a frame, save to test_images, AND run detection on it with threshold 0.1

## Steps

### Single capture (default)

1. Fetch frame with timestamped filename:
```bash
curl -s -o /tmp/catbox-$(date +%Y%m%d-%H%M%S).jpg http://catbox.local:8080/api/frame
```
2. Display the image using the Read tool

### Batch capture (`N` or `batch`)

1. Fetch N frames, 2 seconds apart. Use chained `curl && sleep && curl` commands (NOT shell loops, which break permission matching):
```bash
curl -s -o /tmp/catbox-batch-1.jpg http://catbox.local:8080/api/frame && sleep 2 && curl -s -o /tmp/catbox-batch-2.jpg http://catbox.local:8080/api/frame && sleep 2 && curl -s -o /tmp/catbox-batch-3.jpg http://catbox.local:8080/api/frame
```
2. Display all frames using the Read tool
3. If `batch <name>` mode, copy to test_images:
```bash
cp /tmp/catbox-batch-$i.jpg test_images/<name>$i.jpg
```

### Test capture (`test`)

1. Capture and save to test_images as above
2. Copy to catbox and run detection:
```bash
scp test_images/<name>.jpg catbox:/tmp/<name>.jpg
ssh catbox "cd ~/cat-detector && ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so ./cat-detector test-image /tmp/<name>.jpg --model models/yolo11n.onnx --threshold 0.1"
```

## Notes
- Requires cat-detector service running on catbox
- Returns 503 if no frame is available yet (service just started)
- The `test` mode is useful for checking if the model detects objects in a live frame
