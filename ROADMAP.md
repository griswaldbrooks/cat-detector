# Roadmap

Long-term feature ideas for cat-detector. Not prioritized or scheduled — just things worth building someday.

## Detection History & Analytics

Track visit patterns over time. When do the cats visit most? How long do they stay? Surface trends on the dashboard — daily/weekly visit counts, average duration, peak hours heatmap.

## Dashboard Enhancements

- Live detection overlay on the MJPEG stream (bounding box, confidence score)
- Session timeline view (visual history of visits on a time axis)
- Real-time charts for detection confidence and visit frequency

## Notification Improvements

- Per-cat alerts (different messages or channels depending on which cat is detected)
- Quiet hours (suppress notifications during configured time windows)
- Notification history in the dashboard

## Deployment Tooling

Bundle the full stack (binary + CLIP model + ONNX Runtime + text embeddings + scripts + example config) into a self-contained release artifact. One-command deploy to any fresh x86_64 Linux machine without rebuilding from source. GitHub release assets support up to 2GB, so this is feasible. Evaluate: release tarball, Ansible, Nix, container image. This should become a reusable pattern for future Rust+ML apps.
