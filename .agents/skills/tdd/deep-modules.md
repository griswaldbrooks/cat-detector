# Deep Modules

From "A Philosophy of Software Design":

**Deep module** = small interface + lots of implementation

```
┌─────────────────────┐
│   Small Interface   │  ← Few methods, simple params
├─────────────────────┤
│                     │
│                     │
│  Deep Implementation│  ← Complex logic hidden
│                     │
│                     │
└─────────────────────┘
```

**Shallow module** = large interface + little implementation (avoid)

```
┌─────────────────────────────────┐
│       Large Interface           │  ← Many methods, complex params
├─────────────────────────────────┤
│  Thin Implementation            │  ← Just passes through
└─────────────────────────────────┘
```

## Examples in This Project

**Deep**: `ClipDetector` — simple `detect(&DynamicImage) -> Vec<Detection>` interface hides CLIP preprocessing, ONNX inference, cosine similarity, and softmax classification.

**Deep**: `CatTracker` — simple `update(bool) -> Option<TrackerEvent>` hides hysteresis state machine with enter/exit thresholds and timing.

**Shallow (avoid)**: A notifier that just wraps `Command::new("signal-cli")` with no validation, timeout, or retry logic adds interface without depth.

When designing interfaces, ask:

- Can I reduce the number of methods?
- Can I simplify the parameters?
- Can I hide more complexity inside?
