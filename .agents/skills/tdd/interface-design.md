# Interface Design for Testability

Good interfaces make testing natural. This project uses trait-based DI:

1. **Accept dependencies via generics, don't create them**

   ```rust
   // Testable: App accepts any implementation of its trait bounds
   pub struct App<C: CameraCapture, D: CatDetector, S: ImageStorage, N: Notifier> {
       camera: C,
       detector: D,
       storage: S,
       notifier: N,
   }

   // Hard to test: creates concrete dependencies internally
   pub struct App {
       camera: V4L2Camera,  // Can't substitute for testing
   }
   ```

2. **Return results, don't produce side effects**

   ```rust
   // Testable: returns what happened
   async fn detect(&self, image: &DynamicImage) -> Result<Vec<Detection>, DetectorError>

   // Hard to test: side effect only, nothing to assert on
   fn detect_and_log(&self, image: &DynamicImage)
   ```

3. **Use `async_trait` with `Send + Sync` bounds**

   ```rust
   #[async_trait]
   pub trait CameraCapture: Send + Sync {
       async fn capture_frame(&mut self) -> Result<DynamicImage, CameraError>;
   }
   ```
