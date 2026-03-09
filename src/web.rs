//! Web dashboard for cat detector
//!
//! Provides a web interface for monitoring cat detection status and live streaming.

#[cfg(feature = "web")]
use axum::{
    body::Body,
    extract::{Path as AxumPath, State},
    http::{header, Response, StatusCode},
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
#[cfg(feature = "web")]
use bytes::Bytes;
#[cfg(feature = "web")]
use std::convert::Infallible;
#[cfg(feature = "web")]
use tokio_stream::wrappers::WatchStream;
#[cfg(feature = "web")]
use tokio_stream::StreamExt;

use chrono::{DateTime, Utc};
use image::{DynamicImage, Rgb, RgbImage};
use serde::Serialize;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{watch, RwLock};

use crate::config::WebConfig;
use crate::detector::{BoundingBox, Detection};
#[cfg(feature = "web")]
use crate::session::SessionManager;
use crate::storage::ImageType;

/// Maximum number of recent captures to keep in memory
const MAX_RECENT_CAPTURES: usize = 50;

/// Shared application state between detection loop and web server
#[derive(Debug)]
pub struct WebAppState {
    /// Whether detection is currently running
    pub detecting: bool,
    /// Whether a cat is currently present
    pub cat_present: bool,
    /// Timestamp of last detection
    pub last_detection: Option<DateTime<Utc>>,
    /// When the app started
    pub started_at: DateTime<Utc>,
    /// Recent captures
    pub recent_captures: VecDeque<CaptureInfo>,
    /// Latest frame with bounding boxes (JPEG encoded)
    pub latest_frame: Option<Vec<u8>>,
    /// Latest detections for the current frame
    pub current_detections: Vec<Detection>,
    /// Directory where sessions are stored
    pub sessions_dir: PathBuf,
    /// Directory where captures (images/videos) are stored
    pub captures_dir: PathBuf,
    /// Static system info set at startup
    pub system_info: Option<SystemInfoResponse>,
}

impl Default for WebAppState {
    fn default() -> Self {
        Self {
            detecting: false,
            cat_present: false,
            last_detection: None,
            started_at: Utc::now(),
            recent_captures: VecDeque::new(),
            latest_frame: None,
            current_detections: Vec::new(),
            sessions_dir: PathBuf::from("captures/sessions"),
            captures_dir: PathBuf::from("captures"),
            system_info: None,
        }
    }
}

impl WebAppState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with explicit sessions and captures directories
    pub fn with_dirs(sessions_dir: PathBuf, captures_dir: PathBuf) -> Self {
        Self {
            sessions_dir,
            captures_dir,
            ..Self::default()
        }
    }

    /// Add a capture to the recent captures list
    pub fn add_capture(&mut self, capture: CaptureInfo) {
        self.recent_captures.push_front(capture);
        while self.recent_captures.len() > MAX_RECENT_CAPTURES {
            self.recent_captures.pop_back();
        }
    }

    /// Update the latest frame with bounding boxes drawn
    pub fn update_frame(&mut self, frame: &DynamicImage, detections: &[Detection]) {
        let annotated = draw_bounding_boxes(frame, detections);
        if let Ok(jpeg_data) = encode_jpeg(&annotated) {
            self.latest_frame = Some(jpeg_data);
        }
        self.current_detections = detections.to_vec();
    }
}

/// Information about a captured image
#[derive(Debug, Clone, Serialize)]
pub struct CaptureInfo {
    pub filename: String,
    pub timestamp: DateTime<Utc>,
    #[serde(rename = "type")]
    pub capture_type: String,
    pub path: PathBuf,
}

impl CaptureInfo {
    pub fn new(path: PathBuf, timestamp: DateTime<Utc>, image_type: ImageType) -> Self {
        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let capture_type = match image_type {
            ImageType::Entry => "entry",
            ImageType::Exit => "exit",
            ImageType::Sample => "sample",
        }
        .to_string();

        Self {
            filename,
            timestamp,
            capture_type,
            path,
        }
    }
}

/// Draw bounding boxes on an image
pub fn draw_bounding_boxes(image: &DynamicImage, detections: &[Detection]) -> DynamicImage {
    let mut rgb_image = image.to_rgb8();
    let (width, height) = rgb_image.dimensions();

    for detection in detections {
        draw_box(
            &mut rgb_image,
            &detection.bbox,
            width,
            height,
            detection.confidence,
        );
    }

    DynamicImage::ImageRgb8(rgb_image)
}

/// Draw a single bounding box with confidence label
fn draw_box(
    image: &mut RgbImage,
    bbox: &BoundingBox,
    img_width: u32,
    img_height: u32,
    confidence: f32,
) {
    let color = Rgb([0u8, 255u8, 0u8]); // Green
    let thickness = 2;

    let x1 = (bbox.x.max(0.0) as u32).min(img_width.saturating_sub(1));
    let y1 = (bbox.y.max(0.0) as u32).min(img_height.saturating_sub(1));
    let x2 = ((bbox.x + bbox.width) as u32).min(img_width.saturating_sub(1));
    let y2 = ((bbox.y + bbox.height) as u32).min(img_height.saturating_sub(1));

    // Draw horizontal lines (top and bottom)
    for t in 0..thickness {
        let y_top = (y1 + t).min(img_height - 1);
        let y_bottom = y2.saturating_sub(t).min(img_height - 1);
        for x in x1..=x2 {
            image.put_pixel(x, y_top, color);
            image.put_pixel(x, y_bottom, color);
        }
    }

    // Draw vertical lines (left and right)
    for t in 0..thickness {
        let x_left = (x1 + t).min(img_width - 1);
        let x_right = x2.saturating_sub(t).min(img_width - 1);
        for y in y1..=y2 {
            image.put_pixel(x_left, y, color);
            image.put_pixel(x_right, y, color);
        }
    }

    // Draw confidence label background (simple filled rectangle at top-left of box)
    let label_height = 16u32;
    let label_width = 60u32;
    let label_y_start = y1.saturating_sub(label_height);
    let label_y_end = y1;
    let label_x_end = (x1 + label_width).min(img_width - 1);

    for y in label_y_start..label_y_end {
        for x in x1..label_x_end {
            if y < img_height && x < img_width {
                image.put_pixel(x, y, color);
            }
        }
    }

    // Draw confidence text (simplified - just draw percentage as colored blocks)
    let conf_percent = (confidence * 100.0) as u32;
    let text_color = Rgb([0u8, 0u8, 0u8]); // Black text on green background

    // Simple digit rendering (3x5 pixel digits)
    let digit1 = conf_percent / 10;
    let digit2 = conf_percent % 10;

    draw_digit(
        image,
        digit1,
        x1 + 4,
        label_y_start + 5,
        text_color,
        img_width,
        img_height,
    );
    draw_digit(
        image,
        digit2,
        x1 + 12,
        label_y_start + 5,
        text_color,
        img_width,
        img_height,
    );
    // Draw '%' sign approximation
    if x1 + 22 < img_width && label_y_start + 5 < img_height {
        image.put_pixel(x1 + 20, label_y_start + 5, text_color);
        image.put_pixel(x1 + 22, label_y_start + 9, text_color);
    }
}

/// Draw a simple 3x5 pixel digit
fn draw_digit(
    image: &mut RgbImage,
    digit: u32,
    x: u32,
    y: u32,
    color: Rgb<u8>,
    max_w: u32,
    max_h: u32,
) {
    // Simple 3x5 digit patterns
    let patterns: [[u8; 15]; 10] = [
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], // 0
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1], // 1
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], // 2
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], // 3
        [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1], // 4
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1], // 5
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1], // 6
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], // 7
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], // 8
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], // 9
    ];

    let pattern = &patterns[digit as usize % 10];
    for row in 0..5 {
        for col in 0..3 {
            if pattern[row * 3 + col] == 1 {
                let px = x + col as u32;
                let py = y + row as u32;
                if px < max_w && py < max_h {
                    image.put_pixel(px, py, color);
                }
            }
        }
    }
}

/// Encode an image as JPEG
fn encode_jpeg(image: &DynamicImage) -> Result<Vec<u8>, image::ImageError> {
    let mut buffer = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buffer);
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, 80);
    image.to_rgb8().write_with_encoder(encoder)?;
    Ok(buffer)
}

/// API response for /api/system-info
#[derive(Debug, Clone, Serialize)]
pub struct SystemInfoResponse {
    pub version: String,
    pub model_name: String,
    pub model_format: String,
    pub confidence_threshold: f32,
    pub detection_interval_ms: u64,
    pub camera_resolution: String,
}

/// API response for /api/status
#[derive(Serialize)]
pub struct StatusResponse {
    pub detecting: bool,
    pub cat_present: bool,
    pub last_detection: Option<DateTime<Utc>>,
    pub uptime_secs: u64,
    pub detection_count: usize,
}

/// Shared state type for axum
#[cfg(feature = "web")]
pub type SharedWebState = Arc<RwLock<WebAppState>>;

/// Frame sender for MJPEG streaming
pub type FrameSender = watch::Sender<Option<Vec<u8>>>;
pub type FrameReceiver = watch::Receiver<Option<Vec<u8>>>;

/// Create a new frame channel for MJPEG streaming
pub fn create_frame_channel() -> (FrameSender, FrameReceiver) {
    watch::channel(None)
}

#[cfg(feature = "web")]
pub fn create_router(state: SharedWebState, frame_rx: FrameReceiver) -> Router {
    Router::new()
        .route("/", get(dashboard_handler))
        .route("/sessions", get(sessions_list_handler))
        .route("/sessions/:id", get(session_detail_handler))
        .route("/api/status", get(status_handler))
        .route("/api/system-info", get(system_info_handler))
        .route("/api/captures", get(captures_handler))
        .route("/api/sessions", get(sessions_api_handler))
        .route("/api/sessions/:id", get(session_detail_api_handler))
        .route("/api/frame", get(frame_handler))
        .route("/captures/:filename", get(serve_capture_file))
        .route("/api/stream", get(move || stream_handler(frame_rx.clone())))
        .with_state(state)
}

#[cfg(feature = "web")]
async fn dashboard_handler() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

#[cfg(feature = "web")]
async fn status_handler(State(state): State<SharedWebState>) -> Json<StatusResponse> {
    let state = state.read().await;
    let uptime = Utc::now()
        .signed_duration_since(state.started_at)
        .num_seconds()
        .max(0) as u64;

    Json(StatusResponse {
        detecting: state.detecting,
        cat_present: state.cat_present,
        last_detection: state.last_detection,
        uptime_secs: uptime,
        detection_count: state.current_detections.len(),
    })
}

#[cfg(feature = "web")]
async fn system_info_handler(State(state): State<SharedWebState>) -> impl IntoResponse {
    let state = state.read().await;
    match &state.system_info {
        Some(info) => Json(serde_json::json!({
            "model_name": info.model_name,
            "model_format": info.model_format,
            "confidence_threshold": info.confidence_threshold,
            "detection_interval_ms": info.detection_interval_ms,
            "camera_resolution": info.camera_resolution,
        }))
        .into_response(),
        None => Json(serde_json::json!({})).into_response(),
    }
}

#[cfg(feature = "web")]
async fn captures_handler(State(state): State<SharedWebState>) -> Json<Vec<CaptureInfo>> {
    let state = state.read().await;
    Json(state.recent_captures.iter().cloned().collect())
}

#[cfg(feature = "web")]
async fn frame_handler(State(state): State<SharedWebState>) -> impl IntoResponse {
    let state = state.read().await;
    match &state.latest_frame {
        Some(jpeg_data) => Response::builder()
            .header(header::CONTENT_TYPE, "image/jpeg")
            .body(Body::from(jpeg_data.clone()))
            .unwrap(),
        None => Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .body(Body::from("No frame available"))
            .unwrap(),
    }
}

#[cfg(feature = "web")]
async fn stream_handler(frame_rx: FrameReceiver) -> impl IntoResponse {
    let stream = WatchStream::new(frame_rx).filter_map(|frame: Option<Vec<u8>>| {
        frame.map(|data: Vec<u8>| {
            let boundary = "--frame\r\n";
            let content_type = "Content-Type: image/jpeg\r\n\r\n";
            let end = "\r\n";

            let mut response_bytes = Vec::new();
            response_bytes.extend_from_slice(boundary.as_bytes());
            response_bytes.extend_from_slice(content_type.as_bytes());
            response_bytes.extend_from_slice(&data);
            response_bytes.extend_from_slice(end.as_bytes());

            Ok::<_, Infallible>(Bytes::from(response_bytes))
        })
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "multipart/x-mixed-replace; boundary=frame",
        )
        .body(Body::from_stream(stream))
        .unwrap()
}

#[cfg(feature = "web")]
async fn sessions_api_handler(State(state): State<SharedWebState>) -> impl IntoResponse {
    let state = state.read().await;
    let mgr = SessionManager::new(state.sessions_dir.clone());
    match mgr.load_sessions() {
        Ok(sessions) => Json(sessions).into_response(),
        Err(e) => {
            tracing::warn!("Failed to load sessions: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to load sessions").into_response()
        }
    }
}

#[cfg(feature = "web")]
async fn session_detail_api_handler(
    State(state): State<SharedWebState>,
    AxumPath(id): AxumPath<String>,
) -> impl IntoResponse {
    let state = state.read().await;
    let mgr = SessionManager::new(state.sessions_dir.clone());
    match mgr.load_session(&id) {
        Ok(Some(session)) => Json(session).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, "Session not found").into_response(),
        Err(e) => {
            tracing::warn!("Failed to load session {}: {}", id, e);
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to load session").into_response()
        }
    }
}

#[cfg(feature = "web")]
async fn sessions_list_handler() -> Html<&'static str> {
    Html(SESSIONS_LIST_HTML)
}

#[cfg(feature = "web")]
async fn session_detail_handler() -> Html<&'static str> {
    Html(SESSION_DETAIL_HTML)
}

#[cfg(feature = "web")]
async fn serve_capture_file(
    State(state): State<SharedWebState>,
    AxumPath(filename): AxumPath<String>,
) -> impl IntoResponse {
    // Security: reject path traversal
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return (StatusCode::BAD_REQUEST, "Invalid filename").into_response();
    }

    let state = state.read().await;
    let file_path = state.captures_dir.join(&filename);

    match tokio::fs::read(&file_path).await {
        Ok(data) => {
            let content_type = match file_path.extension().and_then(|e| e.to_str()).unwrap_or("") {
                "jpg" | "jpeg" => "image/jpeg",
                "png" => "image/png",
                "mp4" => "video/mp4",
                "webm" => "video/webm",
                "avi" => "video/x-msvideo",
                _ => "application/octet-stream",
            };

            Response::builder()
                .header(header::CONTENT_TYPE, content_type)
                .body(Body::from(data))
                .unwrap()
                .into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "File not found").into_response(),
    }
}

#[cfg(feature = "web")]
pub async fn run_server(
    config: &WebConfig,
    state: SharedWebState,
    frame_rx: FrameReceiver,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = create_router(state, frame_rx);
    let addr = format!("{}:{}", config.bind_address, config.port);

    tracing::info!("Starting web server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown_rx.changed().await;
        })
        .await?;

    Ok(())
}

/// Embedded dashboard HTML
#[cfg(feature = "web")]
const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cat Detector Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
        }
        h1 { font-size: 24px; }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #666;
            transition: background 0.3s;
        }
        .status-dot.active { background: #4ade80; box-shadow: 0 0 10px #4ade80; }
        .status-dot.detecting { animation: pulse 1s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) {
            .main-content { grid-template-columns: 1fr; }
        }
        .stream-container {
            background: #16213e;
            border-radius: 12px;
            overflow: hidden;
        }
        .stream-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        .stream-placeholder {
            aspect-ratio: 4/3;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
        }
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .card {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
        }
        .card h2 {
            font-size: 14px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 15px;
        }
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4ade80;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
        }
        .captures-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .capture-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #333;
        }
        .capture-item:last-child { border-bottom: none; }
        .capture-type {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }
        .capture-type.entry { background: #4ade80; color: #000; }
        .capture-type.exit { background: #f87171; color: #000; }
        .capture-type.sample { background: #60a5fa; color: #000; }
        .capture-time {
            font-size: 12px;
            color: #888;
        }
        .last-detection {
            font-size: 18px;
            color: #4ade80;
        }
        .info-btn {
            background: none;
            border: 1px solid #555;
            color: #aaa;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .info-btn:hover { border-color: #888; color: #eee; }
        .system-info-panel {
            display: none;
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .system-info-panel.visible { display: block; }
        .system-info-panel h2 {
            font-size: 14px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 12px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 12px;
        }
        .info-grid .info-item {
            font-size: 13px;
        }
        .info-grid .info-label {
            color: #888;
            font-size: 11px;
            text-transform: uppercase;
            margin-bottom: 2px;
        }
        .info-grid .info-value {
            color: #eee;
            font-size: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Cat Detector <span id="versionTag" style="font-size:14px;color:#888;font-weight:normal;"></span></h1>
            <nav style="display:flex;align-items:center;gap:20px;">
                <a href="/sessions" style="color:#60a5fa;text-decoration:none;font-size:14px;">Sessions</a>
                <button class="info-btn" id="infoBtn" title="System Info">i</button>
                <div class="status-indicator">
                    <div class="status-dot" id="statusDot"></div>
                    <span id="statusText">Connecting...</span>
                </div>
            </nav>
        </header>

        <div class="system-info-panel" id="systemInfoPanel">
            <h2>System Info</h2>
            <div class="info-grid" id="systemInfoGrid">Loading...</div>
        </div>

        <div class="main-content">
            <div class="stream-container">
                <img id="stream" src="/api/stream" alt="Live Stream"
                     onerror="this.style.display='none'; document.getElementById('placeholder').style.display='flex';">
                <div id="placeholder" class="stream-placeholder" style="display:none;">
                    <span>Stream unavailable</span>
                </div>
            </div>

            <div class="sidebar">
                <div class="card">
                    <h2>Status</h2>
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-value" id="uptimeValue">0</div>
                            <div class="stat-label">Uptime (min)</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="detectionsValue">0</div>
                            <div class="stat-label">Detections</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <div class="stat-label">Last Detection</div>
                        <div class="last-detection" id="lastDetection">Never</div>
                    </div>
                </div>

                <div class="card">
                    <h2>Recent Captures</h2>
                    <div class="captures-list" id="capturesList">
                        <div style="color: #666; text-align: center; padding: 20px;">
                            No captures yet
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();

                const dot = document.getElementById('statusDot');
                const text = document.getElementById('statusText');

                if (data.cat_present) {
                    dot.className = 'status-dot active';
                    text.textContent = 'Cat Present!';
                } else if (data.detecting) {
                    dot.className = 'status-dot detecting';
                    text.textContent = 'Monitoring...';
                } else {
                    dot.className = 'status-dot';
                    text.textContent = 'Idle';
                }

                document.getElementById('uptimeValue').textContent =
                    Math.floor(data.uptime_secs / 60);
                document.getElementById('detectionsValue').textContent =
                    data.detection_count;

                if (data.last_detection) {
                    const date = new Date(data.last_detection);
                    document.getElementById('lastDetection').textContent =
                        date.toLocaleTimeString();
                }
            } catch (e) {
                console.error('Failed to fetch status:', e);
            }
        }

        async function updateCaptures() {
            try {
                const res = await fetch('/api/captures');
                const captures = await res.json();

                const list = document.getElementById('capturesList');
                if (captures.length === 0) {
                    list.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No captures yet</div>';
                    return;
                }

                list.innerHTML = captures.slice(0, 20).map(c => `
                    <div class="capture-item">
                        <div>
                            <span class="capture-type ${c.type}">${c.type}</span>
                            <span style="margin-left: 8px; font-size: 14px;">${c.filename}</span>
                        </div>
                        <span class="capture-time">${new Date(c.timestamp).toLocaleTimeString()}</span>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to fetch captures:', e);
            }
        }

        // System info panel
        var infoLoaded = false;
        document.getElementById('infoBtn').addEventListener('click', function() {
            var panel = document.getElementById('systemInfoPanel');
            panel.classList.toggle('visible');
            if (!infoLoaded) {
                infoLoaded = true;
                fetch('/api/system-info').then(function(r) { return r.json(); }).then(function(info) {
                    var grid = document.getElementById('systemInfoGrid');
                    var items = [
                        ['Version', info.version || 'Unknown'],
                        ['Model', info.model_name || 'Unknown'],
                        ['Format', info.model_format || 'Unknown'],
                        ['Threshold', info.confidence_threshold != null ? info.confidence_threshold.toFixed(2) : 'N/A'],
                        ['Detection Interval', (info.detection_interval_ms || 0) + 'ms'],
                        ['Camera', info.camera_resolution || 'Unknown'],
                    ];
                    grid.innerHTML = items.map(function(item) {
                        return '<div class="info-item"><div class="info-label">' + item[0] + '</div><div class="info-value">' + item[1] + '</div></div>';
                    }).join('');
                }).catch(function() {
                    document.getElementById('systemInfoGrid').textContent = 'Failed to load';
                    infoLoaded = false;
                });
            }
        });

        // Load version into header
        fetch('/api/system-info').then(function(r) { return r.json(); }).then(function(info) {
            if (info.version) document.getElementById('versionTag').textContent = 'v' + info.version;
        });

        // Initial load
        updateStatus();
        updateCaptures();

        // Periodic updates
        setInterval(updateStatus, 1000);
        setInterval(updateCaptures, 5000);
    </script>
</body>
</html>
"#;

/// Sessions list HTML page
#[cfg(feature = "web")]
const SESSIONS_LIST_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cat Sessions</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 10px;
        }
        h1 { font-size: 24px; }
        a { color: #60a5fa; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .sessions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }
        .session-card {
            background: #16213e;
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.15s;
            cursor: pointer;
        }
        .session-card:hover { transform: translateY(-2px); }
        .session-card a { color: inherit; text-decoration: none; display: block; }
        .session-thumb {
            width: 100%;
            aspect-ratio: 4/3;
            object-fit: cover;
            background: #0f3460;
            display: block;
        }
        .session-thumb-placeholder {
            width: 100%;
            aspect-ratio: 4/3;
            background: #0f3460;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 14px;
        }
        .session-info {
            padding: 16px;
        }
        .session-time {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .session-meta {
            font-size: 13px;
            color: #888;
            display: flex;
            gap: 16px;
        }
        .session-active {
            display: inline-block;
            background: #4ade80;
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        .empty-state p { font-size: 18px; }
        .lightbox-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }
        .lightbox-overlay.active { display: flex; }
        .lightbox-overlay img.lightbox-img {
            max-width: 95vw;
            max-height: 95vh;
            object-fit: contain;
            border-radius: 8px;
            cursor: default;
        }
        .lightbox-nav {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(255,255,255,0.15);
            border: none;
            color: #fff;
            font-size: 36px;
            padding: 12px 16px;
            cursor: pointer;
            border-radius: 8px;
            user-select: none;
        }
        .lightbox-nav:hover { background: rgba(255,255,255,0.3); }
        .lightbox-prev { left: 12px; }
        .lightbox-next { right: 12px; }
        .session-thumb { cursor: pointer; }
    </style>
</head>
<body>
    <div class="lightbox-overlay" id="lightbox">
        <button class="lightbox-nav lightbox-prev" id="lightboxPrev">&#8249;</button>
        <img class="lightbox-img" id="lightboxImg" src="" alt="Full size">
        <button class="lightbox-nav lightbox-next" id="lightboxNext">&#8250;</button>
    </div>
    <div class="container">
        <header>
            <h1>Cat Sessions</h1>
            <a href="/">Dashboard</a>
        </header>
        <div id="sessionsList" class="sessions-grid"></div>
    </div>
    <script>
        function formatDuration(secs) {
            if (secs == null) return 'Active';
            const h = Math.floor(secs / 3600);
            const m = Math.floor((secs % 3600) / 60);
            const s = secs % 60;
            if (h > 0) return h + 'h ' + m + 'm ' + s + 's';
            if (m > 0) return m + 'm ' + s + 's';
            return s + 's';
        }

        function thumbUrl(session) {
            if (session.entry_image) {
                var parts = session.entry_image.replace(/\\/g, '/').split('/');
                return '/captures/' + parts[parts.length - 1];
            }
            return null;
        }

        async function loadSessions() {
            try {
                const res = await fetch('/api/sessions');
                const sessions = await res.json();
                const container = document.getElementById('sessionsList');

                if (sessions.length === 0) {
                    container.innerHTML = '<div class="empty-state"><p>No cat sessions recorded yet</p></div>';
                    return;
                }

                container.innerHTML = sessions.map(function(s) {
                    var thumb = thumbUrl(s);
                    var thumbHtml = thumb
                        ? '<img class="session-thumb" src="' + thumb + '" alt="Entry" loading="lazy">'
                        : '<div class="session-thumb-placeholder">No image</div>';
                    var duration = s.exit_time ? formatDuration(Math.round((new Date(s.exit_time) - new Date(s.entry_time)) / 1000)) : null;
                    var activeTag = s.exit_time ? '' : '<span class="session-active">ACTIVE</span>';
                    var durationText = duration ? duration : 'In progress';
                    var images = (s.sample_images ? s.sample_images.length : 0);
                    var entryDate = new Date(s.entry_time);
                    return '<div class="session-card"><a href="/sessions/' + s.id + '">'
                        + thumbHtml
                        + '<div class="session-info">'
                        + '<div class="session-time">' + entryDate.toLocaleString() + activeTag + '</div>'
                        + '<div class="session-meta"><span>Duration: ' + durationText + '</span><span>Images: ' + images + '</span></div>'
                        + '</div></a></div>';
                }).join('');
            } catch (e) {
                console.error('Failed to load sessions:', e);
                document.getElementById('sessionsList').innerHTML =
                    '<div class="empty-state"><p>Failed to load sessions</p></div>';
            }
        }

        loadSessions().then(function() {
            var overlay = document.getElementById('lightbox');
            var lbImg = document.getElementById('lightboxImg');
            var allImages = [];
            var currentIndex = 0;

            document.querySelectorAll('.session-thumb').forEach(function(img) {
                var idx = allImages.length;
                allImages.push(img.src);
                img.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    currentIndex = idx;
                    lbImg.src = allImages[currentIndex];
                    overlay.classList.add('active');
                });
            });

            function navigate(delta) {
                if (allImages.length === 0) return;
                currentIndex = (currentIndex + delta + allImages.length) % allImages.length;
                lbImg.src = allImages[currentIndex];
            }

            overlay.addEventListener('click', function(e) {
                if (e.target === overlay) overlay.classList.remove('active');
            });
            document.getElementById('lightboxPrev').addEventListener('click', function(e) {
                e.stopPropagation(); navigate(-1);
            });
            document.getElementById('lightboxNext').addEventListener('click', function(e) {
                e.stopPropagation(); navigate(1);
            });
            document.addEventListener('keydown', function(e) {
                if (!overlay.classList.contains('active')) return;
                if (e.key === 'Escape') overlay.classList.remove('active');
                if (e.key === 'ArrowLeft') navigate(-1);
                if (e.key === 'ArrowRight') navigate(1);
            });
        });
    </script>
</body>
</html>
"#;

/// Session detail HTML page
#[cfg(feature = "web")]
const SESSION_DETAIL_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Session Detail</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 10px;
        }
        h1 { font-size: 24px; }
        h2 { font-size: 18px; margin-bottom: 12px; color: #ccc; }
        a { color: #60a5fa; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .nav-links { display: flex; gap: 16px; }
        .info-card {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
        }
        .info-item label {
            display: block;
            font-size: 12px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 4px;
        }
        .info-item .value {
            font-size: 16px;
            font-weight: 500;
        }
        .active-badge {
            display: inline-block;
            background: #4ade80;
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .images-section {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .key-images {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }
        @media (max-width: 600px) {
            .key-images { grid-template-columns: 1fr; }
        }
        .key-image-container {
            text-align: center;
        }
        .key-image-container .label {
            font-size: 13px;
            color: #888;
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        .key-image-container img {
            width: 100%;
            border-radius: 8px;
            display: block;
        }
        .key-image-container .no-image {
            width: 100%;
            aspect-ratio: 4/3;
            background: #0f3460;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 14px;
        }
        .gallery {
            display: flex;
            gap: 12px;
            overflow-x: auto;
            padding-bottom: 8px;
            -webkit-overflow-scrolling: touch;
        }
        @media (min-width: 900px) {
            .gallery {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                overflow-x: visible;
            }
        }
        .gallery img {
            min-width: 200px;
            max-width: 300px;
            width: 100%;
            border-radius: 8px;
            object-fit: cover;
            aspect-ratio: 4/3;
        }
        .video-section {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .video-section video {
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #666;
        }
        #loading {
            text-align: center;
            padding: 60px 20px;
            color: #888;
        }
        .lightbox-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }
        .lightbox-overlay.active { display: flex; }
        .lightbox-overlay img.lightbox-img {
            max-width: 95vw;
            max-height: 95vh;
            object-fit: contain;
            border-radius: 8px;
            cursor: default;
        }
        .lightbox-nav {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(255,255,255,0.15);
            border: none;
            color: #fff;
            font-size: 36px;
            padding: 12px 16px;
            cursor: pointer;
            border-radius: 8px;
            user-select: none;
        }
        .lightbox-nav:hover { background: rgba(255,255,255,0.3); }
        .lightbox-prev { left: 12px; }
        .lightbox-next { right: 12px; }
        .gallery img, .key-image-container img { cursor: pointer; }
    </style>
</head>
<body>
    <div class="lightbox-overlay" id="lightbox">
        <button class="lightbox-nav lightbox-prev" id="lightboxPrev">&#8249;</button>
        <img class="lightbox-img" id="lightboxImg" src="" alt="Full size">
        <button class="lightbox-nav lightbox-next" id="lightboxNext">&#8250;</button>
    </div>
    <div class="container">
        <header>
            <h1 id="pageTitle">Session Detail</h1>
            <div class="nav-links">
                <a href="/sessions">All Sessions</a>
                <a href="/">Dashboard</a>
            </div>
        </header>
        <div id="loading">Loading session...</div>
        <div id="content" style="display:none;"></div>
    </div>
    <script>
        function formatDuration(secs) {
            if (secs == null) return 'In progress';
            const h = Math.floor(secs / 3600);
            const m = Math.floor((secs % 3600) / 60);
            const s = secs % 60;
            if (h > 0) return h + 'h ' + m + 'm ' + s + 's';
            if (m > 0) return m + 'm ' + s + 's';
            return s + 's';
        }

        function fileUrl(path) {
            if (!path) return null;
            var parts = path.replace(/\\/g, '/').split('/');
            return '/captures/' + parts[parts.length - 1];
        }

        async function loadSession() {
            var pathParts = window.location.pathname.split('/');
            var sessionId = pathParts[pathParts.length - 1];

            try {
                var res = await fetch('/api/sessions/' + sessionId);
                if (res.status === 404) {
                    document.getElementById('loading').innerHTML = '<p>Session not found</p>';
                    return;
                }
                var s = await res.json();

                document.getElementById('pageTitle').textContent = 'Session: ' + new Date(s.entry_time).toLocaleString();
                document.getElementById('loading').style.display = 'none';
                var content = document.getElementById('content');
                content.style.display = 'block';

                var durationSecs = s.exit_time ? Math.round((new Date(s.exit_time) - new Date(s.entry_time)) / 1000) : null;
                var statusHtml = s.exit_time
                    ? '<span class="value">' + new Date(s.exit_time).toLocaleString() + '</span>'
                    : '<span class="active-badge">ACTIVE</span>';

                var html = '<div class="info-card"><div class="info-grid">'
                    + '<div class="info-item"><label>Entry Time</label><span class="value">' + new Date(s.entry_time).toLocaleString() + '</span></div>'
                    + '<div class="info-item"><label>Exit Time</label>' + statusHtml + '</div>'
                    + '<div class="info-item"><label>Duration</label><span class="value">' + formatDuration(durationSecs) + '</span></div>'
                    + '<div class="info-item"><label>Sample Images</label><span class="value">' + (s.sample_images ? s.sample_images.length : 0) + '</span></div>'
                    + '</div></div>';

                // Entry/exit images
                var entryUrl = fileUrl(s.entry_image);
                var exitUrl = fileUrl(s.exit_image);
                if (entryUrl || exitUrl) {
                    html += '<div class="images-section"><h2>Entry / Exit</h2><div class="key-images">';
                    html += '<div class="key-image-container"><div class="label">Entry</div>';
                    html += entryUrl ? '<img src="' + entryUrl + '" alt="Entry" loading="lazy">' : '<div class="no-image">No entry image</div>';
                    html += '</div>';
                    html += '<div class="key-image-container"><div class="label">Exit</div>';
                    html += exitUrl ? '<img src="' + exitUrl + '" alt="Exit" loading="lazy">' : '<div class="no-image">No exit image</div>';
                    html += '</div>';
                    html += '</div></div>';
                }

                // Sample images gallery
                if (s.sample_images && s.sample_images.length > 0) {
                    html += '<div class="images-section"><h2>Sample Images (' + s.sample_images.length + ')</h2><div class="gallery">';
                    s.sample_images.forEach(function(img) {
                        var url = fileUrl(img);
                        if (url) {
                            html += '<img src="' + url + '" alt="Sample" loading="lazy">';
                        }
                    });
                    html += '</div></div>';
                }

                // Video
                var videoUrl = fileUrl(s.video_path);
                if (videoUrl) {
                    html += '<div class="video-section"><h2>Video</h2>';
                    html += '<video controls preload="metadata"><source src="' + videoUrl + '" type="video/mp4">Your browser does not support video playback.</video>';
                    html += '</div>';
                }

                content.innerHTML = html;
                initLightbox();
            } catch (e) {
                console.error('Failed to load session:', e);
                document.getElementById('loading').innerHTML = '<p>Failed to load session</p>';
            }
        }

        function initLightbox() {
            var overlay = document.getElementById('lightbox');
            var lbImg = document.getElementById('lightboxImg');
            var allImages = [];
            var currentIndex = 0;

            document.querySelectorAll('.gallery img, .key-image-container img').forEach(function(img) {
                var idx = allImages.length;
                allImages.push(img.src);
                img.addEventListener('click', function() {
                    currentIndex = idx;
                    lbImg.src = allImages[currentIndex];
                    overlay.classList.add('active');
                });
            });

            function navigate(delta) {
                if (allImages.length === 0) return;
                currentIndex = (currentIndex + delta + allImages.length) % allImages.length;
                lbImg.src = allImages[currentIndex];
            }

            overlay.addEventListener('click', function(e) {
                if (e.target === overlay) overlay.classList.remove('active');
            });
            document.getElementById('lightboxPrev').addEventListener('click', function(e) {
                e.stopPropagation(); navigate(-1);
            });
            document.getElementById('lightboxNext').addEventListener('click', function(e) {
                e.stopPropagation(); navigate(1);
            });
            document.addEventListener('keydown', function(e) {
                if (!overlay.classList.contains('active')) return;
                if (e.key === 'Escape') overlay.classList.remove('active');
                if (e.key === 'ArrowLeft') navigate(-1);
                if (e.key === 'ArrowRight') navigate(1);
            });
        }

        loadSession();
    </script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_app_state_default() {
        let state = WebAppState::default();
        assert!(!state.detecting);
        assert!(!state.cat_present);
        assert!(state.last_detection.is_none());
        assert!(state.recent_captures.is_empty());
        assert!(state.latest_frame.is_none());
    }

    #[test]
    fn test_add_capture_limits_size() {
        let mut state = WebAppState::new();

        for i in 0..60 {
            state.add_capture(CaptureInfo {
                filename: format!("test_{}.jpg", i),
                timestamp: Utc::now(),
                capture_type: "entry".to_string(),
                path: PathBuf::from(format!("/tmp/test_{}.jpg", i)),
            });
        }

        assert_eq!(state.recent_captures.len(), MAX_RECENT_CAPTURES);
    }

    #[test]
    fn test_capture_info_from_image_type() {
        let path = PathBuf::from("/captures/cat_entry_20240115_100000.jpg");
        let timestamp = Utc::now();

        let entry = CaptureInfo::new(path.clone(), timestamp, ImageType::Entry);
        assert_eq!(entry.capture_type, "entry");

        let exit = CaptureInfo::new(path.clone(), timestamp, ImageType::Exit);
        assert_eq!(exit.capture_type, "exit");

        let sample = CaptureInfo::new(path, timestamp, ImageType::Sample);
        assert_eq!(sample.capture_type, "sample");
    }

    #[test]
    fn test_draw_bounding_boxes_returns_image() {
        let image = DynamicImage::new_rgb8(640, 480);
        let detections = vec![Detection {
            class_id: 15,
            confidence: 0.9,
            bbox: BoundingBox {
                x: 100.0,
                y: 100.0,
                width: 200.0,
                height: 200.0,
            },
        }];

        let result = draw_bounding_boxes(&image, &detections);
        assert_eq!(result.width(), 640);
        assert_eq!(result.height(), 480);
    }

    #[test]
    fn test_encode_jpeg_produces_valid_data() {
        let image = DynamicImage::new_rgb8(100, 100);
        let result = encode_jpeg(&image);
        assert!(result.is_ok());
        let data = result.unwrap();
        // JPEG magic bytes
        assert!(data.len() > 2);
        assert_eq!(data[0], 0xFF);
        assert_eq!(data[1], 0xD8);
    }

    #[test]
    fn test_update_frame_stores_jpeg() {
        let mut state = WebAppState::new();
        let image = DynamicImage::new_rgb8(100, 100);
        let detections = vec![];

        state.update_frame(&image, &detections);
        assert!(state.latest_frame.is_some());

        let frame = state.latest_frame.unwrap();
        assert_eq!(frame[0], 0xFF);
        assert_eq!(frame[1], 0xD8);
    }

    #[test]
    fn test_status_response_serializes() {
        let response = StatusResponse {
            detecting: true,
            cat_present: false,
            last_detection: Some(Utc::now()),
            uptime_secs: 3600,
            detection_count: 5,
        };

        let json = serde_json::to_string(&response);
        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("\"detecting\":true"));
        assert!(json_str.contains("\"cat_present\":false"));
    }

    #[test]
    fn test_system_info_response_serializes() {
        let response = SystemInfoResponse {
            version: "1.0.0".to_string(),
            model_name: "clip_vitb32_image.onnx".to_string(),
            model_format: "clip".to_string(),
            confidence_threshold: 0.5,
            detection_interval_ms: 500,
            camera_resolution: "640x480".to_string(),
        };

        let json_str = serde_json::to_string(&response).unwrap();
        assert!(json_str.contains("\"model_name\":\"clip_vitb32_image.onnx\""));
        assert!(json_str.contains("\"detection_interval_ms\":500"));
        assert!(json_str.contains("\"camera_resolution\":\"640x480\""));
    }

    #[cfg(feature = "web")]
    #[test]
    fn test_dashboard_html_is_valid() {
        assert!(DASHBOARD_HTML.contains("<!DOCTYPE html>"));
        assert!(DASHBOARD_HTML.contains("/api/status"));
        assert!(DASHBOARD_HTML.contains("/api/captures"));
        assert!(DASHBOARD_HTML.contains("/api/stream"));
    }

    #[cfg(feature = "web")]
    #[test]
    fn test_dashboard_has_system_info_panel() {
        assert!(DASHBOARD_HTML.contains("/api/system-info"));
        assert!(DASHBOARD_HTML.contains("system-info-panel"));
    }

    #[cfg(feature = "web")]
    #[test]
    fn test_session_detail_has_lightbox_modal() {
        assert!(SESSION_DETAIL_HTML.contains("lightbox-overlay"));
        assert!(SESSION_DETAIL_HTML.contains("lightbox-img"));
    }

    #[cfg(feature = "web")]
    #[test]
    fn test_sessions_list_has_lightbox() {
        assert!(SESSIONS_LIST_HTML.contains("lightbox-overlay"));
        assert!(SESSIONS_LIST_HTML.contains("lightbox-img"));
        assert!(SESSIONS_LIST_HTML.contains("ArrowLeft"));
        assert!(SESSIONS_LIST_HTML.contains("ArrowRight"));
        assert!(SESSIONS_LIST_HTML.contains("Escape"));
    }

    #[cfg(feature = "web")]
    #[test]
    fn test_session_detail_lightbox_has_keyboard_nav() {
        assert!(SESSION_DETAIL_HTML.contains("ArrowLeft"));
        assert!(SESSION_DETAIL_HTML.contains("ArrowRight"));
        assert!(SESSION_DETAIL_HTML.contains("Escape"));
    }

    #[cfg(feature = "web")]
    #[tokio::test]
    async fn test_status_endpoint_returns_json() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let state = Arc::new(RwLock::new(WebAppState::new()));
        let (_tx, rx) = create_frame_channel();
        let app = create_router(state, rx);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response.headers().get("content-type").unwrap();
        assert!(content_type.to_str().unwrap().contains("application/json"));
    }

    #[cfg(feature = "web")]
    #[tokio::test]
    async fn test_system_info_endpoint_returns_json() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let mut web_state = WebAppState::new();
        web_state.system_info = Some(SystemInfoResponse {
            version: "1.0.0".to_string(),
            model_name: "clip_vitb32_image.onnx".to_string(),
            model_format: "clip".to_string(),
            confidence_threshold: 0.5,
            detection_interval_ms: 500,
            camera_resolution: "640x480".to_string(),
        });
        let state = Arc::new(RwLock::new(web_state));
        let (_tx, rx) = create_frame_channel();
        let app = create_router(state, rx);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/system-info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(json_str.contains("clip_vitb32_image.onnx"));
        assert!(json_str.contains("\"detection_interval_ms\":500"));
    }

    #[cfg(feature = "web")]
    #[tokio::test]
    async fn test_captures_endpoint_returns_array() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let state = Arc::new(RwLock::new(WebAppState::new()));
        let (_tx, rx) = create_frame_channel();
        let app = create_router(state, rx);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/captures")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.starts_with('['));
        assert!(body_str.ends_with(']'));
    }

    #[cfg(feature = "web")]
    #[tokio::test]
    async fn test_dashboard_returns_html() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let state = Arc::new(RwLock::new(WebAppState::new()));
        let (_tx, rx) = create_frame_channel();
        let app = create_router(state, rx);

        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response.headers().get("content-type").unwrap();
        assert!(content_type.to_str().unwrap().contains("text/html"));
    }

    #[cfg(feature = "web")]
    #[tokio::test]
    async fn test_stream_returns_mjpeg_content_type() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let state = Arc::new(RwLock::new(WebAppState::new()));
        let (_tx, rx) = create_frame_channel();
        let app = create_router(state, rx);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response.headers().get("content-type").unwrap();
        assert!(content_type
            .to_str()
            .unwrap()
            .contains("multipart/x-mixed-replace"));
    }
}
