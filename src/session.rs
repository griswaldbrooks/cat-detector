use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SessionError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// A single cat visit session, from entry to exit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatSession {
    pub id: String,
    pub entry_time: DateTime<Utc>,
    pub exit_time: Option<DateTime<Utc>>,
    pub sample_images: Vec<PathBuf>,
    pub entry_image: Option<PathBuf>,
    pub exit_image: Option<PathBuf>,
    pub video_path: Option<PathBuf>,
}

impl CatSession {
    pub fn new(entry_time: DateTime<Utc>) -> Self {
        let id = format!("session_{}", entry_time.format("%Y%m%d_%H%M%S"));
        Self {
            id,
            entry_time,
            exit_time: None,
            sample_images: Vec::new(),
            entry_image: None,
            exit_image: None,
            video_path: None,
        }
    }

    pub fn duration_secs(&self) -> Option<i64> {
        self.exit_time
            .map(|exit| (exit - self.entry_time).num_seconds())
    }

    pub fn is_active(&self) -> bool {
        self.exit_time.is_none()
    }

    pub fn finalize(&mut self, exit_time: DateTime<Utc>) {
        self.exit_time = Some(exit_time);
    }
}

/// Manages cat sessions — creates, updates, persists
pub struct SessionManager {
    sessions_dir: PathBuf,
    active_session: Option<CatSession>,
}

impl SessionManager {
    pub fn new(sessions_dir: PathBuf) -> Self {
        Self {
            sessions_dir,
            active_session: None,
        }
    }

    /// Start a new session when cat enters
    pub fn start_session(&mut self, entry_time: DateTime<Utc>) -> &CatSession {
        let session = CatSession::new(entry_time);
        self.active_session = Some(session);
        self.active_session.as_ref().unwrap()
    }

    /// Record the entry image path
    pub fn set_entry_image(&mut self, path: PathBuf) {
        if let Some(ref mut session) = self.active_session {
            session.entry_image = Some(path);
        }
    }

    /// Add a sample image to the active session
    pub fn add_sample_image(&mut self, path: PathBuf) {
        if let Some(ref mut session) = self.active_session {
            session.sample_images.push(path);
        }
    }

    /// Set the video path for the active session
    pub fn set_video_path(&mut self, path: PathBuf) {
        if let Some(ref mut session) = self.active_session {
            session.video_path = Some(path);
        }
    }

    /// Record the exit image path
    pub fn set_exit_image(&mut self, path: PathBuf) {
        if let Some(ref mut session) = self.active_session {
            session.exit_image = Some(path);
        }
    }

    /// End the active session and persist it
    pub fn end_session(
        &mut self,
        exit_time: DateTime<Utc>,
    ) -> Result<Option<CatSession>, SessionError> {
        let mut session = match self.active_session.take() {
            Some(s) => s,
            None => return Ok(None),
        };

        session.finalize(exit_time);
        self.persist_session(&session)?;
        Ok(Some(session))
    }

    /// Get the active session if any
    pub fn active_session(&self) -> Option<&CatSession> {
        self.active_session.as_ref()
    }

    /// Load all completed sessions from disk, sorted by entry_time descending
    pub fn load_sessions(&self) -> Result<Vec<CatSession>, SessionError> {
        let mut sessions = Vec::new();

        if !self.sessions_dir.exists() {
            return Ok(sessions);
        }

        for entry in std::fs::read_dir(&self.sessions_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                let data = std::fs::read_to_string(&path)?;
                match serde_json::from_str::<CatSession>(&data) {
                    Ok(session) => sessions.push(session),
                    Err(e) => {
                        tracing::warn!("Failed to parse session {:?}: {}", path, e);
                    }
                }
            }
        }

        sessions.sort_by(|a, b| b.entry_time.cmp(&a.entry_time));
        Ok(sessions)
    }

    /// Load a single session by ID
    pub fn load_session(&self, id: &str) -> Result<Option<CatSession>, SessionError> {
        let path = self.sessions_dir.join(format!("{}.json", id));
        if !path.exists() {
            return Ok(None);
        }
        let data = std::fs::read_to_string(&path)?;
        let session = serde_json::from_str(&data)?;
        Ok(Some(session))
    }

    fn persist_session(&self, session: &CatSession) -> Result<(), SessionError> {
        std::fs::create_dir_all(&self.sessions_dir)?;
        let path = self.sessions_dir.join(format!("{}.json", session.id));
        let data = serde_json::to_string_pretty(session)?;
        std::fs::write(&path, data)?;
        tracing::info!("Persisted session {:?}", path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_new_has_correct_fields() {
        let now = Utc::now();
        let session = CatSession::new(now);

        assert!(session.id.starts_with("session_"));
        assert_eq!(session.entry_time, now);
        assert!(session.exit_time.is_none());
        assert!(session.sample_images.is_empty());
        assert!(session.entry_image.is_none());
        assert!(session.exit_image.is_none());
        assert!(session.video_path.is_none());
        assert!(session.is_active());
        assert!(session.duration_secs().is_none());
    }

    #[test]
    fn test_session_finalize_sets_exit_and_duration() {
        let entry = Utc::now();
        let exit = entry + chrono::Duration::seconds(120);

        let mut session = CatSession::new(entry);
        session.finalize(exit);

        assert!(!session.is_active());
        assert_eq!(session.exit_time, Some(exit));
        assert_eq!(session.duration_secs(), Some(120));
    }

    #[test]
    fn test_session_manager_start_end_lifecycle() {
        let tmp = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(tmp.path().join("sessions"));

        let entry_time = Utc::now();
        let session = mgr.start_session(entry_time);
        assert!(session.is_active());

        // Set images
        mgr.set_entry_image(PathBuf::from("entry.jpg"));
        mgr.add_sample_image(PathBuf::from("sample1.jpg"));
        mgr.add_sample_image(PathBuf::from("sample2.jpg"));
        mgr.set_video_path(PathBuf::from("video.mp4"));

        let exit_time = entry_time + chrono::Duration::seconds(60);
        let completed = mgr.end_session(exit_time).unwrap().unwrap();

        assert!(!completed.is_active());
        assert_eq!(completed.duration_secs(), Some(60));
        assert_eq!(completed.entry_image, Some(PathBuf::from("entry.jpg")));
        assert_eq!(completed.sample_images.len(), 2);
        assert_eq!(completed.video_path, Some(PathBuf::from("video.mp4")));

        // Should be persisted to disk
        let sessions = mgr.load_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, completed.id);
    }

    #[test]
    fn test_session_manager_end_without_start_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(tmp.path().join("sessions"));

        let result = mgr.end_session(Utc::now()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_session_manager_load_by_id() {
        let tmp = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(tmp.path().join("sessions"));

        let entry_time = Utc::now();
        mgr.start_session(entry_time);
        let session_id = mgr.active_session().unwrap().id.clone();

        let exit_time = entry_time + chrono::Duration::seconds(30);
        mgr.end_session(exit_time).unwrap();

        let loaded = mgr.load_session(&session_id).unwrap().unwrap();
        assert_eq!(loaded.id, session_id);
        assert_eq!(loaded.duration_secs(), Some(30));
    }

    #[test]
    fn test_session_manager_load_nonexistent_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(tmp.path().join("sessions"));

        let result = mgr.load_session("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_sessions_sorted_newest_first() {
        let tmp = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(tmp.path().join("sessions"));

        let t1 = Utc::now() - chrono::Duration::hours(2);
        let t2 = Utc::now() - chrono::Duration::hours(1);
        let t3 = Utc::now();

        // Create sessions in chronological order
        mgr.start_session(t1);
        mgr.end_session(t1 + chrono::Duration::seconds(60)).unwrap();

        mgr.start_session(t2);
        mgr.end_session(t2 + chrono::Duration::seconds(60)).unwrap();

        mgr.start_session(t3);
        mgr.end_session(t3 + chrono::Duration::seconds(60)).unwrap();

        let sessions = mgr.load_sessions().unwrap();
        assert_eq!(sessions.len(), 3);
        // Newest first
        assert!(sessions[0].entry_time > sessions[1].entry_time);
        assert!(sessions[1].entry_time > sessions[2].entry_time);
    }
}
