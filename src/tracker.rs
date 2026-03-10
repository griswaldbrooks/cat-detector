use chrono::{DateTime, Duration, Utc};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CatState {
    Absent,
    Present,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackerEvent {
    CatEntered {
        timestamp: DateTime<Utc>,
    },
    CatExited {
        timestamp: DateTime<Utc>,
        entry_time: DateTime<Utc>,
    },
    SampleDue {
        timestamp: DateTime<Utc>,
    },
}

#[derive(Debug)]
pub struct CatTracker {
    state: CatState,
    consecutive_detections: u32,
    consecutive_non_detections: u32,
    enter_threshold: u32,
    exit_threshold: u32,
    sample_interval: Duration,
    entry_time: Option<DateTime<Utc>>,
    last_sample_time: Option<DateTime<Utc>>,
}

impl CatTracker {
    pub fn new(enter_threshold: u32, exit_threshold: u32, sample_interval_secs: u64) -> Self {
        Self {
            state: CatState::Absent,
            consecutive_detections: 0,
            consecutive_non_detections: 0,
            enter_threshold,
            exit_threshold,
            sample_interval: Duration::seconds(sample_interval_secs as i64),
            entry_time: None,
            last_sample_time: None,
        }
    }

    #[allow(dead_code)]
    pub fn current_state(&self) -> CatState {
        self.state
    }

    #[allow(dead_code)]
    pub fn entry_time(&self) -> Option<DateTime<Utc>> {
        self.entry_time
    }

    pub fn process_detection(
        &mut self,
        cat_detected: bool,
        timestamp: DateTime<Utc>,
    ) -> Vec<TrackerEvent> {
        match self.state {
            CatState::Absent => self.process_absent(cat_detected, timestamp),
            CatState::Present => self.process_present(cat_detected, timestamp),
        }
    }

    fn process_absent(
        &mut self,
        cat_detected: bool,
        timestamp: DateTime<Utc>,
    ) -> Vec<TrackerEvent> {
        if !cat_detected {
            self.consecutive_detections = 0;
            return Vec::new();
        }

        self.consecutive_detections += 1;
        self.consecutive_non_detections = 0;

        if self.consecutive_detections < self.enter_threshold {
            return Vec::new();
        }

        self.state = CatState::Present;
        self.entry_time = Some(timestamp);
        self.last_sample_time = Some(timestamp);
        vec![TrackerEvent::CatEntered { timestamp }]
    }

    fn process_present(
        &mut self,
        cat_detected: bool,
        timestamp: DateTime<Utc>,
    ) -> Vec<TrackerEvent> {
        if cat_detected {
            self.consecutive_non_detections = 0;
            return self.check_sample_due(timestamp);
        }

        self.consecutive_non_detections += 1;

        if self.consecutive_non_detections < self.exit_threshold {
            return Vec::new();
        }

        self.state = CatState::Absent;
        let entry_time = self.entry_time.take().unwrap();
        self.last_sample_time = None;
        self.consecutive_detections = 0;
        vec![TrackerEvent::CatExited {
            timestamp,
            entry_time,
        }]
    }

    fn check_sample_due(&mut self, timestamp: DateTime<Utc>) -> Vec<TrackerEvent> {
        let Some(last_sample) = self.last_sample_time else {
            return Vec::new();
        };

        if timestamp - last_sample < self.sample_interval {
            return Vec::new();
        }

        self.last_sample_time = Some(timestamp);
        vec![TrackerEvent::SampleDue { timestamp }]
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.state = CatState::Absent;
        self.consecutive_detections = 0;
        self.consecutive_non_detections = 0;
        self.entry_time = None;
        self.last_sample_time = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timestamp(offset_secs: i64) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2024-01-15T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
            + Duration::seconds(offset_secs)
    }

    #[test]
    fn test_initial_state_is_absent() {
        let tracker = CatTracker::new(3, 5, 10);
        assert_eq!(tracker.current_state(), CatState::Absent);
        assert!(tracker.entry_time().is_none());
    }

    #[test]
    fn test_cat_enters_after_threshold() {
        let mut tracker = CatTracker::new(3, 5, 10);
        let t0 = make_timestamp(0);

        // First two detections: no entry yet
        let events = tracker.process_detection(true, t0);
        assert!(events.is_empty());
        assert_eq!(tracker.current_state(), CatState::Absent);

        let events = tracker.process_detection(true, make_timestamp(1));
        assert!(events.is_empty());
        assert_eq!(tracker.current_state(), CatState::Absent);

        // Third detection: cat enters
        let events = tracker.process_detection(true, make_timestamp(2));
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], TrackerEvent::CatEntered { .. }));
        assert_eq!(tracker.current_state(), CatState::Present);
    }

    #[test]
    fn test_cat_exits_after_threshold() {
        let mut tracker = CatTracker::new(3, 5, 10);

        // Get cat into present state
        for i in 0..3 {
            tracker.process_detection(true, make_timestamp(i));
        }
        assert_eq!(tracker.current_state(), CatState::Present);

        // First four non-detections: still present
        for i in 3..7 {
            let events = tracker.process_detection(false, make_timestamp(i));
            if i < 7 {
                assert!(events.is_empty());
            }
        }

        // Fifth non-detection: cat exits
        let events = tracker.process_detection(false, make_timestamp(8));
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], TrackerEvent::CatExited { .. }));
        assert_eq!(tracker.current_state(), CatState::Absent);
    }

    #[test]
    fn test_hysteresis_gaps_dont_trigger_exit() {
        let mut tracker = CatTracker::new(3, 5, 10);

        // Get cat into present state
        for i in 0..3 {
            tracker.process_detection(true, make_timestamp(i));
        }
        assert_eq!(tracker.current_state(), CatState::Present);

        // Brief gap (4 non-detections, less than threshold of 5)
        for i in 3..7 {
            tracker.process_detection(false, make_timestamp(i));
        }
        assert_eq!(tracker.current_state(), CatState::Present);

        // Cat detected again - resets non-detection counter
        let events = tracker.process_detection(true, make_timestamp(7));
        assert!(events.is_empty()); // Just a normal detection, no event
        assert_eq!(tracker.current_state(), CatState::Present);

        // Another brief gap
        for i in 8..11 {
            tracker.process_detection(false, make_timestamp(i));
        }
        assert_eq!(tracker.current_state(), CatState::Present);
    }

    #[test]
    fn test_hysteresis_brief_detections_dont_trigger_entry() {
        let mut tracker = CatTracker::new(3, 5, 10);

        // Brief detection (2 times, less than threshold of 3)
        tracker.process_detection(true, make_timestamp(0));
        tracker.process_detection(true, make_timestamp(1));
        assert_eq!(tracker.current_state(), CatState::Absent);

        // Non-detection resets counter
        tracker.process_detection(false, make_timestamp(2));
        assert_eq!(tracker.current_state(), CatState::Absent);

        // Another brief detection
        tracker.process_detection(true, make_timestamp(3));
        tracker.process_detection(true, make_timestamp(4));
        assert_eq!(tracker.current_state(), CatState::Absent);
    }

    #[test]
    fn test_sample_interval_timing() {
        let mut tracker = CatTracker::new(1, 5, 10); // 1 detection to enter, 10 sec sample

        // Cat enters immediately (threshold=1)
        let events = tracker.process_detection(true, make_timestamp(0));
        assert!(matches!(events[0], TrackerEvent::CatEntered { .. }));

        // 5 seconds later: no sample yet
        let events = tracker.process_detection(true, make_timestamp(5));
        assert!(events.is_empty());

        // 10 seconds later: sample due
        let events = tracker.process_detection(true, make_timestamp(10));
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], TrackerEvent::SampleDue { .. }));

        // 15 seconds later: no sample yet (only 5 since last)
        let events = tracker.process_detection(true, make_timestamp(15));
        assert!(events.is_empty());

        // 20 seconds later: sample due again
        let events = tracker.process_detection(true, make_timestamp(20));
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], TrackerEvent::SampleDue { .. }));
    }

    #[test]
    fn test_exit_event_includes_entry_time() {
        let mut tracker = CatTracker::new(1, 1, 10);
        let entry_ts = make_timestamp(0);
        let exit_ts = make_timestamp(100);

        tracker.process_detection(true, entry_ts);
        let events = tracker.process_detection(false, exit_ts);

        if let TrackerEvent::CatExited {
            timestamp,
            entry_time,
        } = &events[0]
        {
            assert_eq!(*timestamp, exit_ts);
            assert_eq!(*entry_time, entry_ts);
        } else {
            panic!("Expected CatExited event");
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let mut tracker = CatTracker::new(1, 5, 10);

        // Get cat into present state
        tracker.process_detection(true, make_timestamp(0));
        assert_eq!(tracker.current_state(), CatState::Present);

        // Reset
        tracker.reset();
        assert_eq!(tracker.current_state(), CatState::Absent);
        assert!(tracker.entry_time().is_none());
    }

    #[test]
    fn test_multiple_enter_exit_cycles() {
        let mut tracker = CatTracker::new(2, 2, 10);
        let mut all_events = Vec::new();

        // First cycle: enter
        tracker.process_detection(true, make_timestamp(0));
        all_events.extend(tracker.process_detection(true, make_timestamp(1)));
        assert_eq!(tracker.current_state(), CatState::Present);

        // First cycle: exit
        tracker.process_detection(false, make_timestamp(10));
        all_events.extend(tracker.process_detection(false, make_timestamp(11)));
        assert_eq!(tracker.current_state(), CatState::Absent);

        // Second cycle: enter
        tracker.process_detection(true, make_timestamp(20));
        all_events.extend(tracker.process_detection(true, make_timestamp(21)));
        assert_eq!(tracker.current_state(), CatState::Present);

        // Verify we got 2 enter events and 1 exit event
        let enters: Vec<_> = all_events
            .iter()
            .filter(|e| matches!(e, TrackerEvent::CatEntered { .. }))
            .collect();
        let exits: Vec<_> = all_events
            .iter()
            .filter(|e| matches!(e, TrackerEvent::CatExited { .. }))
            .collect();

        assert_eq!(enters.len(), 2);
        assert_eq!(exits.len(), 1);
    }

    #[test]
    fn test_non_detection_resets_detection_counter_when_absent() {
        let mut tracker = CatTracker::new(3, 5, 10);

        // Two detections
        tracker.process_detection(true, make_timestamp(0));
        tracker.process_detection(true, make_timestamp(1));

        // One non-detection resets counter
        tracker.process_detection(false, make_timestamp(2));

        // Now need 3 more detections to enter
        tracker.process_detection(true, make_timestamp(3));
        tracker.process_detection(true, make_timestamp(4));
        assert_eq!(tracker.current_state(), CatState::Absent);

        tracker.process_detection(true, make_timestamp(5));
        assert_eq!(tracker.current_state(), CatState::Present);
    }

    #[test]
    fn test_detection_resets_non_detection_counter_when_present() {
        let mut tracker = CatTracker::new(1, 5, 10);

        // Enter
        tracker.process_detection(true, make_timestamp(0));
        assert_eq!(tracker.current_state(), CatState::Present);

        // 4 non-detections (not enough to exit)
        for i in 1..5 {
            tracker.process_detection(false, make_timestamp(i));
        }
        assert_eq!(tracker.current_state(), CatState::Present);

        // One detection resets counter
        tracker.process_detection(true, make_timestamp(5));

        // 4 more non-detections (still not enough)
        for i in 6..10 {
            tracker.process_detection(false, make_timestamp(i));
        }
        assert_eq!(tracker.current_state(), CatState::Present);
    }
}
