//! # Cache Eviction Policies & Memory Pressure Management
//!
//! Provides intelligent sequence eviction strategies (LRU) when the physical
//! KV-cache block pool encounters high memory pressure.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Abstract interface for sequence cache eviction policies.
pub trait EvictionPolicy: Send + Sync {
    /// Records or refreshes access activity for an active sequence.
    fn record_access(&mut self, seq_id: u64);

    /// Unregisters a sequence when it finishes or is dropped.
    fn remove_sequence(&mut self, seq_id: u64);

    /// Selects the best candidate sequence to evict under memory pressure.
    fn select_eviction_candidate(&self) -> Option<u64>;

    /// Number of tracked active sequences.
    fn len(&self) -> usize;

    /// Whether any sequences are currently tracked.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Least-Recently-Used (LRU) eviction policy tracking access timestamps.
#[derive(Debug, Default)]
pub struct LruEvictionPolicy {
    queue: VecDeque<u64>,
    timestamps: HashMap<u64, Instant>,
}

impl LruEvictionPolicy {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            timestamps: HashMap::new(),
        }
    }
}

impl EvictionPolicy for LruEvictionPolicy {
    fn record_access(&mut self, seq_id: u64) {
        self.timestamps.insert(seq_id, Instant::now());
        self.queue.retain(|&id| id != seq_id);
        self.queue.push_back(seq_id);
    }

    fn remove_sequence(&mut self, seq_id: u64) {
        self.timestamps.remove(&seq_id);
        self.queue.retain(|&id| id != seq_id);
    }

    fn select_eviction_candidate(&self) -> Option<u64> {
        self.queue.front().copied()
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_eviction_policy() {
        let mut policy = LruEvictionPolicy::new();
        policy.record_access(101);
        policy.record_access(102);
        policy.record_access(103);

        // 101 was accessed first -> least recently used
        assert_eq!(policy.select_eviction_candidate(), Some(101));

        // Access 101 again -> 102 becomes least recently used
        policy.record_access(101);
        assert_eq!(policy.select_eviction_candidate(), Some(102));

        // Remove 102 -> 103 becomes candidate
        policy.remove_sequence(102);
        assert_eq!(policy.select_eviction_candidate(), Some(103));
    }
}
