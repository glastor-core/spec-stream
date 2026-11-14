#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Real-time speculative decoding metrics and telemetry for enterprise monitoring.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpeculativeMetrics {
    /// Total speculative iterations/cycles executed.
    pub total_cycles: usize,
    /// Total number of candidate tokens proposed by the draft model.
    pub draft_tokens_proposed: usize,
    /// Total number of draft candidate tokens accepted by the target model.
    pub draft_tokens_accepted: usize,
    /// Total tokens produced and committed to the output stream.
    pub total_tokens_committed: usize,
    /// Number of times the entire draft batch was accepted completely (bonus token achieved).
    pub perfect_draft_cycles: usize,
}

impl SpeculativeMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a completed verification outcome.
    pub fn record_cycle(
        &mut self,
        proposed_count: usize,
        accepted_count: usize,
        total_committed: usize,
    ) {
        self.total_cycles += 1;
        self.draft_tokens_proposed += proposed_count;
        self.draft_tokens_accepted += accepted_count;
        self.total_tokens_committed += total_committed;

        if accepted_count == proposed_count && proposed_count > 0 {
            self.perfect_draft_cycles += 1;
        }
    }

    /// Acceptance rate (\alpha): Fraction of proposed draft tokens that were accepted.
    pub fn acceptance_rate(&self) -> f32 {
        if self.draft_tokens_proposed == 0 {
            return 0.0;
        }
        self.draft_tokens_accepted as f32 / self.draft_tokens_proposed as f32
    }

    /// Effective acceleration factor (Speedup):
    /// Average number of tokens generated per target model forward step.
    /// Baseline without speculative decoding is strictly 1.0 token / step.
    pub fn effective_speedup(&self) -> f32 {
        if self.total_cycles == 0 {
            return 1.0;
        }
        self.total_tokens_committed as f32 / self.total_cycles as f32
    }
}
