use serde::{Deserialize, Serialize};
use spec_stream_core::{SpeculativeMetrics, TokenId};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc::Receiver;
use tokio_stream::Stream;

/// Events emitted during the speculative generation stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum GenerationEvent {
    /// A committed token ready for output.
    Token {
        token_id: TokenId,
        /// True if this was an accepted draft candidate, false if target correction/bonus.
        is_draft: bool,
    },
    /// Telemetry burst summary for the speculative cycle.
    Burst {
        accepted: usize,
        total_produced: usize,
    },
    /// Adaptive K changed based on empirical acceptance rate.
    AdaptiveKChanged {
        old_k: usize,
        new_k: usize,
        acceptance_rate: f32,
    },
    /// Updated performance metrics and speedup ratio.
    Metrics(SpeculativeMetrics),
    /// Generation stream finished.
    Finished {
        total_tokens: usize,
        reason: String,
    },
}

/// An asynchronous stream producing speculative generation events.
pub struct SpeculativeStream {
    receiver: Receiver<GenerationEvent>,
}

impl SpeculativeStream {
    pub fn new(receiver: Receiver<GenerationEvent>) -> Self {
        Self { receiver }
    }
}

impl Stream for SpeculativeStream {
    type Item = GenerationEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

