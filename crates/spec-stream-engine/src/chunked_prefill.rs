//! # Chunked Prefill Scheduler
//!
//! Provides chunked prefill scheduling to prevent head-of-line blocking and memory spikes
//! during long context ingestion.
//!
//! Large prompts (e.g. 2k–8k tokens) are segmented into configurable chunks (e.g. 128–512 tokens),
//! interleaving prompt KV-cache prefilling with active speculative decoding bursts.

use crate::error::EngineError;
use crate::model::ModelEngine;
use spec_stream_cache::PagedKvCacheManager;
use spec_stream_core::TokenId;
use std::collections::VecDeque;
use std::sync::Arc;

/// Configuration for chunked prefill processing.
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// Maximum number of prompt tokens processed in a single prefill step (default: 128).
    pub chunk_size: usize,
    /// Maximum number of concurrent prompts in prefill stage (default: 4).
    pub max_concurrent_prefills: usize,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 128,
            max_concurrent_prefills: 4,
        }
    }
}

/// An active prompt ingestion task being processed in chunks.
#[derive(Debug, Clone)]
pub struct PrefillTask {
    pub sequence_id: u64,
    pub prompt: Vec<TokenId>,
    pub processed_tokens: usize,
}

impl PrefillTask {
    pub fn new(sequence_id: u64, prompt: Vec<TokenId>) -> Self {
        Self {
            sequence_id,
            prompt,
            processed_tokens: 0,
        }
    }

    /// Whether all prompt tokens have been completely ingested into the KV-cache.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.processed_tokens >= self.prompt.len()
    }

    /// Number of remaining tokens left to prefill.
    #[inline]
    pub fn remaining_tokens(&self) -> usize {
        self.prompt.len().saturating_sub(self.processed_tokens)
    }

    /// Progress ratio between 0.0 and 1.0.
    #[inline]
    pub fn progress(&self) -> f32 {
        if self.prompt.is_empty() {
            1.0
        } else {
            self.processed_tokens as f32 / self.prompt.len() as f32
        }
    }
}

/// Chunked prefill orchestrator.
pub struct ChunkedPrefillScheduler {
    config: ChunkedPrefillConfig,
    cache_manager: Arc<PagedKvCacheManager>,
    pending_tasks: VecDeque<PrefillTask>,
    completed_tasks: Vec<u64>,
}

impl ChunkedPrefillScheduler {
    pub fn new(cache_manager: Arc<PagedKvCacheManager>, config: ChunkedPrefillConfig) -> Self {
        Self {
            config,
            cache_manager,
            pending_tasks: VecDeque::new(),
            completed_tasks: Vec::new(),
        }
    }

    /// Submits a new prompt sequence for chunked prefill.
    pub fn submit_prompt(&mut self, sequence_id: u64, prompt: Vec<TokenId>) {
        self.cache_manager.create_sequence(sequence_id);
        self.pending_tasks
            .push_back(PrefillTask::new(sequence_id, prompt));
    }

    /// Number of sequences currently undergoing chunked prefill.
    pub fn pending_count(&self) -> usize {
        self.pending_tasks.len()
    }

    /// Executes one chunked prefill step for the front task using the target model.
    /// Returns the number of tokens prefilled in this step, and whether the sequence completed.
    pub async fn step_prefill<M: ModelEngine>(
        &mut self,
        model: &mut M,
    ) -> Result<Option<(u64, usize, bool)>, EngineError> {
        let mut task = match self.pending_tasks.pop_front() {
            Some(t) => t,
            None => return Ok(None),
        };

        if task.is_complete() {
            self.completed_tasks.push(task.sequence_id);
            return Ok(Some((task.sequence_id, 0, true)));
        }

        let start = task.processed_tokens;
        let end = (start + self.config.chunk_size).min(task.prompt.len());
        let chunk = &task.prompt[start..end];
        let chunk_len = chunk.len();

        // Append this chunk to the Paged KV-Cache
        self.cache_manager.append_tokens(task.sequence_id, chunk)?;

        // Execute batch forward pass for this prefill chunk
        let _logits = model.forward_batch(chunk).await?;

        task.processed_tokens += chunk_len;
        let is_now_complete = task.is_complete();

        if is_now_complete {
            self.completed_tasks.push(task.sequence_id);
        } else {
            // Re-enqueue task for subsequent chunk processing
            self.pending_tasks.push_back(task.clone());
        }

        Ok(Some((task.sequence_id, chunk_len, is_now_complete)))
    }

    /// Drains IDs of sequences that finished their prefill stage and are ready for speculative decoding.
    pub fn take_completed_sequences(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.completed_tasks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SyntheticModel;

    #[tokio::test]
    async fn test_chunked_prefill_step_execution() {
        let cache = Arc::new(PagedKvCacheManager::with_defaults(64));
        let config = ChunkedPrefillConfig {
            chunk_size: 10,
            max_concurrent_prefills: 2,
        };

        let mut scheduler = ChunkedPrefillScheduler::new(cache.clone(), config);
        let mut model = SyntheticModel::new("test-model", 100, 0.9, 42);

        // Prompt of 25 tokens -> chunks of 10, 10, 5
        let prompt: Vec<TokenId> = (1..=25).collect();
        scheduler.submit_prompt(501, prompt);

        assert_eq!(scheduler.pending_count(), 1);

        // Step 1: 10 tokens
        let step1 = scheduler.step_prefill(&mut model).await.unwrap().unwrap();
        assert_eq!(step1, (501, 10, false));

        // Step 2: 10 tokens
        let step2 = scheduler.step_prefill(&mut model).await.unwrap().unwrap();
        assert_eq!(step2, (501, 10, false));

        // Step 3: 5 tokens -> complete!
        let step3 = scheduler.step_prefill(&mut model).await.unwrap().unwrap();
        assert_eq!(step3, (501, 5, true));

        let completed = scheduler.take_completed_sequences();
        assert_eq!(completed, vec![501]);
        assert_eq!(scheduler.pending_count(), 0);
    }
}
