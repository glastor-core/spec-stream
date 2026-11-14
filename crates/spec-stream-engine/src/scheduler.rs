use crate::error::EngineError;
use crate::model::ModelEngine;
use crate::pipeline::SpeculativeConfig;
use crate::stream::GenerationEvent;
use spec_stream_cache::PagedKvCacheManager;
use spec_stream_core::{DraftProposal, Logits, SpeculativeMetrics, SpeculativeVerifier, TokenId};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio::time::Instant;

/// An incoming inference request submitted to the batch scheduler.
pub struct QueuedRequest {
    pub sequence_id: u64,
    pub prompt: Vec<TokenId>,
    pub config: SpeculativeConfig,
    pub sender: mpsc::Sender<GenerationEvent>,
    pub priority: usize,
    pub created_at: Instant,
}

impl QueuedRequest {
    pub fn new(
        sequence_id: u64,
        prompt: Vec<TokenId>,
        config: SpeculativeConfig,
        sender: mpsc::Sender<GenerationEvent>,
    ) -> Self {
        Self {
            sequence_id,
            prompt,
            config,
            sender,
            priority: 0,
            created_at: Instant::now(),
        }
    }
}

/// An active in-flight sequence currently participating in iteration-level batching.
struct ActiveSequence {
    sequence_id: u64,
    current_prefix: Vec<TokenId>,
    total_generated: usize,
    config: SpeculativeConfig,
    current_k: usize,
    ema_acceptance: Option<f32>,
    metrics: SpeculativeMetrics,
    sender: mpsc::Sender<GenerationEvent>,
}

/// Real-time operational statistics of the continuous batch scheduler.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub active_sequences: usize,
    pub queued_requests: usize,
    pub completed_sequences: usize,
    pub total_tokens_emitted: usize,
    pub total_batch_steps: usize,
}

/// High-throughput continuous batch scheduler executing iteration-level speculative inference.
/// Dynamically pools multiple concurrent requests into unified forward passes.
pub struct ContinuousBatchScheduler<D: ModelEngine, T: ModelEngine> {
    draft_model: Arc<AsyncMutex<D>>,
    target_model: Arc<AsyncMutex<T>>,
    cache_manager: Arc<PagedKvCacheManager>,
    queue: Arc<StdMutex<VecDeque<QueuedRequest>>>,
    active: Arc<StdMutex<Vec<ActiveSequence>>>,
    max_batch_size: usize,
    completed_counter: Arc<AtomicUsize>,
    tokens_emitted_counter: Arc<AtomicUsize>,
    batch_steps_counter: Arc<AtomicUsize>,
}

impl<D: ModelEngine + 'static, T: ModelEngine + 'static> ContinuousBatchScheduler<D, T> {
    pub fn new(
        draft_model: D,
        target_model: T,
        cache_manager: Arc<PagedKvCacheManager>,
        max_batch_size: usize,
    ) -> Self {
        Self {
            draft_model: Arc::new(AsyncMutex::new(draft_model)),
            target_model: Arc::new(AsyncMutex::new(target_model)),
            cache_manager,
            queue: Arc::new(StdMutex::new(VecDeque::new())),
            active: Arc::new(StdMutex::new(Vec::new())),
            max_batch_size: max_batch_size.max(1),
            completed_counter: Arc::new(AtomicUsize::new(0)),
            tokens_emitted_counter: Arc::new(AtomicUsize::new(0)),
            batch_steps_counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Enqueues an inference request for continuous batch processing.
    pub fn submit(&self, request: QueuedRequest) -> Result<(), EngineError> {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(request);
        Ok(())
    }

    /// Number of requests currently waiting in the arrival queue.
    pub fn queue_len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Number of requests currently active in the generation batch.
    pub fn active_len(&self) -> usize {
        self.active.lock().unwrap().len()
    }

    /// Retrieves current scheduler telemetry.
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            active_sequences: self.active.lock().unwrap().len(),
            queued_requests: self.queue.lock().unwrap().len(),
            completed_sequences: self.completed_counter.load(Ordering::Relaxed),
            total_tokens_emitted: self.tokens_emitted_counter.load(Ordering::Relaxed),
            total_batch_steps: self.batch_steps_counter.load(Ordering::Relaxed),
        }
    }

    /// Executes one iteration of continuous batching across all currently scheduled sequences.
    /// Returns the total number of committed tokens produced across all sequences in this iteration.
    pub async fn step_iteration(&self) -> Result<usize, EngineError> {
        self.batch_steps_counter.fetch_add(1, Ordering::Relaxed);

        // 1. Admission Control: Promote queued requests up to max_batch_size
        {
            let mut active = self.active.lock().unwrap();
            let mut queue = self.queue.lock().unwrap();

            while active.len() < self.max_batch_size && !queue.is_empty() {
                // Check if the cache has sufficient blocks
                if self.cache_manager.free_blocks() < 2 {
                    break;
                }

                if let Some(req) = queue.pop_front() {
                    self.cache_manager.create_sequence(req.sequence_id);
                    let _ = self
                        .cache_manager
                        .append_tokens(req.sequence_id, &req.prompt);

                    let initial_k = req.config.k_draft_tokens;
                    active.push(ActiveSequence {
                        sequence_id: req.sequence_id,
                        current_prefix: req.prompt,
                        total_generated: 0,
                        config: req.config,
                        current_k: initial_k,
                        ema_acceptance: None,
                        metrics: SpeculativeMetrics::new(),
                        sender: req.sender,
                    });
                }
            }
        }

        let num_active = self.active.lock().unwrap().len();
        if num_active == 0 {
            return Ok(0);
        }

        // 2. Draft Proposals Phase (per sequence)
        struct SeqProposal {
            seq_idx: usize,
            last_token: TokenId,
            proposal: DraftProposal,
        }

        let mut proposals = Vec::with_capacity(num_active);
        {
            let active = self.active.lock().unwrap();
            for (idx, seq) in active.iter().enumerate() {
                if let Some(&last_t) = seq.current_prefix.last() {
                    proposals.push(SeqProposal {
                        seq_idx: idx,
                        last_token: last_t,
                        proposal: DraftProposal::new(),
                    });
                }
            }
        }

        // Generate draft tokens
        for p in &mut proposals {
            let k = self.active.lock().unwrap()[p.seq_idx].current_k;
            let mut draft_token = p.last_token;

            for _ in 0..k {
                let logits = {
                    let mut draft_guard = self.draft_model.lock().await;
                    draft_guard.forward_token(draft_token).await?
                };

                let next_id = match logits.argmax() {
                    Ok(id) => id,
                    Err(_) => break,
                };

                let prob = logits
                    .softmax()
                    .ok()
                    .and_then(|probs| probs.get(next_id as usize).copied())
                    .unwrap_or(0.85);

                p.proposal.push(next_id, prob);
                draft_token = next_id;
            }
        }

        // 3. Target Model Batch Forward Pass & Verification
        let mut total_tokens_this_step = 0;
        let mut finished_indices = Vec::new();

        for p in proposals {
            if p.proposal.is_empty() {
                continue;
            }

            let proposed_ids = p.proposal.token_ids();
            let seq_id = self.active.lock().unwrap()[p.seq_idx].sequence_id;
            let _ = self.cache_manager.append_tokens(seq_id, &proposed_ids);

            let mut target_inputs = Vec::with_capacity(proposed_ids.len() + 1);
            target_inputs.push(p.last_token);
            target_inputs.extend_from_slice(&proposed_ids);

            let target_logits: Vec<Logits> = {
                let mut target_guard = self.target_model.lock().await;
                target_guard.forward_batch(&target_inputs).await?
            };

            let verifier = {
                let active = self.active.lock().unwrap();
                SpeculativeVerifier::new(active[p.seq_idx].config.strategy)
            };

            let verif_result = match verifier.verify(&p.proposal, &target_logits) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Scheduler verification error: {:?}", e);
                    continue;
                }
            };

            // 4. KV-Cache Rollback & Commit in O(1)
            let mut active = self.active.lock().unwrap();
            let seq = &mut active[p.seq_idx];

            let valid_tokens_to_keep = seq.current_prefix.len() + verif_result.accepted_count;
            let _ = self
                .cache_manager
                .rollback_sequence(seq.sequence_id, valid_tokens_to_keep);
            let _ = self
                .cache_manager
                .append_tokens(seq.sequence_id, &[verif_result.emitted_target_token]);

            seq.metrics.record_cycle(
                p.proposal.len(),
                verif_result.accepted_count,
                verif_result.total_tokens_produced,
            );

            // Emit accepted draft tokens
            for &tok in &verif_result.accepted_tokens {
                seq.current_prefix.push(tok);
                seq.total_generated += 1;
                total_tokens_this_step += 1;
                let _ = seq.sender.try_send(GenerationEvent::Token {
                    token_id: tok,
                    is_draft: true,
                });
            }

            // Emit target token
            seq.current_prefix.push(verif_result.emitted_target_token);
            seq.total_generated += 1;
            total_tokens_this_step += 1;
            let _ = seq.sender.try_send(GenerationEvent::Token {
                token_id: verif_result.emitted_target_token,
                is_draft: false,
            });

            // Adaptive K update
            let acc_rate = verif_result.accepted_count as f32 / p.proposal.len().max(1) as f32;
            let ema = match seq.ema_acceptance {
                Some(prev) => 0.5 * acc_rate + 0.5 * prev,
                None => acc_rate,
            };
            seq.ema_acceptance = Some(ema);

            if seq.config.adaptive_k {
                if ema >= 0.75 && seq.current_k < seq.config.max_k {
                    seq.current_k += 1;
                } else if ema <= 0.35 && seq.current_k > seq.config.min_k {
                    seq.current_k -= 1;
                }
            }

            // Check termination conditions
            let eos_reached = seq.config.eos_token_id == Some(verif_result.emitted_target_token);
            let max_reached = seq.total_generated >= seq.config.max_new_tokens;

            if eos_reached || max_reached {
                let reason = if eos_reached { "EOS" } else { "MaxTokens" };
                let _ = seq.sender.try_send(GenerationEvent::Finished {
                    total_tokens: seq.total_generated,
                    reason: reason.into(),
                });
                finished_indices.push(p.seq_idx);
            }
        }

        self.tokens_emitted_counter
            .fetch_add(total_tokens_this_step, Ordering::Relaxed);

        // 5. Retire finished sequences and cleanup their cache
        if !finished_indices.is_empty() {
            finished_indices.sort_unstable_by(|a, b| b.cmp(a)); // remove in reverse order
            let mut active = self.active.lock().unwrap();
            for idx in finished_indices {
                let finished_seq = active.swap_remove(idx);
                let _ = self.cache_manager.remove_sequence(finished_seq.sequence_id);
                self.completed_counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(total_tokens_this_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SyntheticModel;

    #[tokio::test]
    async fn test_continuous_batch_scheduler() {
        let vocab_size = 50;
        let draft = SyntheticModel::new("draft", vocab_size, 0.8, 42);
        let target = SyntheticModel::new("target", vocab_size, 0.95, 1337);
        let cache = Arc::new(PagedKvCacheManager::with_defaults(64));

        let scheduler = ContinuousBatchScheduler::new(draft, target, cache, 4);

        let (tx1, mut rx1) = mpsc::channel(32);
        let (tx2, mut rx2) = mpsc::channel(32);

        let config1 = SpeculativeConfig {
            max_new_tokens: 6,
            ..Default::default()
        };
        let config2 = SpeculativeConfig {
            max_new_tokens: 8,
            ..Default::default()
        };

        scheduler
            .submit(QueuedRequest::new(101, vec![1, 2], config1, tx1))
            .unwrap();
        scheduler
            .submit(QueuedRequest::new(102, vec![3, 4], config2, tx2))
            .unwrap();

        assert_eq!(scheduler.queue_len(), 2);
        assert_eq!(scheduler.active_len(), 0);

        // Run iterations until both complete
        let mut iterations = 0;
        while scheduler.active_len() > 0 || scheduler.queue_len() > 0 {
            let tokens = scheduler.step_iteration().await.unwrap();
            iterations += 1;
            if iterations > 20 {
                break;
            }
            if tokens == 0 && scheduler.active_len() == 0 {
                break;
            }
        }

        let stats = scheduler.stats();
        assert_eq!(stats.completed_sequences, 2);
        assert!(stats.total_tokens_emitted >= 14);

        // Drain channels
        let mut tokens_req1 = 0;
        while let Ok(event) = rx1.try_recv() {
            if let GenerationEvent::Token { .. } = event {
                tokens_req1 += 1;
            }
        }
        assert!(tokens_req1 >= 6);

        let mut tokens_req2 = 0;
        while let Ok(event) = rx2.try_recv() {
            if let GenerationEvent::Token { .. } = event {
                tokens_req2 += 1;
            }
        }
        assert!(tokens_req2 >= 8);
    }
}
