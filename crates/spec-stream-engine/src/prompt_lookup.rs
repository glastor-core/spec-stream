//! # Prompt Lookup Speculative Decoding Engine
//!
//! Implements Prompt Lookup Decoding (n-gram assisted drafting), inspired by
//! Saxena & Arora (2023) and vLLM's zero-model speculative decoding.
//!
//! In tasks like RAG, code editing, summarization, and structured generation,
//! generated tokens are frequently repeated substrings of the prompt or recent context.
//! This engine extracts candidate tokens directly from the prompt in nanoseconds
//! with **zero GPU compute** and **zero draft model weights**.

use crate::error::EngineError;
use crate::model::ModelEngine;
use spec_stream_core::{DraftProposal, Logits, TokenId};

/// Configuration for prompt lookup n-gram search.
#[derive(Debug, Clone)]
pub struct PromptLookupConfig {
    /// Maximum n-gram size to search for matching suffixes (default: 3).
    pub max_ngram_size: usize,
    /// Minimum n-gram size allowed for a valid match (default: 1).
    pub min_ngram_size: usize,
    /// Default vocab size for generating synthetic logits.
    pub vocab_size: usize,
}

impl Default for PromptLookupConfig {
    fn default() -> Self {
        Self {
            max_ngram_size: 3,
            min_ngram_size: 1,
            vocab_size: 32000,
        }
    }
}

/// Prompt Lookup speculative drafting engine.
///
/// Searches for matching n-gram suffixes in the prompt context to predict subsequent tokens.
#[derive(Debug, Clone)]
pub struct PromptLookupEngine {
    /// The reference prompt or document context tokens.
    context: Vec<TokenId>,
    config: PromptLookupConfig,
    /// Step counter for telemetry.
    step_counter: usize,
}

impl PromptLookupEngine {
    /// Creates a new prompt lookup engine initialized with a prompt context.
    pub fn new(context: Vec<TokenId>, config: PromptLookupConfig) -> Self {
        Self {
            context,
            config,
            step_counter: 0,
        }
    }

    /// Appends generated tokens to the active context pool so the engine learns from ongoing generation.
    pub fn append_to_context(&mut self, tokens: &[TokenId]) {
        self.context.extend_from_slice(tokens);
    }

    /// Finds candidate tokens by matching the suffix of `history` against `self.context`.
    pub fn find_candidate_tokens(&self, history: &[TokenId], k: usize) -> Vec<TokenId> {
        if history.is_empty() || self.context.is_empty() || k == 0 {
            return Vec::new();
        }

        let max_n = self.config.max_ngram_size.min(history.len());
        let min_n = self.config.min_ngram_size;

        // Try largest n-gram match first down to min_n
        for n in (min_n..=max_n).rev() {
            let query_ngram = &history[history.len() - n..];

            // Scan context for the latest match of this n-gram (excluding the exact tail)
            if self.context.len() < n + 1 {
                continue;
            }

            let search_limit = self.context.len().saturating_sub(n);
            // Search backwards so we prioritize the most recent context
            for i in (0..search_limit).rev() {
                if &self.context[i..i + n] == query_ngram {
                    // Match found! Extract up to k subsequent tokens
                    let start = i + n;
                    let end = (start + k).min(self.context.len());
                    let candidates = self.context[start..end].to_vec();
                    if !candidates.is_empty() {
                        return candidates;
                    }
                }
            }
        }

        Vec::new()
    }

    /// Generates a speculative `DraftProposal` directly from prompt n-gram matching.
    pub fn draft_proposal(&self, history: &[TokenId], k: usize) -> DraftProposal {
        let candidates = self.find_candidate_tokens(history, k);
        let mut proposal = DraftProposal::new();

        for (idx, &token_id) in candidates.iter().enumerate() {
            // High confidence for prompt matches (scaling slightly with position)
            let prob = 0.95 - (idx as f32 * 0.05).max(0.0);
            proposal.push(token_id, prob);
        }

        proposal
    }
}

impl ModelEngine for PromptLookupEngine {
    async fn forward_batch(&mut self, tokens: &[TokenId]) -> Result<Vec<Logits>, EngineError> {
        self.step_counter += 1;
        let mut results = Vec::with_capacity(tokens.len() + 1);

        // For each token in tokens, predict next token using lookup
        for (idx, &tok) in tokens.iter().enumerate() {
            let history = &tokens[..=idx];
            let candidates = self.find_candidate_tokens(history, 1);
            let next_tok = candidates.first().copied().unwrap_or(tok);

            let mut vals = vec![0.0f32; self.config.vocab_size];
            if (next_tok as usize) < self.config.vocab_size {
                vals[next_tok as usize] = 10.0;
            }
            results.push(Logits::new(vals));
        }

        // Bonus prediction
        let last_token = tokens.last().copied().unwrap_or(1);
        let mut bonus_vals = vec![0.0f32; self.config.vocab_size];
        let bonus_idx = (last_token as usize + 1) % self.config.vocab_size;
        bonus_vals[bonus_idx] = 10.0;
        results.push(Logits::new(bonus_vals));

        Ok(results)
    }

    async fn forward_token(&mut self, token: TokenId) -> Result<Logits, EngineError> {
        self.step_counter += 1;
        let candidates = self.find_candidate_tokens(&[token], 1);
        let next_tok = candidates.first().copied().unwrap_or(token);

        let mut vals = vec![0.0f32; self.config.vocab_size];
        if (next_tok as usize) < self.config.vocab_size {
            vals[next_tok as usize] = 10.0;
        }

        Ok(Logits::new(vals))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_lookup_exact_ngram_match() {
        // Prompt context: "The quick brown fox jumps over the lazy dog"
        // Tokens:        [ 10,  20,   30,   40, 50,   60,  10, 70,  80 ]
        let context = vec![10, 20, 30, 40, 50, 60, 10, 70, 80];
        let engine = PromptLookupEngine::new(context, PromptLookupConfig::default());

        // History: "The quick" -> [10, 20]
        // Should predict: "brown fox jumps" -> [30, 40, 50]
        let candidates = engine.find_candidate_tokens(&[10, 20], 3);
        assert_eq!(candidates, vec![30, 40, 50]);
    }

    #[test]
    fn test_prompt_lookup_fallback_smaller_ngram() {
        let context = vec![100, 200, 300, 400];
        let engine = PromptLookupEngine::new(context, PromptLookupConfig::default());

        // Query: [999, 200] -> 2-gram [999, 200] has no match, but 1-gram [200] matches!
        let candidates = engine.find_candidate_tokens(&[999, 200], 2);
        assert_eq!(candidates, vec![300, 400]);
    }

    #[test]
    fn test_prompt_lookup_draft_proposal() {
        let context = vec![1, 2, 3, 4, 5, 6, 7];
        let engine = PromptLookupEngine::new(context, PromptLookupConfig::default());

        let proposal = engine.draft_proposal(&[1, 2], 3);
        assert_eq!(proposal.len(), 3);
        assert_eq!(proposal.token_ids(), vec![3, 4, 5]);
        assert!(proposal.candidates[0].draft_prob > 0.8);
    }
}
