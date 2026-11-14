use crate::logits::TokenId;
use smallvec::SmallVec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a single candidate token produced by the draft model.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TokenCandidate {
    /// Token identifier.
    pub token_id: TokenId,
    /// Draft probability assigned to this token (used in stochastic rejection sampling).
    pub draft_prob: f32,
    /// Sequential position index within the speculative burst.
    pub position: usize,
}

/// Represents the speculative draft sequence proposed by the draft model.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DraftProposal {
    /// List of proposed token candidates.
    pub candidates: SmallVec<[TokenCandidate; 8]>,
}

impl DraftProposal {
    pub fn new() -> Self {
        Self {
            candidates: SmallVec::new(),
        }
    }

    pub fn push(&mut self, token_id: TokenId, draft_prob: f32) {
        let position = self.candidates.len();
        self.candidates.push(TokenCandidate {
            token_id,
            draft_prob,
            position,
        });
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn token_ids(&self) -> Vec<TokenId> {
        self.candidates.iter().map(|c| c.token_id).collect()
    }
}

/// The outcome of the speculative verification stage executed against the target model.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VerificationResult {
    /// Candidates from the draft model that were successfully verified.
    pub accepted_tokens: SmallVec<[TokenId; 8]>,
    /// The next token sampled from the target model distribution.
    /// This is either:
    /// - The correction token if a candidate was rejected.
    /// - The bonus next-token if all K candidates were accepted!
    pub emitted_target_token: TokenId,
    /// How many draft tokens were accepted before the first rejection (0..=K).
    pub accepted_count: usize,
    /// Total number of tokens committed in this single speculative cycle
    /// (accepted_count + 1 emitted token).
    pub total_tokens_produced: usize,
    /// The index in the draft sequence where rejection occurred (or None if all accepted).
    pub rejection_index: Option<usize>,
    /// How many positions in the KV-cache need to be preserved from the draft.
    pub kv_valid_length: usize,
}

impl VerificationResult {
    /// Returns an iterator over all committed tokens in order:
    /// [accepted_0, accepted_1, ..., accepted_k, emitted_target_token]
    pub fn all_committed_tokens(&self) -> impl Iterator<Item = TokenId> + '_ {
        self.accepted_tokens
            .iter()
            .copied()
            .chain(std::iter::once(self.emitted_target_token))
    }
}

/// Strategy for accepting or rejecting proposed speculative tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerificationStrategy {
    /// Greedy decoding: Accept candidate if and only if target argmax == draft token.
    Greedy,
    /// Stochastic rejection sampling adhering to Leviathan et al. (2023).
    /// Guarantees that the resulting token stream strictly follows the target model's probability distribution.
    Stochastic { temperature: f32 },
}

/// Advanced sampling configuration for token generation and logits processing.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SamplingParams {
    /// Temperature scaling (> 0.0). Defaults to 1.0 (or 0.0 for pure greedy).
    pub temperature: f32,
    /// Top-K filtering: keep only top K highest probability tokens (0 = disabled).
    pub top_k: usize,
    /// Top-P (nucleus) filtering: keep cumulative probability <= top_p (1.0 = disabled).
    pub top_p: f32,
    /// Min-P filtering: keep tokens with probability >= min_p * max_prob (0.0 = disabled).
    pub min_p: f32,
    /// Repetition penalty factor (> 0.0, 1.0 = disabled, > 1.0 = penalize).
    pub repetition_penalty: f32,
    /// Presence penalty (-2.0..2.0, 0.0 = disabled).
    pub presence_penalty: f32,
    /// Frequency penalty (-2.0..2.0, 0.0 = disabled).
    pub frequency_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}
