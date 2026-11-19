use crate::error::CoreError;
use crate::logits::{Logits, TokenId};
use crate::types::{DraftProposal, VerificationResult, VerificationStrategy};
use rand::{rngs::StdRng, Rng, RngExt, SeedableRng};
use smallvec::SmallVec;

/// Speculative decoding verifier engine.
/// Compares draft candidates against target model logits in parallel.
#[derive(Debug, Clone)]
pub struct SpeculativeVerifier {
    strategy: VerificationStrategy,
}

impl SpeculativeVerifier {
    pub fn new(strategy: VerificationStrategy) -> Self {
        Self { strategy }
    }

    pub fn greedy() -> Self {
        Self {
            strategy: VerificationStrategy::Greedy,
        }
    }

    pub fn stochastic(temperature: f32) -> Result<Self, CoreError> {
        if temperature <= 0.0 || temperature.is_nan() {
            return Err(CoreError::InvalidTemperature(temperature));
        }
        Ok(Self {
            strategy: VerificationStrategy::Stochastic { temperature },
        })
    }

    /// Verifies draft candidates against the target model's logits batch.
    ///
    /// * `proposal`: The $K$ candidate tokens proposed by the draft model.
    /// * `target_logits`: The $K + 1$ logit slices computed by the target model in a single forward pass:
    ///   - `target_logits[i]` predicts the token following candidate prefix `[0..i]`.
    ///   - `target_logits[K]` predicts the bonus token if all $K$ tokens are accepted.
    pub fn verify(
        &self,
        proposal: &DraftProposal,
        target_logits: &[Logits],
    ) -> Result<VerificationResult, CoreError> {
        let mut rng = StdRng::from_rng(&mut rand::rng());
        self.verify_with_rng(proposal, target_logits, &mut rng)
    }

    /// Verifies draft candidates using a provided pseudo-random generator (for deterministic tests).
    pub fn verify_with_rng<R: Rng>(
        &self,
        proposal: &DraftProposal,
        target_logits: &[Logits],
        rng: &mut R,
    ) -> Result<VerificationResult, CoreError> {
        let k = proposal.len();
        if k == 0 {
            return Err(CoreError::EmptyDraftProposal);
        }

        // Target model must evaluate all k prefix positions plus the trailing bonus position
        if target_logits.len() < k + 1 {
            return Err(CoreError::DimensionMismatch {
                expected: k + 1,
                actual: target_logits.len(),
            });
        }

        match self.strategy {
            VerificationStrategy::Greedy => self.verify_greedy(proposal, target_logits),
            VerificationStrategy::Stochastic { temperature } => {
                self.verify_stochastic(proposal, target_logits, temperature, rng)
            }
        }
    }

    fn verify_greedy(
        &self,
        proposal: &DraftProposal,
        target_logits: &[Logits],
    ) -> Result<VerificationResult, CoreError> {
        let mut accepted_tokens = SmallVec::new();
        let k = proposal.len();

        for (i, candidate) in proposal.candidates.iter().enumerate() {
            let target_pred = target_logits[i].argmax()?;

            if target_pred == candidate.token_id {
                accepted_tokens.push(candidate.token_id);
            } else {
                // Mismatch: reject candidate at index i.
                // The target prediction at this index serves as the correction token.
                return Ok(VerificationResult {
                    accepted_count: accepted_tokens.len(),
                    accepted_tokens,
                    emitted_target_token: target_pred,
                    total_tokens_produced: i + 1,
                    rejection_index: Some(i),
                    kv_valid_length: i, // keep draft KV-cache up to index i
                });
            }
        }

        // All K tokens were accepted!
        // Sample the (K+1)-th bonus token from target_logits[K]
        let bonus_token = target_logits[k].argmax()?;

        Ok(VerificationResult {
            accepted_count: k,
            accepted_tokens,
            emitted_target_token: bonus_token,
            total_tokens_produced: k + 1,
            rejection_index: None,
            kv_valid_length: k,
        })
    }

    fn verify_stochastic<R: Rng>(
        &self,
        proposal: &DraftProposal,
        target_logits: &[Logits],
        temperature: f32,
        rng: &mut R,
    ) -> Result<VerificationResult, CoreError> {
        let mut accepted_tokens = SmallVec::new();
        let k = proposal.len();

        for (i, candidate) in proposal.candidates.iter().enumerate() {
            let mut target_scaled = target_logits[i].clone();
            target_scaled.apply_temperature(temperature)?;
            let target_probs = target_scaled.softmax()?;

            let token_idx = candidate.token_id as usize;
            if token_idx >= target_probs.len() {
                return Err(CoreError::DimensionMismatch {
                    expected: target_probs.len(),
                    actual: token_idx,
                });
            }

            let p_target = target_probs[token_idx];
            let q_draft = candidate.draft_prob.max(1e-8);

            let r: f32 = rng.random();
            let acceptance_ratio = (p_target / q_draft).min(1.0);

            if r <= acceptance_ratio {
                // Accepted
                accepted_tokens.push(candidate.token_id);
            } else {
                // Rejected: Sample correction token from normalized max(0, p(x) - q(x))
                let correction_token = self.sample_residual(&target_probs, proposal, i, rng)?;

                let accepted_count = accepted_tokens.len();
                return Ok(VerificationResult {
                    accepted_count,
                    accepted_tokens,
                    emitted_target_token: correction_token,
                    total_tokens_produced: accepted_count + 1,
                    rejection_index: Some(i),
                    kv_valid_length: i,
                });
            }
        }

        // All K accepted! Sample bonus token from target_logits[K]
        let mut bonus_scaled = target_logits[k].clone();
        bonus_scaled.apply_temperature(temperature)?;
        let bonus_token = bonus_scaled.sample(temperature, rng)?;

        Ok(VerificationResult {
            accepted_count: k,
            accepted_tokens,
            emitted_target_token: bonus_token,
            total_tokens_produced: k + 1,
            rejection_index: None,
            kv_valid_length: k,
        })
    }

    /// Samples from the positive difference distribution: max(0, p(x) - q(x))
    fn sample_residual<R: Rng>(
        &self,
        target_probs: &[f32],
        proposal: &DraftProposal,
        rejected_step: usize,
        rng: &mut R,
    ) -> Result<TokenId, CoreError> {
        let rejected_cand = &proposal.candidates[rejected_step];
        let rejected_id = rejected_cand.token_id as usize;

        let mut residual = Vec::with_capacity(target_probs.len());
        let mut sum = 0.0f32;

        for (idx, &p) in target_probs.iter().enumerate() {
            let q = if idx == rejected_id {
                rejected_cand.draft_prob
            } else {
                0.0
            };
            let diff = (p - q).max(0.0);
            residual.push(diff);
            sum += diff;
        }

        if sum <= 1e-8 || sum.is_nan() {
            // Fallback: standard sample from target_probs directly
            let sample_val: f32 = rng.random();
            let mut accum = 0.0f32;
            for (idx, &p) in target_probs.iter().enumerate() {
                accum += p;
                if sample_val <= accum {
                    return Ok(idx as TokenId);
                }
            }
            return Ok((target_probs.len() - 1) as TokenId);
        }

        let inv_sum = 1.0 / sum;
        for diff in &mut residual {
            *diff *= inv_sum;
        }

        let sample_val: f32 = rng.random();
        let mut accum = 0.0f32;
        for (idx, &p) in residual.iter().enumerate() {
            accum += p;
            if sample_val <= accum {
                return Ok(idx as TokenId);
            }
        }

        Ok((residual.len() - 1) as TokenId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logits(vocab_size: usize, target_token: usize) -> Logits {
        let mut vals = vec![0.0f32; vocab_size];
        vals[target_token] = 10.0; // high probability peak
        Logits::new(vals)
    }

    #[test]
    fn test_greedy_all_accepted() {
        let verifier = SpeculativeVerifier::greedy();
        let mut proposal = DraftProposal::new();
        proposal.push(10, 0.9);
        proposal.push(20, 0.8);
        proposal.push(30, 0.7);

        let target_logits = vec![
            make_logits(50, 10), // matches candidate 0
            make_logits(50, 20), // matches candidate 1
            make_logits(50, 30), // matches candidate 2
            make_logits(50, 42), // bonus token
        ];

        let res = verifier.verify(&proposal, &target_logits).unwrap();
        assert_eq!(res.accepted_count, 3);
        assert_eq!(res.accepted_tokens.as_slice(), &[10, 20, 30]);
        assert_eq!(res.emitted_target_token, 42);
        assert_eq!(res.total_tokens_produced, 4);
        assert_eq!(res.rejection_index, None);
        assert_eq!(res.kv_valid_length, 3);
    }

    #[test]
    fn test_greedy_early_rejection() {
        let verifier = SpeculativeVerifier::greedy();
        let mut proposal = DraftProposal::new();
        proposal.push(10, 0.9);
        proposal.push(20, 0.8); // will be rejected
        proposal.push(30, 0.7);

        let target_logits = vec![
            make_logits(50, 10), // matches candidate 0
            make_logits(50, 49), // MISMATCH! Target prefers 49
            make_logits(50, 30),
            make_logits(50, 42),
        ];

        let res = verifier.verify(&proposal, &target_logits).unwrap();
        assert_eq!(res.accepted_count, 1);
        assert_eq!(res.accepted_tokens.as_slice(), &[10]);
        assert_eq!(res.emitted_target_token, 49); // correction token
        assert_eq!(res.total_tokens_produced, 2);
        assert_eq!(res.rejection_index, Some(1));
        assert_eq!(res.kv_valid_length, 1);
    }
}
