use crate::error::EngineError;
use rand::{rngs::StdRng, Rng, SeedableRng};
use spec_stream_core::{Logits, TokenId};
use std::future::Future;

/// Abstract inference engine interface for draft and target models.
/// Compatible with Candle, Burn, ONNX Runtime, or custom model runtimes.
pub trait ModelEngine: Send + Sync {
    /// Computes logits for a sequence of tokens in a single parallel forward pass.
    fn forward_batch(
        &mut self,
        tokens: &[TokenId],
    ) -> impl Future<Output = Result<Vec<Logits>, EngineError>> + Send;

    /// Computes logits for a single token step.
    fn forward_token(
        &mut self,
        token: TokenId,
    ) -> impl Future<Output = Result<Logits, EngineError>> + Send;
}

/// A deterministic synthetic model for benchmarking, testing, and pipeline demonstrations.
/// Simulates model predictions with a controllable agreement/acceptance probability.
#[derive(Debug, Clone)]
pub struct SyntheticModel {
    pub name: String,
    pub vocab_size: usize,
    pub agreement_rate: f32,
    pub step_counter: usize,
    rng: StdRng,
}

impl SyntheticModel {
    pub fn new(name: impl Into<String>, vocab_size: usize, agreement_rate: f32, seed: u64) -> Self {
        Self {
            name: name.into(),
            vocab_size,
            agreement_rate,
            step_counter: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn generate_logits_for(&mut self, expected_token: TokenId) -> Logits {
        let mut vals = vec![0.0f32; self.vocab_size];
        let will_agree = self.rng.gen::<f32>() <= self.agreement_rate;

        let target = if will_agree {
            expected_token as usize % self.vocab_size
        } else {
            self.rng.gen_range(0..self.vocab_size)
        };

        // Give the target token a prominent peak
        vals[target] = 8.0;

        // Add minor Gaussian-like noise to others
        for (i, v) in vals.iter_mut().enumerate() {
            if i != target {
                *v = self.rng.gen_range(0.0..1.5);
            }
        }

        Logits::new(vals)
    }
}

impl ModelEngine for SyntheticModel {
    async fn forward_batch(&mut self, tokens: &[TokenId]) -> Result<Vec<Logits>, EngineError> {
        self.step_counter += 1;
        let mut results = Vec::with_capacity(tokens.len() + 1);

        for &t in tokens {
            // Predict candidate following token t
            results.push(self.generate_logits_for(t + 1));
        }

        // The bonus prediction at the end
        let last_token = tokens.last().copied().unwrap_or(1);
        results.push(self.generate_logits_for(last_token + 2));

        Ok(results)
    }

    async fn forward_token(&mut self, token: TokenId) -> Result<Logits, EngineError> {
        self.step_counter += 1;
        Ok(self.generate_logits_for(token + 1))
    }
}

