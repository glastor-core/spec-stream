use crate::error::CoreError;
use crate::types::SamplingParams;
use rand::Rng;
use std::collections::HashMap;

/// Strongly-typed token ID representation.
pub type TokenId = u32;

/// A vector of raw unnormalized model logits.
#[derive(Debug, Clone, PartialEq)]
pub struct Logits {
    pub values: Vec<f32>,
}

impl Logits {
    /// Creates a new Logits instance from a raw vector of floats.
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Number of vocabulary items.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.values.len()
    }

    /// Computes the greedy argmax token index.
    pub fn argmax(&self) -> Result<TokenId, CoreError> {
        if self.values.is_empty() {
            return Err(CoreError::EmptyLogits);
        }

        let mut max_idx = 0;
        let mut max_val = self.values[0];

        for (idx, &val) in self.values.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        Ok(max_idx as TokenId)
    }

    /// Scales the logits by a positive temperature parameter.
    pub fn apply_temperature(&mut self, temperature: f32) -> Result<(), CoreError> {
        if temperature <= 0.0 || temperature.is_nan() {
            return Err(CoreError::InvalidTemperature(temperature));
        }

        let inv_temp = 1.0 / temperature;
        for val in &mut self.values {
            *val *= inv_temp;
        }

        Ok(())
    }

    /// Filters logits to keep only the top-K highest scoring tokens.
    /// All other positions are set to -INFINITY.
    pub fn apply_top_k(&mut self, k: usize) {
        if k == 0 || k >= self.values.len() {
            return;
        }

        // Find the k-th largest value by sorting indexed pairs or partitioning
        let mut sorted_vals: Vec<f32> = self
            .values
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect();
        sorted_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(&kth_val) = sorted_vals.get(k - 1) {
            let mut kept = 0;
            for val in &mut self.values {
                if *val > kth_val || (*val == kth_val && kept < k) {
                    kept += 1;
                } else {
                    *val = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// Filters logits using nucleus (top-p) sampling.
    /// Keeps the smallest set of tokens whose cumulative probability is >= top_p.
    pub fn apply_top_p(&mut self, top_p: f32) -> Result<(), CoreError> {
        if top_p <= 0.0 || top_p > 1.0 || top_p.is_nan() {
            return Err(CoreError::InvalidTopP(top_p));
        }

        if top_p >= 1.0 || self.values.len() <= 1 {
            return Ok(());
        }

        let probs = self.softmax()?;
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumulative = 0.0f32;
        let mut cutoff_idx = 0;

        for (rank, &(_, prob)) in indexed.iter().enumerate() {
            cumulative += prob;
            cutoff_idx = rank;
            if cumulative >= top_p {
                break;
            }
        }

        // Indices to keep: indexed[0..=cutoff_idx]
        let mut keep = vec![false; self.values.len()];
        for &(idx, _) in &indexed[0..=cutoff_idx] {
            keep[idx] = true;
        }

        for (idx, val) in self.values.iter_mut().enumerate() {
            if !keep[idx] {
                *val = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    /// Filters logits using min-p sampling.
    /// Retains tokens with probability >= min_p * max_probability.
    pub fn apply_min_p(&mut self, min_p: f32) -> Result<(), CoreError> {
        if !(0.0..=1.0).contains(&min_p) || min_p.is_nan() {
            return Err(CoreError::InvalidMinP(min_p));
        }

        if min_p <= 0.0 || self.values.is_empty() {
            return Ok(());
        }

        let probs = self.softmax()?;
        let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
        let threshold = min_p * max_prob;

        for (idx, &prob) in probs.iter().enumerate() {
            if prob < threshold {
                self.values[idx] = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    /// Applies repetition penalty to tokens that have already been generated in the context.
    pub fn apply_repetition_penalty(
        &mut self,
        context_tokens: &[TokenId],
        penalty: f32,
    ) -> Result<(), CoreError> {
        if penalty <= 0.0 || penalty.is_nan() {
            return Err(CoreError::InvalidPenalty(penalty));
        }

        if (penalty - 1.0).abs() < 1e-6 || context_tokens.is_empty() {
            return Ok(());
        }

        for &token_id in context_tokens {
            let idx = token_id as usize;
            if let Some(val) = self.values.get_mut(idx) {
                if *val > 0.0 {
                    *val /= penalty;
                } else {
                    *val *= penalty;
                }
            }
        }

        Ok(())
    }

    /// Applies presence and frequency penalties (OpenAI style).
    pub fn apply_presence_and_frequency_penalties(
        &mut self,
        context_tokens: &[TokenId],
        presence_penalty: f32,
        frequency_penalty: f32,
    ) {
        if (presence_penalty.abs() < 1e-6 && frequency_penalty.abs() < 1e-6)
            || context_tokens.is_empty()
        {
            return;
        }

        let mut counts = HashMap::new();
        for &tok in context_tokens {
            *counts.entry(tok).or_insert(0usize) += 1;
        }

        for (tok, count) in counts {
            let idx = tok as usize;
            if let Some(val) = self.values.get_mut(idx) {
                *val -= presence_penalty + (count as f32 * frequency_penalty);
            }
        }
    }

    /// Computes numerically stable softmax probabilities.
    pub fn softmax(&self) -> Result<Vec<f32>, CoreError> {
        if self.values.is_empty() {
            return Err(CoreError::EmptyLogits);
        }

        let max_val = self
            .values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut exp_sum = 0.0f32;
        let mut probs = Vec::with_capacity(self.values.len());

        for &v in &self.values {
            let exp_v = if v == f32::NEG_INFINITY {
                0.0
            } else {
                (v - max_val).exp()
            };
            probs.push(exp_v);
            exp_sum += exp_v;
        }

        if exp_sum <= 0.0 || exp_sum.is_nan() {
            let uniform = 1.0 / self.values.len() as f32;
            return Ok(vec![uniform; self.values.len()]);
        }

        let inv_sum = 1.0 / exp_sum;
        for p in &mut probs {
            *p *= inv_sum;
        }

        Ok(probs)
    }

    /// Samples a token from the distribution using standard inverse transform sampling.
    pub fn sample<R: Rng>(&self, temperature: f32, rng: &mut R) -> Result<TokenId, CoreError> {
        if temperature < 1e-4 {
            return self.argmax();
        }

        let mut scaled = self.clone();
        scaled.apply_temperature(temperature)?;
        let probs = scaled.softmax()?;

        let sample_val: f32 = rng.gen();
        let mut cumulative = 0.0f32;

        for (idx, &p) in probs.iter().enumerate() {
            cumulative += p;
            if sample_val <= cumulative {
                return Ok(idx as TokenId);
            }
        }

        // Fallback in case of floating-point rounding
        Ok((self.values.len() - 1) as TokenId)
    }

    /// Samples a token using full sampling parameters (temperature, top_k, top_p, penalties).
    pub fn sample_with_params<R: Rng>(
        &self,
        params: &SamplingParams,
        context_tokens: &[TokenId],
        rng: &mut R,
    ) -> Result<TokenId, CoreError> {
        if params.temperature < 1e-4 {
            let mut adjusted = self.clone();
            adjusted.apply_repetition_penalty(context_tokens, params.repetition_penalty)?;
            adjusted.apply_presence_and_frequency_penalties(
                context_tokens,
                params.presence_penalty,
                params.frequency_penalty,
            );
            return adjusted.argmax();
        }

        let mut processed = self.clone();
        processed.apply_repetition_penalty(context_tokens, params.repetition_penalty)?;
        processed.apply_presence_and_frequency_penalties(
            context_tokens,
            params.presence_penalty,
            params.frequency_penalty,
        );
        processed.apply_temperature(params.temperature)?;

        if params.top_k > 0 {
            processed.apply_top_k(params.top_k);
        }

        if params.top_p < 1.0 {
            processed.apply_top_p(params.top_p)?;
        }

        if params.min_p > 0.0 {
            processed.apply_min_p(params.min_p)?;
        }

        let probs = processed.softmax()?;
        let sample_val: f32 = rng.gen();
        let mut cumulative = 0.0f32;

        for (idx, &p) in probs.iter().enumerate() {
            cumulative += p;
            if sample_val <= cumulative {
                return Ok(idx as TokenId);
            }
        }

        Ok((self.values.len() - 1) as TokenId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_top_k_filtering() {
        let mut logits = Logits::new(vec![1.0, 10.0, 5.0, 2.0, 8.0]);
        logits.apply_top_k(2); // Should keep 10.0 (idx 1) and 8.0 (idx 4)
        assert_eq!(logits.values[1], 10.0);
        assert_eq!(logits.values[4], 8.0);
        assert_eq!(logits.values[0], f32::NEG_INFINITY);
        assert_eq!(logits.values[2], f32::NEG_INFINITY);
        assert_eq!(logits.values[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_p_filtering() {
        let mut logits = Logits::new(vec![10.0, 1.0, 0.5, 0.1]);
        logits.apply_top_p(0.8).unwrap();
        // Index 0 has vastly dominant probability (~0.999), so only it is kept
        assert_eq!(logits.values[0], 10.0);
        assert_eq!(logits.values[1], f32::NEG_INFINITY);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = Logits::new(vec![4.0, -2.0, 5.0]);
        logits.apply_repetition_penalty(&[0, 1], 2.0).unwrap();
        assert!((logits.values[0] - 2.0).abs() < 1e-5); // 4.0 / 2.0
        assert!((logits.values[1] - (-4.0)).abs() < 1e-5); // -2.0 * 2.0
        assert_eq!(logits.values[2], 5.0); // unchanged
    }

    #[test]
    fn test_sampling_with_params() {
        let logits = Logits::new(vec![1.0, 10.0, 2.0]);
        let mut rng = StdRng::seed_from_u64(42);
        let params = SamplingParams {
            temperature: 0.1,
            top_k: 1,
            ..Default::default()
        };
        let token = logits.sample_with_params(&params, &[], &mut rng).unwrap();
        assert_eq!(token, 1);
    }
}
