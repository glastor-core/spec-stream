//! # GGUF Quantization Engine
//!
//! Provides pure-Rust, zero-dependency de-quantization and high-performance dot product
//! kernels for standard GGUF quantization formats:
//! - **Q8_0**: 8-bit quantized weights with FP16 block scales (34 bytes per 32 elements).
//! - **Q4_0**: 4-bit quantized weights with FP16 block scales (18 bytes per 32 elements).

use crate::error::EngineError;
use crate::model::ModelEngine;
use spec_stream_core::{Logits, TokenId};

/// Number of elements per GGML quantization block.
pub const QK_K: usize = 32;

/// Converts an IEEE 754 half-precision (FP16) 16-bit integer to a single-precision (FP32) float.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    let f_sign = sign << 31;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(f_sign);
        }
        // Subnormal number
        let mut m = mant;
        let mut e = 0;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        m &= 0x3FF;
        let f_exp = (127 - 15 + 1 - e) << 23;
        let f_mant = m << 13;
        return f32::from_bits(f_sign | f_exp | f_mant);
    } else if exp == 0x1F {
        let f_exp = 0xFF << 23;
        let f_mant = mant << 13;
        return f32::from_bits(f_sign | f_exp | f_mant);
    }

    let f_exp = (exp + 127 - 15) << 23;
    let f_mant = mant << 13;
    f32::from_bits(f_sign | f_exp | f_mant)
}

/// Converts a single-precision float (FP32) into an IEEE 754 half-precision (FP16) integer.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf or NaN
        let h_mant = if mant != 0 { 0x200 } else { 0 };
        return ((sign << 15) | (0x1F << 10) | h_mant) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | (0x1F << 10)) as u16;
    } else if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let m = (mant | 0x800000) >> (1 - new_exp);
        return ((sign << 15) | (m >> 13)) as u16;
    }

    ((sign << 15) | ((new_exp as u32) << 10) | (mant >> 13)) as u16
}

/// Quantization format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// 8-bit quantization with FP16 scale per 32 elements.
    Q8_0,
    /// 4-bit quantization with FP16 scale per 32 elements.
    Q4_0,
}

/// Dequantizes a Q8_0 byte slice into an FP32 output buffer.
pub fn dequantize_q8_0(src: &[u8], dst: &mut [f32]) {
    assert_eq!(dst.len() % QK_K, 0, "Destination must be multiple of 32");
    let num_blocks = dst.len() / QK_K;
    assert_eq!(
        src.len(),
        num_blocks * 34,
        "Source length mismatch for Q8_0"
    );

    for b in 0..num_blocks {
        let src_offset = b * 34;
        let scale_bytes = [src[src_offset], src[src_offset + 1]];
        let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));

        let dst_offset = b * QK_K;
        for i in 0..QK_K {
            let q = src[src_offset + 2 + i] as i8;
            dst[dst_offset + i] = q as f32 * scale;
        }
    }
}

/// Dequantizes a Q4_0 byte slice into an FP32 output buffer.
pub fn dequantize_q4_0(src: &[u8], dst: &mut [f32]) {
    assert_eq!(dst.len() % QK_K, 0, "Destination must be multiple of 32");
    let num_blocks = dst.len() / QK_K;
    assert_eq!(
        src.len(),
        num_blocks * 18,
        "Source length mismatch for Q4_0"
    );

    for b in 0..num_blocks {
        let src_offset = b * 18;
        let scale_bytes = [src[src_offset], src[src_offset + 1]];
        let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));

        let dst_offset = b * QK_K;
        for i in 0..16 {
            let byte = src[src_offset + 2 + i];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;

            dst[dst_offset + i] = q0 as f32 * scale;
            dst[dst_offset + i + 16] = q1 as f32 * scale;
        }
    }
}

/// Computes the dot product between a Q8_0 quantized weight row and an FP32 vector directly without full dequantization.
pub fn vec_dot_q8_0(w_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Validated hardware feature presence via CPUID
            return unsafe { vec_dot_q8_0_avx2(w_bytes, x) };
        }
    }

    vec_dot_q8_0_scalar(w_bytes, x)
}

/// Fallback scalar implementation of Q8_0 dot product.
pub fn vec_dot_q8_0_scalar(w_bytes: &[u8], x: &[f32]) -> f32 {
    let num_blocks = x.len() / QK_K;
    let mut sum = 0.0f32;

    for b in 0..num_blocks {
        let src_offset = b * 34;
        let scale = f16_to_f32(u16::from_le_bytes([
            w_bytes[src_offset],
            w_bytes[src_offset + 1],
        ]));
        let x_offset = b * QK_K;

        let mut block_acc = 0.0f32;
        for i in 0..QK_K {
            let q = w_bytes[src_offset + 2 + i] as i8;
            block_acc += q as f32 * x[x_offset + i];
        }
        sum += block_acc * scale;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q8_0_avx2(w_bytes: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let num_blocks = x.len() / QK_K;
    let mut sum = 0.0f32;

    for b in 0..num_blocks {
        let src_offset = b * 34;
        let scale = f16_to_f32(u16::from_le_bytes([
            w_bytes[src_offset],
            w_bytes[src_offset + 1],
        ]));
        let x_ptr = x.as_ptr().add(b * QK_K);
        let q_ptr = w_bytes.as_ptr().add(src_offset + 2);

        let mut acc = _mm256_setzero_ps();

        for i in 0..4 {
            let i8_slice = std::slice::from_raw_parts(q_ptr.add(i * 8), 8);
            let mut i64_buf = [0u8; 8];
            i64_buf.copy_from_slice(i8_slice);
            let q_i64 = i64::from_ne_bytes(i64_buf);
            let q128 = _mm_cvtsi64_si128(q_i64);

            let q256_i32 = _mm256_cvtepi8_epi32(q128);
            let q256_f32 = _mm256_cvtepi32_ps(q256_i32);
            let x256_f32 = _mm256_loadu_ps(x_ptr.add(i * 8));

            acc = _mm256_fmadd_ps(q256_f32, x256_f32, acc);
        }

        let h128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let shuf = _mm_movehdup_ps(h128);
        let sums = _mm_add_ps(h128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let total = _mm_add_ss(sums, shuf2);
        let block_sum = _mm_cvtss_f32(total);

        sum += block_sum * scale;
    }

    sum
}

/// Computes the dot product between a Q4_0 quantized weight row and an FP32 vector directly without full dequantization.
pub fn vec_dot_q4_0(w_bytes: &[u8], x: &[f32]) -> f32 {
    let num_blocks = x.len() / QK_K;
    let mut sum = 0.0f32;

    for b in 0..num_blocks {
        let src_offset = b * 18;
        let scale = f16_to_f32(u16::from_le_bytes([
            w_bytes[src_offset],
            w_bytes[src_offset + 1],
        ]));
        let x_offset = b * QK_K;

        let mut block_acc = 0.0f32;
        for i in 0..16 {
            let byte = w_bytes[src_offset + 2 + i];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;

            block_acc += q0 as f32 * x[x_offset + i];
            block_acc += q1 as f32 * x[x_offset + i + 16];
        }
        sum += block_acc * scale;
    }

    sum
}

/// Quantizes an FP32 slice of 32 elements into a Q8_0 block (34 bytes).
pub fn quantize_q8_0_block(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len(), QK_K);
    assert_eq!(dst.len(), 34);

    let mut amax = 0.0f32;
    for &val in src {
        amax = amax.max(val.abs());
    }

    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let f16_scale = f32_to_f16(d);
    let scale_bytes = f16_scale.to_le_bytes();
    dst[0] = scale_bytes[0];
    dst[1] = scale_bytes[1];

    for i in 0..QK_K {
        let q = (src[i] * id).round().clamp(-128.0, 127.0) as i8;
        dst[2 + i] = q as u8;
    }
}

/// A memory-efficient quantized linear projection model (Q8_0 or Q4_0).
/// Reduces LM head weight memory by up to 75%–87% compared to FP32.
#[derive(Debug, Clone)]
pub struct QuantizedLinearModel {
    pub name: String,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub quant_type: QuantType,
    /// Raw quantized bytes matrix of shape [vocab_size, bytes_per_row].
    pub quantized_weights: Vec<u8>,
    pub bytes_per_row: usize,
    /// Unquantized or dequantized token embedding table.
    pub embeddings: Vec<f32>,
}

impl QuantizedLinearModel {
    /// Creates a Q8_0 quantized linear model from FP32 weights.
    pub fn from_f32_weights(
        name: impl Into<String>,
        vocab_size: usize,
        hidden_dim: usize,
        weights: &[f32],
        embeddings: Vec<f32>,
    ) -> Result<Self, EngineError> {
        if hidden_dim % QK_K != 0 {
            return Err(EngineError::InvalidConfig(format!(
                "Hidden dimension {} must be a multiple of {}",
                hidden_dim, QK_K
            )));
        }

        let num_blocks = hidden_dim / QK_K;
        let bytes_per_row = num_blocks * 34; // Q8_0
        let mut quantized_weights = vec![0u8; vocab_size * bytes_per_row];

        for row in 0..vocab_size {
            let row_offset_f32 = row * hidden_dim;
            let row_offset_bytes = row * bytes_per_row;

            for b in 0..num_blocks {
                let src_slice =
                    &weights[row_offset_f32 + b * QK_K..row_offset_f32 + (b + 1) * QK_K];
                let dst_slice = &mut quantized_weights
                    [row_offset_bytes + b * 34..row_offset_bytes + (b + 1) * 34];
                quantize_q8_0_block(src_slice, dst_slice);
            }
        }

        Ok(Self {
            name: name.into(),
            vocab_size,
            hidden_dim,
            quant_type: QuantType::Q8_0,
            quantized_weights,
            bytes_per_row,
            embeddings,
        })
    }

    /// Projects hidden states into logits using quantized dot products.
    pub fn project_hidden(&self, hidden: &[f32]) -> Logits {
        assert_eq!(hidden.len(), self.hidden_dim);
        let mut logits = Vec::with_capacity(self.vocab_size);

        for row in 0..self.vocab_size {
            let offset = row * self.bytes_per_row;
            let row_bytes = &self.quantized_weights[offset..offset + self.bytes_per_row];

            let dot = match self.quant_type {
                QuantType::Q8_0 => vec_dot_q8_0(row_bytes, hidden),
                QuantType::Q4_0 => vec_dot_q4_0(row_bytes, hidden),
            };
            logits.push(dot);
        }

        Logits::new(logits)
    }

    /// Computes logits directly for an input token ID.
    pub fn project_token(&self, token: TokenId) -> Logits {
        let token_idx = (token as usize) % self.vocab_size;
        let emb_offset = token_idx * self.hidden_dim;
        let hidden = &self.embeddings[emb_offset..emb_offset + self.hidden_dim];
        self.project_hidden(hidden)
    }
}

impl ModelEngine for QuantizedLinearModel {
    async fn forward_batch(&mut self, tokens: &[TokenId]) -> Result<Vec<Logits>, EngineError> {
        let mut out = Vec::with_capacity(tokens.len());
        for &tok in tokens {
            out.push(self.project_token(tok));
        }
        Ok(out)
    }

    async fn forward_token(&mut self, token: TokenId) -> Result<Logits, EngineError> {
        Ok(self.project_token(token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32_roundtrip() {
        let test_vals = [0.0f32, 1.0, -1.0, 0.5, 2.5, -15.25, 100.0];
        for &v in &test_vals {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!((v - back).abs() < 1e-2, "Failed for {v}: got {back}");
        }
    }

    #[test]
    fn test_q8_0_quantize_dequantize() {
        let mut original = [0.0f32; 32];
        for (i, val) in original.iter_mut().enumerate() {
            *val = (i as f32 - 16.0) * 0.25;
        }

        let mut block = [0u8; 34];
        quantize_q8_0_block(&original, &mut block);

        let mut dequantized = [0.0f32; 32];
        dequantize_q8_0(&block, &mut dequantized);

        for i in 0..32 {
            assert!(
                (original[i] - dequantized[i]).abs() < 0.05,
                "Diff too large at {i}: orig {}, got {}",
                original[i],
                dequantized[i]
            );
        }
    }

    #[test]
    fn test_quantized_linear_model() {
        let vocab_size = 16;
        let hidden_dim = 32;
        let size = vocab_size * hidden_dim;

        let weights: Vec<f32> = (0..size).map(|i| (i as f32 % 10.0) * 0.1).collect();
        let embeddings: Vec<f32> = (0..size).map(|i| ((i + 5) as f32 % 10.0) * 0.1).collect();

        let model = QuantizedLinearModel::from_f32_weights(
            "test-q8", vocab_size, hidden_dim, &weights, embeddings,
        )
        .expect("Failed to build quantized model");

        assert_eq!(model.bytes_per_row, 34);
        assert_eq!(model.quantized_weights.len(), 16 * 34);

        let logits = model.project_token(2);
        assert_eq!(logits.vocab_size(), 16);
    }

    #[test]
    fn test_vec_dot_q8_0_simd_equivalence() {
        let mut original = [0.0f32; 64];
        let mut x = [0.0f32; 64];
        for i in 0..64 {
            original[i] = (i as f32 - 32.0) * 0.15;
            x[i] = (i as f32 % 7.0) * 0.3;
        }

        let mut w_bytes = vec![0u8; 68]; // 2 blocks of 34 bytes
        quantize_q8_0_block(&original[0..32], &mut w_bytes[0..34]);
        quantize_q8_0_block(&original[32..64], &mut w_bytes[34..68]);

        let scalar_res = vec_dot_q8_0_scalar(&w_bytes, &x);
        let simd_res = vec_dot_q8_0(&w_bytes, &x);

        assert!(
            (scalar_res - simd_res).abs() < 1e-4,
            "SIMD dot product ({simd_res}) must match scalar dot product ({scalar_res})"
        );
    }
}
