//! # Model Weights & Metadata Parsers
//!
//! Provides zero-copy / streaming parsers for GGUF (v2/v3) binary headers and
//! SafeTensors metadata, alongside lightweight weight loaders for neural model runtimes.

use crate::error::EngineError;
use crate::model::ModelEngine;
use spec_stream_core::{Logits, TokenId};
use std::collections::HashMap;
use std::io::{Read, Seek};

/// Magic identifier for GGUF files (`b"GGUF"`).
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// Value type in GGUF metadata key-value pairs.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// Metadata describing a single tensor stored inside a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
}

/// Parsed header and metadata of a GGUF model file.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: u64,
}

impl GgufMetadata {
    /// Parses a GGUF header from any reader with seek capabilities.
    pub fn parse<R: Read + Seek>(reader: &mut R) -> Result<Self, EngineError> {
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| EngineError::Inference(format!("Failed to read GGUF magic: {}", e)))?;

        if magic != GGUF_MAGIC {
            return Err(EngineError::Inference(format!(
                "Invalid GGUF magic: expected {:?}, got {:?}",
                GGUF_MAGIC, magic
            )));
        }

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader
            .read_exact(&mut buf4)
            .map_err(|e| EngineError::Inference(e.to_string()))?;
        let version = u32::from_le_bytes(buf4);
        if !(2..=3).contains(&version) {
            return Err(EngineError::Inference(format!(
                "Unsupported GGUF version: {}",
                version
            )));
        }

        reader
            .read_exact(&mut buf8)
            .map_err(|e| EngineError::Inference(e.to_string()))?;
        let tensor_count = u64::from_le_bytes(buf8);

        reader
            .read_exact(&mut buf8)
            .map_err(|e| EngineError::Inference(e.to_string()))?;
        let metadata_kv_count = u64::from_le_bytes(buf8);

        let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);

        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(reader)?;
            let val = read_gguf_value(reader)?;
            metadata.insert(key, val);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(reader)?;
            reader
                .read_exact(&mut buf4)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            let n_dims = u32::from_le_bytes(buf4);

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                reader
                    .read_exact(&mut buf8)
                    .map_err(|e| EngineError::Inference(e.to_string()))?;
                dims.push(u64::from_le_bytes(buf8));
            }

            reader
                .read_exact(&mut buf4)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            let tensor_type = u32::from_le_bytes(buf4);

            reader
                .read_exact(&mut buf8)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            let offset = u64::from_le_bytes(buf8);

            tensors.push(GgufTensorInfo {
                name,
                dimensions: dims,
                tensor_type,
                offset,
            });
        }

        let pos = reader
            .stream_position()
            .map_err(|e| EngineError::Inference(e.to_string()))?;
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::Uint32(a)) => *a as u64,
            Some(GgufValue::Uint64(a)) => *a,
            _ => 32,
        };
        let remainder = pos % alignment;
        let data_offset = if remainder == 0 {
            pos
        } else {
            pos + (alignment - remainder)
        };

        Ok(Self {
            version,
            tensor_count,
            metadata_kv_count,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Retrieves an optional string parameter from GGUF metadata.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Retrieves an optional u64 parameter from GGUF metadata.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(GgufValue::Uint64(v)) => Some(*v),
            Some(GgufValue::Uint32(v)) => Some(*v as u64),
            Some(GgufValue::Int64(v)) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String, EngineError> {
    let mut buf8 = [0u8; 8];
    reader
        .read_exact(&mut buf8)
        .map_err(|e| EngineError::Inference(e.to_string()))?;
    let len = u64::from_le_bytes(buf8) as usize;

    let mut str_bytes = vec![0u8; len];
    reader
        .read_exact(&mut str_bytes)
        .map_err(|e| EngineError::Inference(e.to_string()))?;

    String::from_utf8(str_bytes)
        .map_err(|e| EngineError::Inference(format!("GGUF invalid UTF-8 string: {}", e)))
}

fn read_gguf_value<R: Read>(reader: &mut R) -> Result<GgufValue, EngineError> {
    let mut buf4 = [0u8; 4];
    reader
        .read_exact(&mut buf4)
        .map_err(|e| EngineError::Inference(e.to_string()))?;
    let val_type = u32::from_le_bytes(buf4);

    read_gguf_value_typed(reader, val_type)
}

fn read_gguf_value_typed<R: Read>(reader: &mut R, val_type: u32) -> Result<GgufValue, EngineError> {
    let mut buf1 = [0u8; 1];
    let mut buf2 = [0u8; 2];
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    match val_type {
        0 => {
            reader
                .read_exact(&mut buf1)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Uint8(buf1[0]))
        }
        1 => {
            reader
                .read_exact(&mut buf1)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Int8(buf1[0] as i8))
        }
        2 => {
            reader
                .read_exact(&mut buf2)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Uint16(u16::from_le_bytes(buf2)))
        }
        3 => {
            reader
                .read_exact(&mut buf2)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Int16(i16::from_le_bytes(buf2)))
        }
        4 => {
            reader
                .read_exact(&mut buf4)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Uint32(u32::from_le_bytes(buf4)))
        }
        5 => {
            reader
                .read_exact(&mut buf4)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Int32(i32::from_le_bytes(buf4)))
        }
        6 => {
            reader
                .read_exact(&mut buf4)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Float32(f32::from_le_bytes(buf4)))
        }
        7 => {
            reader
                .read_exact(&mut buf1)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Bool(buf1[0] != 0))
        }
        8 => {
            let s = read_gguf_string(reader)?;
            Ok(GgufValue::String(s))
        }
        9 => {
            reader
                .read_exact(&mut buf4)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            let elem_type = u32::from_le_bytes(buf4);
            reader
                .read_exact(&mut buf8)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            let count = u64::from_le_bytes(buf8) as usize;

            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_gguf_value_typed(reader, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => {
            reader
                .read_exact(&mut buf8)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Uint64(u64::from_le_bytes(buf8)))
        }
        11 => {
            reader
                .read_exact(&mut buf8)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Int64(i64::from_le_bytes(buf8)))
        }
        12 => {
            reader
                .read_exact(&mut buf8)
                .map_err(|e| EngineError::Inference(e.to_string()))?;
            Ok(GgufValue::Float64(f64::from_le_bytes(buf8)))
        }
        other => Err(EngineError::Inference(format!(
            "Unsupported GGUF value type ID: {}",
            other
        ))),
    }
}

/// Metadata description for SafeTensors files.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SafeTensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Parsed SafeTensors header.
#[derive(Debug, Clone)]
pub struct SafeTensorsMetadata {
    pub tensors: HashMap<String, SafeTensorEntry>,
    pub data_offset: usize,
}

impl SafeTensorsMetadata {
    /// Parses SafeTensors header from bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() < 8 {
            return Err(EngineError::Inference(
                "SafeTensors buffer too short for 8-byte header length".into(),
            ));
        }

        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let data_offset = 8 + header_len;

        if bytes.len() < data_offset {
            return Err(EngineError::Inference(
                "SafeTensors header exceeds buffer length".into(),
            ));
        }

        let header_json = std::str::from_utf8(&bytes[8..data_offset]).map_err(|e| {
            EngineError::Inference(format!("SafeTensors header not valid UTF-8: {}", e))
        })?;

        let tensors: HashMap<String, SafeTensorEntry> = serde_json::from_str(header_json)
            .map_err(|e| EngineError::Inference(format!("SafeTensors JSON invalid: {}", e)))?;

        Ok(Self {
            tensors,
            data_offset,
        })
    }
}

/// A lightweight, real linear projection model engine for fast embeddings and head inference.
///
/// Multiplies an input token embedding against an output projection matrix `W` of shape `[vocab_size, hidden_dim]`.
/// Supports direct loading of weights for speculative drafting.
#[derive(Debug, Clone)]
pub struct LinearProjectionModel {
    pub name: String,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    /// Weights matrix of shape `[vocab_size, hidden_dim]` in row-major order.
    pub weights: Vec<f32>,
    /// Embedding lookup table `[vocab_size, hidden_dim]`.
    pub embeddings: Vec<f32>,
}

impl LinearProjectionModel {
    /// Creates a new projection model with initial weights.
    pub fn new(name: impl Into<String>, vocab_size: usize, hidden_dim: usize) -> Self {
        let size = vocab_size * hidden_dim;
        let mut weights = Vec::with_capacity(size);
        let mut embeddings = Vec::with_capacity(size);

        // Simple normalized initialization
        let inv = 1.0 / (hidden_dim as f32).sqrt();
        for i in 0..size {
            let val = ((i % 17) as f32 - 8.0) * inv * 0.1;
            weights.push(val);
            embeddings.push(val);
        }

        Self {
            name: name.into(),
            vocab_size,
            hidden_dim,
            weights,
            embeddings,
        }
    }

    /// Computes logits from hidden states via linear projection: `logits = hidden * W^T`.
    pub fn project_hidden(&self, hidden: &[f32]) -> Logits {
        assert_eq!(
            hidden.len(),
            self.hidden_dim,
            "Hidden dimension mismatch in projection"
        );
        let mut logits = Vec::with_capacity(self.vocab_size);

        for row in 0..self.vocab_size {
            let offset = row * self.hidden_dim;
            let mut dot = 0.0f32;
            for (col, &h) in hidden.iter().enumerate().take(self.hidden_dim) {
                dot += h * self.weights[offset + col];
            }
            logits.push(dot);
        }

        Logits::new(logits)
    }

    /// Computes logits directly for an input token using its embedded representation.
    pub fn project_token(&self, token: TokenId) -> Logits {
        let token_idx = (token as usize) % self.vocab_size;
        let emb_offset = token_idx * self.hidden_dim;
        let hidden = &self.embeddings[emb_offset..emb_offset + self.hidden_dim];
        self.project_hidden(hidden)
    }
}

impl ModelEngine for LinearProjectionModel {
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
    use std::io::Cursor;

    #[test]
    fn test_linear_projection_model_forward() {
        let model = LinearProjectionModel::new("test-linear", 16, 8);
        assert_eq!(model.vocab_size, 16);
        assert_eq!(model.hidden_dim, 8);

        let logits = model.project_token(3);
        assert_eq!(logits.vocab_size(), 16);
    }

    #[test]
    fn test_safetensors_metadata_parser() {
        let json_header =
            r#"{"lm_head.weight":{"dtype":"F32","shape":[16,8],"data_offsets":[0,512]}}"#;
        let mut buffer = Vec::new();
        let header_len = json_header.len() as u64;
        buffer.extend_from_slice(&header_len.to_le_bytes());
        buffer.extend_from_slice(json_header.as_bytes());
        buffer.resize(buffer.len() + 512, 0u8);

        let parsed = SafeTensorsMetadata::parse(&buffer).expect("Must parse SafeTensors");
        assert_eq!(parsed.tensors.len(), 1);
        let entry = &parsed.tensors["lm_head.weight"];
        assert_eq!(entry.dtype, "F32");
        assert_eq!(entry.shape, vec![16, 8]);
        assert_eq!(entry.data_offsets, [0, 512]);
    }

    #[test]
    fn test_gguf_metadata_parser_synthetic() {
        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(b"GGUF");
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count = 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // Metadata KV count = 2
        buf.extend_from_slice(&2u64.to_le_bytes());

        // KV 1: "general.architecture" -> String "llama"
        let k1 = "general.architecture";
        buf.extend_from_slice(&(k1.len() as u64).to_le_bytes());
        buf.extend_from_slice(k1.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // Type 8 = String
        let v1 = "llama";
        buf.extend_from_slice(&(v1.len() as u64).to_le_bytes());
        buf.extend_from_slice(v1.as_bytes());

        // KV 2: "llama.context_length" -> Uint32 4096
        let k2 = "llama.context_length";
        buf.extend_from_slice(&(k2.len() as u64).to_le_bytes());
        buf.extend_from_slice(k2.as_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // Type 4 = Uint32
        buf.extend_from_slice(&4096u32.to_le_bytes());

        // Tensor 1: "token_embd.weight", n_dims=2, dims=[4096, 32000], type=0, offset=0
        let t1 = "token_embd.weight";
        buf.extend_from_slice(&(t1.len() as u64).to_le_bytes());
        buf.extend_from_slice(t1.as_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
        buf.extend_from_slice(&4096u64.to_le_bytes());
        buf.extend_from_slice(&32000u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        let mut cursor = Cursor::new(buf);
        let meta = GgufMetadata::parse(&mut cursor).expect("GGUF parse failed");

        assert_eq!(meta.version, 3);
        assert_eq!(meta.tensor_count, 1);
        assert_eq!(meta.get_string("general.architecture"), Some("llama"));
        assert_eq!(meta.get_u64("llama.context_length"), Some(4096));
        assert_eq!(meta.tensors.len(), 1);
        assert_eq!(meta.tensors[0].name, "token_embd.weight");
        assert_eq!(meta.tensors[0].dimensions, vec![4096, 32000]);
    }
}
