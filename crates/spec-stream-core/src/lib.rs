//! # spec-stream-core
//!
//! Core primitives, data structures, and mathematical verification algorithms for
//! speculative decoding in high-throughput Large Language Model (LLM) inference.
//!
//! Designed to be model-agnostic, zero-allocation where possible, and strictly compliant
//! with exact output probability distributions (Leviathan et al., 2023).

pub mod error;
pub mod logits;
pub mod metrics;
pub mod tokenizer;
pub mod tree;
pub mod types;
pub mod verifier;

pub use error::CoreError;
pub use logits::{Logits, TokenId};
pub use metrics::SpeculativeMetrics;
pub use tokenizer::{ByteLevelTokenizer, Tokenizer};
pub use tree::{SpeculativeTree, TreeNode, TreeVerificationResult};
pub use types::{DraftProposal, SamplingParams, TokenCandidate, VerificationResult, VerificationStrategy};
pub use verifier::SpeculativeVerifier;

