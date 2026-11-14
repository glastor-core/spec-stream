use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum CoreError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty logits provided for sampling or verification")]
    EmptyLogits,

    #[error("Invalid temperature: {0} (must be strictly positive)")]
    InvalidTemperature(f32),

    #[error("Invalid top_p parameter: {0} (must be in range (0.0, 1.0])")]
    InvalidTopP(f32),

    #[error("Invalid min_p parameter: {0} (must be in range [0.0, 1.0])")]
    InvalidMinP(f32),

    #[error("Invalid repetition penalty: {0} (must be strictly positive)")]
    InvalidPenalty(f32),

    #[error("Invalid draft proposal length: expected at least 1 token")]
    EmptyDraftProposal,

    #[error("Tree structure error: {0}")]
    TreeError(String),
}

