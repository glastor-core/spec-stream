use spec_stream_cache::CacheError;
use spec_stream_core::CoreError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Core speculative verification error: {0}")]
    Core(#[from] CoreError),

    #[error("Paged KV-cache error: {0}")]
    Cache(#[from] CacheError),

    #[error("Model inference error: {0}")]
    Inference(String),

    #[error("Engine channel closed unexpectedly")]
    ChannelClosed,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}
