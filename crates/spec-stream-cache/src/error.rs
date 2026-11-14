use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum CacheError {
    #[error(
        "Out of memory: No physical KV-cache blocks available in pool (capacity: {total_blocks})"
    )]
    OutOfMemory { total_blocks: usize },

    #[error("Block {0} not found in allocator")]
    BlockNotFound(usize),

    #[error(
        "Invalid sequence state: requested token index {index} exceeds sequence length {length}"
    )]
    IndexOutOfBounds { index: usize, length: usize },

    #[error("Rollback error: target length {target_len} exceeds current length {current_len}")]
    InvalidRollback {
        target_len: usize,
        current_len: usize,
    },

    #[error("Sequence {0} not registered in cache manager")]
    SequenceNotFound(u64),

    #[error("Serialization / Snapshot error: {0}")]
    SerializationError(String),
}
