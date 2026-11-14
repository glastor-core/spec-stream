use spec_stream_core::TokenId;

/// Unique physical block index in the memory pool.
pub type PhysicalBlockId = usize;

/// Size of a block in number of tokens (typically 16, 32, or 64).
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Represents a physical memory block storing KV-tensors for a fixed number of tokens.
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    pub id: PhysicalBlockId,
    pub block_size: usize,
    /// Reference count for Copy-on-Write and prefix sharing.
    pub ref_count: usize,
    /// Number of valid tokens currently stored in this block.
    pub num_tokens: usize,
    /// Cached hash of the token sequence prefix stored in this block (for prefix caching).
    pub prefix_hash: Option<u64>,
    /// Token IDs stored in this block (for tracking and prefix matching).
    pub tokens: Vec<TokenId>,
}

impl PhysicalBlock {
    pub fn new(id: PhysicalBlockId, block_size: usize) -> Self {
        Self {
            id,
            block_size,
            ref_count: 0,
            num_tokens: 0,
            prefix_hash: None,
            tokens: Vec::with_capacity(block_size),
        }
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.num_tokens >= self.block_size
    }

    #[inline]
    pub fn is_free(&self) -> bool {
        self.ref_count == 0
    }

    #[inline]
    pub fn reset(&mut self) {
        self.ref_count = 0;
        self.num_tokens = 0;
        self.prefix_hash = None;
        self.tokens.clear();
    }
}
