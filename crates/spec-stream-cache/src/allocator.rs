use crate::block::{PhysicalBlock, PhysicalBlockId, DEFAULT_BLOCK_SIZE};
use crate::error::CacheError;
use spec_stream_core::TokenId;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Computes an incremental hash over a slice of tokens given an optional parent block hash.
pub fn compute_block_prefix_hash(parent_hash: Option<u64>, tokens: &[TokenId]) -> u64 {
    let mut hasher = DefaultHasher::new();
    parent_hash.hash(&mut hasher);
    tokens.hash(&mut hasher);
    hasher.finish()
}

/// Central allocator managing the physical pool of KV-cache blocks.
#[derive(Debug)]
pub struct BlockAllocator {
    pub block_size: usize,
    pub total_blocks: usize,
    pub(crate) blocks: Vec<PhysicalBlock>,
    pub(crate) free_list: Vec<PhysicalBlockId>,
    /// Prefix cache: maps a 64-bit sequence hash to an existing shared physical block.
    pub(crate) prefix_cache: HashMap<u64, PhysicalBlockId>,
}

impl BlockAllocator {
    /// Initializes a pool with `total_blocks` of capacity `block_size` tokens each.
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        let mut blocks = Vec::with_capacity(total_blocks);
        let mut free_list = Vec::with_capacity(total_blocks);

        for id in 0..total_blocks {
            blocks.push(PhysicalBlock::new(id, block_size));
            free_list.push(id);
        }

        Self {
            block_size,
            total_blocks,
            blocks,
            free_list,
            prefix_cache: HashMap::new(),
        }
    }

    pub fn with_default_block_size(total_blocks: usize) -> Self {
        Self::new(total_blocks, DEFAULT_BLOCK_SIZE)
    }

    /// Number of physical blocks currently available for allocation.
    #[inline]
    pub fn num_free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Number of physical blocks currently in use.
    #[inline]
    pub fn num_allocated_blocks(&self) -> usize {
        self.total_blocks - self.free_list.len()
    }

    /// Number of distinct prefix entries cached.
    #[inline]
    pub fn num_prefix_entries(&self) -> usize {
        self.prefix_cache.len()
    }

    /// Allocates an empty physical block from the free list.
    pub fn allocate(&mut self) -> Result<PhysicalBlockId, CacheError> {
        let block_id = self.free_list.pop().ok_or(CacheError::OutOfMemory {
            total_blocks: self.total_blocks,
        })?;

        let block = &mut self.blocks[block_id];
        block.ref_count = 1;
        block.num_tokens = 0;
        block.tokens.clear();
        block.prefix_hash = None;

        Ok(block_id)
    }

    /// Duplicates an existing physical block into a new block with ref_count = 1 (Copy-on-Write).
    pub fn clone_block(
        &mut self,
        src_block_id: PhysicalBlockId,
    ) -> Result<PhysicalBlockId, CacheError> {
        let src_block = self
            .blocks
            .get(src_block_id)
            .ok_or(CacheError::BlockNotFound(src_block_id))?
            .clone();

        let new_block_id = self.allocate()?;
        let dest = &mut self.blocks[new_block_id];
        dest.num_tokens = src_block.num_tokens;
        dest.tokens = src_block.tokens.clone();
        dest.prefix_hash = None; // Reset hash on new uncommitted CoW copy
        dest.ref_count = 1;

        Ok(new_block_id)
    }

    /// Increments ref_count for sharing an existing block (e.g. prompt prefix).
    pub fn inc_ref(&mut self, block_id: PhysicalBlockId) -> Result<(), CacheError> {
        let block = self
            .blocks
            .get_mut(block_id)
            .ok_or(CacheError::BlockNotFound(block_id))?;
        block.ref_count += 1;
        Ok(())
    }

    /// Decrements ref_count and frees the block if ref_count hits zero.
    pub fn free(&mut self, block_id: PhysicalBlockId) -> Result<(), CacheError> {
        let block = self
            .blocks
            .get_mut(block_id)
            .ok_or(CacheError::BlockNotFound(block_id))?;

        if block.ref_count > 0 {
            block.ref_count -= 1;
            if block.ref_count == 0 {
                if let Some(hash) = block.prefix_hash {
                    self.prefix_cache.remove(&hash);
                }
                block.reset();
                self.free_list.push(block_id);
            }
        }

        Ok(())
    }

    /// Registers a full block into the prefix cache for reuse.
    pub fn register_prefix(&mut self, hash: u64, block_id: PhysicalBlockId) {
        if let Some(block) = self.blocks.get_mut(block_id) {
            block.prefix_hash = Some(hash);
            self.prefix_cache.insert(hash, block_id);
        }
    }

    /// Searches for an existing shared block by prefix hash.
    pub fn find_prefix(&mut self, hash: u64) -> Option<PhysicalBlockId> {
        if let Some(&block_id) = self.prefix_cache.get(&hash) {
            if let Some(block) = self.blocks.get_mut(block_id) {
                block.ref_count += 1;
                return Some(block_id);
            }
        }
        None
    }

    /// Returns a reference to a physical block.
    pub fn get_block(&self, block_id: PhysicalBlockId) -> Result<&PhysicalBlock, CacheError> {
        self.blocks
            .get(block_id)
            .ok_or(CacheError::BlockNotFound(block_id))
    }

    /// Returns a mutable reference to a physical block.
    pub fn get_block_mut(
        &mut self,
        block_id: PhysicalBlockId,
    ) -> Result<&mut PhysicalBlock, CacheError> {
        self.blocks
            .get_mut(block_id)
            .ok_or(CacheError::BlockNotFound(block_id))
    }

    /// Defragments the free list by sorting block IDs so future allocations favor lowest contiguous physical pages.
    pub fn defragment_free_list(&mut self) {
        // Sort descending so pop() returns lowest block index first (0, 1, 2...)
        self.free_list.sort_unstable_by(|a, b| b.cmp(a));
    }

    /// Computes pool utilization ratio between 0.0 (empty) and 1.0 (fully saturated).
    #[inline]
    pub fn utilization_ratio(&self) -> f32 {
        self.num_allocated_blocks() as f32 / self.total_blocks.max(1) as f32
    }

    /// Computes memory fragmentation ratio of the free block list.
    /// Returns 0.0 when free blocks form a single contiguous block range, higher when scattered.
    pub fn fragmentation_ratio(&self) -> f32 {
        if self.free_list.len() <= 1 {
            return 0.0;
        }

        let mut sorted = self.free_list.clone();
        sorted.sort_unstable();

        let mut gaps = 0;
        for window in sorted.windows(2) {
            if window[1] != window[0] + 1 {
                gaps += 1;
            }
        }

        gaps as f32 / (self.free_list.len() - 1) as f32
    }
}
