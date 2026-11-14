use crate::allocator::BlockAllocator;
use crate::block::DEFAULT_BLOCK_SIZE;
use crate::error::CacheError;
use crate::eviction::{EvictionPolicy, LruEvictionPolicy};
use crate::table::BlockTable;
use parking_lot::Mutex;
use spec_stream_core::TokenId;
use std::collections::HashMap;

/// Thread-safe high-level manager orchestrating KV-caches across multiple concurrent inference sequences.
#[derive(Debug)]
pub struct PagedKvCacheManager {
    pub(crate) allocator: Mutex<BlockAllocator>,
    tables: Mutex<HashMap<u64, BlockTable>>,
    eviction: Mutex<LruEvictionPolicy>,
    block_size: usize,
}

impl PagedKvCacheManager {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        Self {
            allocator: Mutex::new(BlockAllocator::new(total_blocks, block_size)),
            tables: Mutex::new(HashMap::new()),
            eviction: Mutex::new(LruEvictionPolicy::new()),
            block_size,
        }
    }

    pub fn with_defaults(total_blocks: usize) -> Self {
        Self::new(total_blocks, DEFAULT_BLOCK_SIZE)
    }

    /// Registers a new sequence/request with the cache manager.
    pub fn create_sequence(&self, seq_id: u64) {
        let mut tables = self.tables.lock();
        tables.insert(seq_id, BlockTable::new(self.block_size));
        self.eviction.lock().record_access(seq_id);
    }

    /// Appends tokens to an active sequence.
    pub fn append_tokens(&self, seq_id: u64, tokens: &[TokenId]) -> Result<(), CacheError> {
        let mut tables = self.tables.lock();
        let table = tables
            .get_mut(&seq_id)
            .ok_or(CacheError::SequenceNotFound(seq_id))?;

        let mut allocator = self.allocator.lock();
        table.append_tokens(tokens, &mut allocator)?;
        self.eviction.lock().record_access(seq_id);
        Ok(())
    }

    /// Executes an O(1) rollback on the sequence when speculative tokens are rejected.
    pub fn rollback_sequence(
        &self,
        seq_id: u64,
        valid_token_count: usize,
    ) -> Result<usize, CacheError> {
        let mut tables = self.tables.lock();
        let table = tables
            .get_mut(&seq_id)
            .ok_or(CacheError::SequenceNotFound(seq_id))?;

        let mut allocator = self.allocator.lock();
        let res = table.rollback(valid_token_count, &mut allocator);
        self.eviction.lock().record_access(seq_id);
        res
    }

    /// Releases all KV-cache memory held by a sequence when the request finishes.
    pub fn remove_sequence(&self, seq_id: u64) -> Result<(), CacheError> {
        self.eviction.lock().remove_sequence(seq_id);
        let mut tables = self.tables.lock();
        if let Some(mut table) = tables.remove(&seq_id) {
            let mut allocator = self.allocator.lock();
            table.free_all(&mut allocator)?;
        }
        Ok(())
    }

    /// Evicts the least recently used sequence to reclaim blocks under memory pressure.
    /// Returns the sequence ID of the evicted sequence, if any.
    pub fn evict_lru_sequence(&self) -> Result<Option<u64>, CacheError> {
        let candidate = self.eviction.lock().select_eviction_candidate();
        if let Some(seq_id) = candidate {
            self.remove_sequence(seq_id)?;
            Ok(Some(seq_id))
        } else {
            Ok(None)
        }
    }

    /// Defragments the free physical block list to optimize spatial memory cache locality.
    pub fn defragment(&self) {
        self.allocator.lock().defragment_free_list();
    }

    /// Current physical block pool utilization ratio [0.0 - 1.0].
    pub fn utilization_ratio(&self) -> f32 {
        self.allocator.lock().utilization_ratio()
    }

    /// Current free memory fragmentation ratio [0.0 - 1.0].
    pub fn fragmentation_ratio(&self) -> f32 {
        self.allocator.lock().fragmentation_ratio()
    }

    /// Clones and forks a sequence's KV-cache table (e.g. for speculative branch evaluation).
    pub fn fork_sequence(&self, src_id: u64, dest_id: u64) -> Result<(), CacheError> {
        let mut tables = self.tables.lock();
        let src_table = tables
            .get(&src_id)
            .ok_or(CacheError::SequenceNotFound(src_id))?
            .clone();

        let mut allocator = self.allocator.lock();
        // Increment ref count for shared blocks (Copy-on-Write)
        for &block_id in &src_table.physical_blocks {
            allocator.inc_ref(block_id)?;
        }

        tables.insert(dest_id, src_table);
        Ok(())
    }

    /// Returns the number of free physical blocks in the pool.
    pub fn free_blocks(&self) -> usize {
        self.allocator.lock().num_free_blocks()
    }

    /// Returns the number of allocated physical blocks.
    pub fn allocated_blocks(&self) -> usize {
        self.allocator.lock().num_allocated_blocks()
    }

    /// Returns the number of active prefix cache entries.
    pub fn prefix_entries(&self) -> usize {
        self.allocator.lock().num_prefix_entries()
    }
}

