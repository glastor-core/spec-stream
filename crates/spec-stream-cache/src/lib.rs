//! # spec-stream-cache
//!
//! Paged KV-cache memory manager for Large Language Models.
//! Implements virtual-to-physical block tables, prefix caching (prompt sharing),
//! and O(1) rollbacks for speculative decoding branch pruning.

pub mod allocator;
pub mod block;
pub mod error;
pub mod eviction;
pub mod manager;
pub mod snapshot;
pub mod table;

pub use allocator::BlockAllocator;
pub use block::{PhysicalBlock, PhysicalBlockId, DEFAULT_BLOCK_SIZE};
pub use error::CacheError;
pub use eviction::{EvictionPolicy, LruEvictionPolicy};
pub use manager::PagedKvCacheManager;
pub use snapshot::{PrefixCacheSnapshot, PrefixSnapshotEntry};
pub use table::BlockTable;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_allocation_and_rollback() {
        let block_size = 4;
        let mut allocator = BlockAllocator::new(10, block_size);
        let mut table = BlockTable::new(block_size);

        assert_eq!(allocator.num_free_blocks(), 10);

        // Append 7 tokens -> should occupy 2 physical blocks (ceil(7/4) = 2)
        let tokens: Vec<u32> = (1..=7).collect();
        table.append_tokens(&tokens, &mut allocator).unwrap();

        assert_eq!(table.seq_len, 7);
        assert_eq!(table.physical_blocks.len(), 2);
        assert_eq!(allocator.num_free_blocks(), 8);

        // Speculative decoding rejected the last 4 tokens -> roll back to valid count = 3
        // 3 tokens fit inside 1 physical block (ceil(3/4) = 1)
        // The second physical block must be immediately returned to the free list!
        let discarded = table.rollback(3, &mut allocator).unwrap();
        assert_eq!(discarded, 4);
        assert_eq!(table.seq_len, 3);
        assert_eq!(table.physical_blocks.len(), 1);
        assert_eq!(allocator.num_free_blocks(), 9); // Block was properly recycled!

        // Free remaining sequence
        table.free_all(&mut allocator).unwrap();
        assert_eq!(allocator.num_free_blocks(), 10);
    }
}

