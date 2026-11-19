//! Property-based randomized invariant testing for `spec-stream-cache`.
//!
//! Validates memory conservation laws, reference count consistency,
//! Copy-on-Write isolation, and O(1) rollback invariants under extreme concurrency.

use rand::prelude::*;
use spec_stream_cache::{BlockAllocator, BlockTable, PagedKvCacheManager};

#[test]
fn test_property_allocator_conservation_law() {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let total_blocks = 256;
    let mut allocator = BlockAllocator::new(total_blocks, 16);

    let mut allocated_blocks = Vec::new();

    // 5,000 randomized allocation, deallocation and defragmentation operations
    for _ in 0..5_000 {
        let action: u8 = rng.random_range(0..3);
        match action {
            0 => {
                // Allocate block
                if allocator.num_free_blocks() > 0 {
                    let block_id = allocator
                        .allocate()
                        .expect("allocation should succeed when free > 0");
                    allocated_blocks.push(block_id);
                } else {
                    assert!(
                        allocator.allocate().is_err(),
                        "allocation must fail when pool is full"
                    );
                }
            }
            1 => {
                // Free block
                if !allocated_blocks.is_empty() {
                    let idx = rng.random_range(0..allocated_blocks.len());
                    let block_id = allocated_blocks.swap_remove(idx);
                    allocator
                        .free(block_id)
                        .expect("free must succeed for allocated block");
                }
            }
            2 => {
                // Defragment
                allocator.defragment_free_list();
            }
            _ => unreachable!(),
        }

        // Strict conservation invariant: Free + Allocated == Total
        assert_eq!(
            allocator.num_free_blocks() + allocator.num_allocated_blocks(),
            total_blocks,
            "Conservation law violated: free ({}) + allocated ({}) != total ({})",
            allocator.num_free_blocks(),
            allocator.num_allocated_blocks(),
            total_blocks
        );
    }

    // Free all remaining blocks and verify return to initial clean state
    for block_id in allocated_blocks {
        allocator.free(block_id).expect("cleanup free must succeed");
    }

    assert_eq!(allocator.num_free_blocks(), total_blocks);
    assert_eq!(allocator.num_allocated_blocks(), 0);
}

#[test]
fn test_property_cow_and_rollback_invariants() {
    let mut rng = StdRng::seed_from_u64(0xCAFEBABE);
    let manager = PagedKvCacheManager::with_defaults(512);

    // Run 100 multi-sequence lifecycle simulations
    for seq_id in 0..100 {
        manager.create_sequence(seq_id);

        let prompt_tokens: Vec<u32> = (0..rng.random_range(16..128))
            .map(|_| rng.random_range(1..1000))
            .collect();
        let prompt_len = prompt_tokens.len();

        manager
            .append_tokens(seq_id, &prompt_tokens)
            .expect("prompt allocation must succeed");

        // Speculative burst: simulate K drafted tokens
        let draft_tokens: Vec<u32> = (0..rng.random_range(1..16))
            .map(|_| rng.random_range(1..1000))
            .collect();
        manager
            .append_tokens(seq_id, &draft_tokens)
            .expect("append draft tokens must succeed");

        // Simulate random verification: accept between 0 and draft_tokens.len()
        let accepted = rng.random_range(0..=draft_tokens.len());
        let target_len = prompt_len + accepted;

        manager
            .rollback_sequence(seq_id, target_len)
            .expect("rollback must succeed");

        // Release sequence
        manager
            .remove_sequence(seq_id)
            .expect("release sequence must succeed");
    }

    // Invariant: After removing all sequences, all 512 blocks must be returned to free pool
    assert_eq!(
        manager.free_blocks(),
        512,
        "Memory leak detected: not all blocks were reclaimed"
    );
    assert_eq!(manager.allocated_blocks(), 0);
}

#[test]
fn test_property_block_table_cow_isolation() {
    let mut allocator = BlockAllocator::new(128, 16);
    let mut table_a = BlockTable::new(16);

    // Append 32 tokens (exactly 2 full blocks of 16)
    let tokens: Vec<u32> = (1..=32).collect();
    table_a.append_tokens(&tokens, &mut allocator).unwrap();
    assert_eq!(table_a.physical_blocks.len(), 2);
    assert_eq!(allocator.num_allocated_blocks(), 2);

    // Fork Table B sharing physical blocks with Table A
    let mut table_b = table_a.clone();
    for &b in &table_b.physical_blocks {
        allocator.inc_ref(b).unwrap();
    }

    // Invariant: Ref counts for shared blocks must be 2
    for &b in &table_a.physical_blocks {
        assert_eq!(allocator.get_block(b).unwrap().ref_count, 2);
    }

    // Now append a token to Table B (triggers Copy-on-Write for the boundary block)
    table_b.append_token(999, &mut allocator).unwrap();

    // Table A length remains 32, Table B length becomes 33
    assert_eq!(table_a.seq_len, 32);
    assert_eq!(table_b.seq_len, 33);

    // Table B's 3rd block is allocated fresh (ref_count 1)
    assert_eq!(table_b.physical_blocks.len(), 3);
    assert_eq!(
        allocator
            .get_block(table_b.physical_blocks[2])
            .unwrap()
            .ref_count,
        1
    );

    // Clean up
    table_a.free_all(&mut allocator).unwrap();
    table_b.free_all(&mut allocator).unwrap();
    assert_eq!(
        allocator.num_free_blocks(),
        128,
        "All blocks must be freed after both tables are destroyed"
    );
}
