use crate::allocator::{compute_block_prefix_hash, BlockAllocator};
use crate::block::PhysicalBlockId;
use crate::error::CacheError;
use spec_stream_core::TokenId;

/// Virtual-to-Physical block mapping table for a single sequence or request.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Ordered list of physical blocks allocated to this sequence.
    pub physical_blocks: Vec<PhysicalBlockId>,
    /// Prefix hash of each committed block in this sequence.
    pub prefix_hashes: Vec<u64>,
    /// Number of tokens currently registered in the sequence.
    pub seq_len: usize,
    /// Size of each block in tokens.
    pub block_size: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            physical_blocks: Vec::new(),
            prefix_hashes: Vec::new(),
            seq_len: 0,
            block_size,
        }
    }

    /// Appends a new token to this sequence's KV-cache, allocating a block if needed
    /// and performing Copy-on-Write if the target block is shared.
    pub fn append_token(
        &mut self,
        token_id: TokenId,
        allocator: &mut BlockAllocator,
    ) -> Result<(), CacheError> {
        // Check if we need to allocate a new physical block
        if self.seq_len % self.block_size == 0 {
            let block_id = allocator.allocate()?;
            self.physical_blocks.push(block_id);
        }

        let last_idx = self.physical_blocks.len() - 1;
        let current_block_id = self.physical_blocks[last_idx];

        // Copy-on-Write: If block is currently shared (ref_count > 1), clone it first
        let block_id_to_use = if allocator.get_block(current_block_id)?.ref_count > 1 {
            let new_block_id = allocator.clone_block(current_block_id)?;
            allocator.free(current_block_id)?;
            self.physical_blocks[last_idx] = new_block_id;
            new_block_id
        } else {
            current_block_id
        };

        let block = allocator.get_block_mut(block_id_to_use)?;
        block.tokens.push(token_id);
        block.num_tokens += 1;
        self.seq_len += 1;

        // If the block just became full, compute and register its prefix hash
        if block.is_full() {
            let parent_hash = self.prefix_hashes.last().copied();
            let hash = compute_block_prefix_hash(parent_hash, &block.tokens);
            self.prefix_hashes.push(hash);
            allocator.register_prefix(hash, block_id_to_use);
        }

        Ok(())
    }

    /// Appends a slice of tokens in batch with automatic prefix cache matching.
    pub fn append_tokens(
        &mut self,
        mut tokens: &[TokenId],
        allocator: &mut BlockAllocator,
    ) -> Result<(), CacheError> {
        // 1. Fill any partial trailing block first
        while !tokens.is_empty() && self.seq_len % self.block_size != 0 {
            self.append_token(tokens[0], allocator)?;
            tokens = &tokens[1..];
        }

        // 2. We are at a clean block boundary: attempt prefix cache lookups for full blocks
        while tokens.len() >= self.block_size {
            let chunk = &tokens[..self.block_size];
            let parent_hash = self.prefix_hashes.last().copied();
            let hash = compute_block_prefix_hash(parent_hash, chunk);

            if let Some(shared_block_id) = allocator.find_prefix(hash) {
                // Prefix Cache HIT! Reuse shared physical block without new allocation
                self.physical_blocks.push(shared_block_id);
                self.prefix_hashes.push(hash);
                self.seq_len += self.block_size;
            } else {
                // Prefix Cache MISS: allocate new physical block and register it
                let block_id = allocator.allocate()?;
                {
                    let block = allocator.get_block_mut(block_id)?;
                    block.tokens.extend_from_slice(chunk);
                    block.num_tokens = self.block_size;
                }
                allocator.register_prefix(hash, block_id);
                self.physical_blocks.push(block_id);
                self.prefix_hashes.push(hash);
                self.seq_len += self.block_size;
            }

            tokens = &tokens[self.block_size..];
        }

        // 3. Append any trailing remaining tokens (< block_size)
        for &t in tokens {
            self.append_token(t, allocator)?;
        }

        Ok(())
    }

    /// Speculative Decoding Rollback:
    /// Instantly rewinds the sequence to `valid_token_count` tokens.
    /// Trailing unneeded blocks are freed, and partial boundary blocks undergo
    /// Copy-on-Write if they were shared.
    pub fn rollback(
        &mut self,
        valid_token_count: usize,
        allocator: &mut BlockAllocator,
    ) -> Result<usize, CacheError> {
        if valid_token_count > self.seq_len {
            return Err(CacheError::InvalidRollback {
                target_len: valid_token_count,
                current_len: self.seq_len,
            });
        }

        let tokens_discarded = self.seq_len - valid_token_count;
        if tokens_discarded == 0 {
            return Ok(0);
        }

        // Calculate how many blocks should remain
        let required_blocks = valid_token_count.div_ceil(self.block_size);

        // Free trailing physical blocks that are no longer needed
        while self.physical_blocks.len() > required_blocks {
            let block_id = self.physical_blocks.pop().unwrap();
            allocator.free(block_id)?;
            if self.prefix_hashes.len() > self.physical_blocks.len() {
                self.prefix_hashes.pop();
            }
        }

        // Adjust the last block's token count if necessary
        if let Some(&last_block_id) = self.physical_blocks.last() {
            let tokens_in_last =
                if valid_token_count % self.block_size == 0 && valid_token_count > 0 {
                    self.block_size
                } else {
                    valid_token_count % self.block_size
                };

            // If last block was full and had registered prefix hash, but is now truncated:
            if tokens_in_last < self.block_size
                && self.prefix_hashes.len() == self.physical_blocks.len()
            {
                self.prefix_hashes.pop();
            }

            let last_idx = self.physical_blocks.len() - 1;
            // Copy-on-Write: If the partial block is shared, clone it before truncating
            let block_to_truncate = if allocator.get_block(last_block_id)?.ref_count > 1 {
                let new_block_id = allocator.clone_block(last_block_id)?;
                allocator.free(last_block_id)?;
                self.physical_blocks[last_idx] = new_block_id;
                new_block_id
            } else {
                last_block_id
            };

            let last_block = allocator.get_block_mut(block_to_truncate)?;
            last_block.num_tokens = tokens_in_last;
            last_block.tokens.truncate(tokens_in_last);
            last_block.prefix_hash = None;
        }

        self.seq_len = valid_token_count;
        Ok(tokens_discarded)
    }

    /// Translates a logical token position (0..seq_len) to its physical (block_id, offset) coordinates.
    pub fn translate_logical_offset(
        &self,
        logical_index: usize,
    ) -> Result<(PhysicalBlockId, usize), CacheError> {
        if logical_index >= self.seq_len {
            return Err(CacheError::IndexOutOfBounds {
                index: logical_index,
                length: self.seq_len,
            });
        }

        let block_idx = logical_index / self.block_size;
        let offset = logical_index % self.block_size;
        Ok((self.physical_blocks[block_idx], offset))
    }

    /// Cleans up and frees all physical blocks allocated to this sequence.
    pub fn free_all(&mut self, allocator: &mut BlockAllocator) -> Result<(), CacheError> {
        for &block_id in &self.physical_blocks {
            allocator.free(block_id)?;
        }
        self.physical_blocks.clear();
        self.prefix_hashes.clear();
        self.seq_len = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_caching_sharing() {
        let block_size = 4;
        let mut allocator = BlockAllocator::new(10, block_size);
        let mut table1 = BlockTable::new(block_size);
        let mut table2 = BlockTable::new(block_size);

        let system_prompt = vec![101, 102, 103, 104]; // exactly 1 block

        // First sequence populates the block
        table1
            .append_tokens(&system_prompt, &mut allocator)
            .unwrap();
        assert_eq!(allocator.num_free_blocks(), 9);
        assert_eq!(allocator.num_prefix_entries(), 1);

        // Second sequence with identical prompt should HIT prefix cache and allocate ZERO new blocks!
        table2
            .append_tokens(&system_prompt, &mut allocator)
            .unwrap();
        assert_eq!(allocator.num_free_blocks(), 9); // Still 9 free blocks!
        assert_eq!(table1.physical_blocks[0], table2.physical_blocks[0]); // Shared physical block!
        assert_eq!(
            allocator
                .get_block(table1.physical_blocks[0])
                .unwrap()
                .ref_count,
            2
        );

        // Freeing table1 leaves block alive for table2
        table1.free_all(&mut allocator).unwrap();
        assert_eq!(allocator.num_free_blocks(), 9);
        assert_eq!(
            allocator
                .get_block(table2.physical_blocks[0])
                .unwrap()
                .ref_count,
            1
        );

        table2.free_all(&mut allocator).unwrap();
        assert_eq!(allocator.num_free_blocks(), 10);
    }

    #[test]
    fn test_copy_on_write() {
        let block_size = 4;
        let mut allocator = BlockAllocator::new(10, block_size);
        let mut table1 = BlockTable::new(block_size);
        let mut table2 = BlockTable::new(block_size);

        let prompt = vec![1, 2, 3, 4];
        table1.append_tokens(&prompt, &mut allocator).unwrap();
        table2.append_tokens(&prompt, &mut allocator).unwrap();

        // Both share block 0
        let shared_id = table1.physical_blocks[0];
        assert_eq!(table2.physical_blocks[0], shared_id);

        // Now table2 appends a new token -> it will allocate block 1 for itself
        table2.append_token(99, &mut allocator).unwrap();
        assert_eq!(table2.seq_len, 5);
        assert_eq!(table1.seq_len, 4);

        // table2 rolls back to 3 tokens (which was inside the shared block)
        table2.rollback(3, &mut allocator).unwrap();
        // table2 should have performed Copy-on-Write on block 0!
        assert_ne!(table1.physical_blocks[0], table2.physical_blocks[0]);
        // Table 1 must still have its original 4 tokens
        let block1 = allocator.get_block(table1.physical_blocks[0]).unwrap();
        assert_eq!(block1.num_tokens, 4);
        assert_eq!(block1.tokens, vec![1, 2, 3, 4]);

        // Table 2 must have only 3 tokens
        let block2 = allocator.get_block(table2.physical_blocks[0]).unwrap();
        assert_eq!(block2.num_tokens, 3);
        assert_eq!(block2.tokens, vec![1, 2, 3]);
    }
}
