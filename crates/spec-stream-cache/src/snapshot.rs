use crate::allocator::BlockAllocator;
use crate::error::CacheError;
use crate::manager::PagedKvCacheManager;
use serde::{Deserialize, Serialize};
use spec_stream_core::TokenId;

/// Represents a single warm prefix entry in a persistent snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixSnapshotEntry {
    /// 64-bit cumulative hash of this prefix block.
    pub hash: u64,
    /// Token IDs stored in the block.
    pub tokens: Vec<TokenId>,
}

/// A serialized snapshot of warm prefix cache blocks for instant cold-start loading across restarts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixCacheSnapshot {
    /// Size of each physical block in tokens.
    pub block_size: usize,
    /// Collection of cached prefix entries.
    pub entries: Vec<PrefixSnapshotEntry>,
}

impl PrefixCacheSnapshot {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            entries: Vec::new(),
        }
    }

    /// Serializes the snapshot to a JSON string.
    pub fn to_json(&self) -> Result<String, CacheError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| CacheError::SerializationError(e.to_string()))
    }

    /// Deserializes a snapshot from a JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, CacheError> {
        serde_json::from_str(json_str)
            .map_err(|e| CacheError::SerializationError(e.to_string()))
    }
}

impl BlockAllocator {
    /// Exports all active prefix cache entries into a snapshot.
    pub fn export_prefix_snapshot(&self) -> PrefixCacheSnapshot {
        let mut snapshot = PrefixCacheSnapshot::new(self.block_size);

        for (&hash, &block_id) in &self.prefix_cache {
            if let Some(block) = self.blocks.get(block_id) {
                snapshot.entries.push(PrefixSnapshotEntry {
                    hash,
                    tokens: block.tokens.clone(),
                });
            }
        }

        snapshot
    }

    /// Imports warm prefix entries from a snapshot into the allocator pool.
    /// Returns the number of prefix blocks successfully restored.
    pub fn import_prefix_snapshot(
        &mut self,
        snapshot: &PrefixCacheSnapshot,
    ) -> Result<usize, CacheError> {
        let mut imported = 0;

        for entry in &snapshot.entries {
            // Only import if not already present and we have free blocks
            if !self.prefix_cache.contains_key(&entry.hash) && !self.free_list.is_empty() {
                let block_id = self.allocate()?;
                {
                    let block = self.get_block_mut(block_id)?;
                    block.tokens = entry.tokens.clone();
                    block.num_tokens = entry.tokens.len();
                    block.prefix_hash = Some(entry.hash);
                    // Retain a baseline reference for persistent warm caching
                    block.ref_count = 1;
                }
                self.prefix_cache.insert(entry.hash, block_id);
                imported += 1;
            }
        }

        Ok(imported)
    }
}

impl PagedKvCacheManager {
    /// Exports current warm prefix cache to a snapshot.
    pub fn export_snapshot(&self) -> PrefixCacheSnapshot {
        self.allocator.lock().export_prefix_snapshot()
    }

    /// Imports a prefix cache snapshot into the active pool.
    pub fn import_snapshot(&self, snapshot: &PrefixCacheSnapshot) -> Result<usize, CacheError> {
        self.allocator.lock().import_prefix_snapshot(snapshot)
    }

    /// Saves the warm prefix cache to a JSON string.
    pub fn save_snapshot_json(&self) -> Result<String, CacheError> {
        self.export_snapshot().to_json()
    }

    /// Loads and warms the cache from a JSON snapshot string.
    pub fn load_snapshot_json(&self, json_str: &str) -> Result<usize, CacheError> {
        let snapshot = PrefixCacheSnapshot::from_json(json_str)?;
        self.import_snapshot(&snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_cache_snapshot_roundtrip() {
        let block_size = 4;
        let mut allocator = BlockAllocator::new(10, block_size);

        // Populate a block and register prefix
        let block_id = allocator.allocate().unwrap();
        let tokens = vec![11, 22, 33, 44];
        {
            let block = allocator.get_block_mut(block_id).unwrap();
            block.tokens = tokens.clone();
            block.num_tokens = 4;
        }
        let hash = 0xDEADBEEFCAFE;
        allocator.register_prefix(hash, block_id);

        // Export to JSON
        let snapshot = allocator.export_prefix_snapshot();
        assert_eq!(snapshot.entries.len(), 1);
        let json = snapshot.to_json().unwrap();

        // Create a new fresh allocator and import the snapshot
        let mut new_allocator = BlockAllocator::new(10, block_size);
        let loaded_snapshot = PrefixCacheSnapshot::from_json(&json).unwrap();
        let imported = new_allocator.import_prefix_snapshot(&loaded_snapshot).unwrap();

        assert_eq!(imported, 1);
        assert_eq!(new_allocator.num_prefix_entries(), 1);
        assert!(new_allocator.find_prefix(hash).is_some());
    }
}

