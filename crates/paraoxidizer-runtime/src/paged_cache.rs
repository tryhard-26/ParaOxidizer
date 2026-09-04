use paraoxidizer_core::error::{PoxError, Result};
use std::sync::{Arc, Mutex};

pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// A single fixed-size physical memory block holding KV activations for multiple tokens
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    pub block_id: usize,
    pub block_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    // Storage flattened: [block_token_idx, layer, head, dim]
    pub k_data: Vec<f32>,
    pub v_data: Vec<f32>,
}

impl PhysicalBlock {
    pub fn new(
        block_id: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let total_elements = block_size * num_layers * num_kv_heads * head_dim;
        Self {
            block_id,
            block_size,
            num_layers,
            num_kv_heads,
            head_dim,
            k_data: vec![0.0f32; total_elements],
            v_data: vec![0.0f32; total_elements],
        }
    }

    #[inline]
    fn get_offset(&self, token_idx: usize, layer: usize, head: usize) -> usize {
        let per_token = self.num_layers * self.num_kv_heads * self.head_dim;
        let per_layer = self.num_kv_heads * self.head_dim;
        (token_idx * per_token) + (layer * per_layer) + (head * self.head_dim)
    }

    pub fn write_kv(
        &mut self,
        token_idx: usize,
        layer: usize,
        head: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        if token_idx >= self.block_size {
            return Err(PoxError::Runtime(
                "Token index exceeds block capacity".into(),
            ));
        }
        let offset = self.get_offset(token_idx, layer, head);
        let len = k.len().min(self.head_dim);
        self.k_data[offset..offset + len].copy_from_slice(&k[..len]);
        self.v_data[offset..offset + len].copy_from_slice(&v[..len]);
        Ok(())
    }
}

/// Global allocator and manager of physical memory blocks for PagedAttention
#[derive(Debug)]
pub struct BlockManager {
    pub block_size: usize,
    pub total_blocks: usize,
    pub free_block_ids: Vec<usize>,
    pub blocks: Vec<PhysicalBlock>,
}

impl BlockManager {
    pub fn new(
        total_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let mut blocks = Vec::with_capacity(total_blocks);
        let mut free_block_ids = Vec::with_capacity(total_blocks);
        for id in 0..total_blocks {
            blocks.push(PhysicalBlock::new(
                id,
                block_size,
                num_layers,
                num_kv_heads,
                head_dim,
            ));
            free_block_ids.push(id);
        }
        Self {
            block_size,
            total_blocks,
            free_block_ids,
            blocks,
        }
    }

    pub fn allocate_block(&mut self) -> Result<usize> {
        self.free_block_ids.pop().ok_or_else(|| {
            PoxError::Runtime(
                "PagedAttention Out of Memory: No free physical blocks available".into(),
            )
        })
    }

    pub fn free_block(&mut self, block_id: usize) {
        if block_id < self.total_blocks && !self.free_block_ids.contains(&block_id) {
            self.free_block_ids.push(block_id);
        }
    }

    pub fn available_blocks(&self) -> usize {
        self.free_block_ids.len()
    }

    pub fn memory_usage_bytes(&self) -> usize {
        let total_elems: usize = self
            .blocks
            .iter()
            .map(|b| b.k_data.len() + b.v_data.len())
            .sum();
        total_elems * 4
    }
}

/// Virtual sequence block table mapping sequence tokens to non-contiguous physical blocks
#[derive(Debug, Clone)]
pub struct BlockTable {
    pub block_size: usize,
    pub allocated_block_ids: Vec<usize>,
    pub sequence_len: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            allocated_block_ids: Vec::new(),
            sequence_len: 0,
        }
    }

    pub fn append_token(&mut self, manager: &mut BlockManager) -> Result<(usize, usize)> {
        let pos = self.sequence_len;
        let block_idx = pos / self.block_size;
        let token_offset = pos % self.block_size;

        if block_idx >= self.allocated_block_ids.len() {
            let new_block_id = manager.allocate_block()?;
            self.allocated_block_ids.push(new_block_id);
        }

        self.sequence_len += 1;
        let phys_block_id = self.allocated_block_ids[block_idx];
        Ok((phys_block_id, token_offset))
    }

    pub fn free(&mut self, manager: &mut BlockManager) {
        for &id in &self.allocated_block_ids {
            manager.free_block(id);
        }
        self.allocated_block_ids.clear();
        self.sequence_len = 0;
    }
}

/// Thread-safe shared PagedAttention KV-Cache system
#[derive(Clone)]
pub struct PagedKvCache {
    pub manager: Arc<Mutex<BlockManager>>,
}

impl PagedKvCache {
    pub fn new(
        total_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let manager =
            BlockManager::new(total_blocks, block_size, num_layers, num_kv_heads, head_dim);
        Self {
            manager: Arc::new(Mutex::new(manager)),
        }
    }

    pub fn new_sequence(&self) -> BlockTable {
        let manager = self.manager.lock().unwrap();
        BlockTable::new(manager.block_size)
    }

    pub fn append(
        &self,
        table: &mut BlockTable,
        layer: usize,
        head: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        let mut manager = self.manager.lock().unwrap();
        let (phys_id, token_offset) = table.append_token(&mut manager)?;
        manager.blocks[phys_id].write_kv(token_offset, layer, head, k, v)
    }

    pub fn free_sequence(&self, table: &mut BlockTable) {
        let mut manager = self.manager.lock().unwrap();
        table.free(&mut manager);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_kv_cache_allocation_and_free() {
        let cache = PagedKvCache::new(4, 16, 2, 2, 32);
        let mut seq1 = cache.new_sequence();

        // Append 20 tokens -> requires 2 blocks (16 + 4)
        for i in 0..20 {
            let k = vec![i as f32; 32];
            let v = vec![i as f32 * 2.0; 32];
            cache.append(&mut seq1, 0, 0, &k, &v).unwrap();
        }

        assert_eq!(seq1.allocated_block_ids.len(), 2);
        assert_eq!(seq1.sequence_len, 20);

        {
            let manager = cache.manager.lock().unwrap();
            assert_eq!(manager.available_blocks(), 2); // 4 total - 2 used
        }

        // Free sequence
        cache.free_sequence(&mut seq1);
        assert_eq!(seq1.sequence_len, 0);

        {
            let manager = cache.manager.lock().unwrap();
            assert_eq!(manager.available_blocks(), 4); // All 4 returned
        }
    }
}
