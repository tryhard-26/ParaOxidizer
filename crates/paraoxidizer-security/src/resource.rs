use paraoxidizer_core::error::{PoxError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_file_size_bytes: u64,
    pub max_tensor_count: usize,
    pub max_tensor_dimensions: usize,
    pub max_allocation_bytes: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            // 100 GB max file size
            max_file_size_bytes: 100 * 1024 * 1024 * 1024,
            max_tensor_count: 100_000,
            max_tensor_dimensions: 1_000_000,
            max_allocation_bytes: 64 * 1024 * 1024 * 1024,
        }
    }
}

impl ResourceLimits {
    pub fn validate_file_size(&self, size: u64) -> Result<()> {
        if size > self.max_file_size_bytes {
            return Err(PoxError::ResourceLimit(format!(
                "File size {} exceeds maximum quota {}",
                size, self.max_file_size_bytes
            )));
        }
        Ok(())
    }

    pub fn validate_tensor_count(&self, count: usize) -> Result<()> {
        if count > self.max_tensor_count {
            return Err(PoxError::ResourceLimit(format!(
                "Tensor count {} exceeds maximum allowed {}",
                count, self.max_tensor_count
            )));
        }
        Ok(())
    }

    pub fn validate_tensor_bounds(
        &self,
        offset: u64,
        len: u64,
        file_size: u64,
    ) -> Result<()> {
        let end = offset.checked_add(len).ok_or_else(|| {
            PoxError::Security("Arithmetic overflow calculating tensor boundaries".into())
        })?;

        if end > file_size {
            return Err(PoxError::Security(format!(
                "Tensor boundary [{}..{}] extends past file size {}",
                offset, end, file_size
            )));
        }
        Ok(())
    }
}
