use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use half::f16;
use paraoxidizer_core::error::Result;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

/// Policy for detecting and preserving parameter outliers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutlierPolicy {
    Disabled,
    Conservative,
    Automatic,
    Aggressive,
}

impl OutlierPolicy {
    pub fn from_str_name(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "disabled" | "none" => OutlierPolicy::Disabled,
            "conservative" => OutlierPolicy::Conservative,
            "aggressive" => OutlierPolicy::Aggressive,
            _ => OutlierPolicy::Automatic,
        }
    }

    /// Sigma multiplier threshold for considering a value an outlier
    pub fn sigma_threshold(&self) -> Option<f32> {
        match self {
            OutlierPolicy::Disabled => None,
            OutlierPolicy::Conservative => Some(4.5),
            OutlierPolicy::Automatic => Some(3.5),
            OutlierPolicy::Aggressive => Some(2.8),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SparseOutlierTable {
    pub indices: Vec<u32>,
    pub values: Vec<f16>,
}

impl SparseOutlierTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Extract outliers from a mutable slice of f32 weights, replacing them with 0.0 in-place
    pub fn extract_and_zero_outliers(weights: &mut [f32], policy: OutlierPolicy) -> Option<Self> {
        let sigma_mult = policy.sigma_threshold()?;
        if weights.is_empty() {
            return None;
        }

        // Compute mean and standard deviation
        let sum: f32 = weights.iter().sum();
        let mean = sum / weights.len() as f32;
        let var_sum: f32 = weights.iter().map(|&w| (w - mean).powi(2)).sum();
        let std_dev = (var_sum / weights.len() as f32).sqrt();

        let threshold = sigma_mult * std_dev;
        let mut table = SparseOutlierTable::new();

        for (i, w) in weights.iter_mut().enumerate() {
            if (*w - mean).abs() > threshold {
                table.indices.push(i as u32);
                table.values.push(f16::from_f32(*w));
                *w = 0.0; // Zero out so INT4 range isn't inflated
            }
        }

        if table.is_empty() {
            None
        } else {
            Some(table)
        }
    }

    /// Restore outliers on top of a dequantized float vector
    pub fn apply_to(&self, dequantized: &mut [f32]) {
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            let i = idx as usize;
            if i < dequantized.len() {
                dequantized[i] = val.to_f32();
            }
        }
    }

    /// Serialize to binary byte stream:
    /// count (u32), indices (count * u32), values (count * u16)
    pub fn to_bytes(&self) -> Vec<u8> {
        let count = self.indices.len() as u32;
        let mut buf = Vec::with_capacity(4 + (count as usize) * 6);
        buf.write_u32::<LittleEndian>(count).unwrap();
        for &idx in &self.indices {
            buf.write_u32::<LittleEndian>(idx).unwrap();
        }
        for &val in &self.values {
            buf.write_u16::<LittleEndian>(val.to_bits()).unwrap();
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Ok(Self::new());
        }
        let mut cursor = Cursor::new(bytes);
        let count = cursor.read_u32::<LittleEndian>()? as usize;
        let mut indices = Vec::with_capacity(count);
        for _ in 0..count {
            indices.push(cursor.read_u32::<LittleEndian>()?);
        }
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(f16::from_bits(cursor.read_u16::<LittleEndian>()?));
        }
        Ok(Self { indices, values })
    }
}
