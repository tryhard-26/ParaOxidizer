use serde::{Deserialize, Serialize};

/// Supported numerical data types in ParaOxidizer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit IEEE 754 floating point
    F32,
    /// 16-bit IEEE 754 half-precision floating point
    F16,
    /// 16-bit Brain floating point
    BF16,
    /// 8-bit signed integer (symmetric or asymmetric)
    I8,
    /// 4-bit packed integer (two 4-bit weights per byte)
    I4,
    /// 8-bit FP8 E4M3 format
    F8E4M3,
    /// 8-bit FP8 E5M2 format
    F8E5M2,
}

impl DType {
    pub fn bits(&self) -> usize {
        match self {
            DType::F32 => 32,
            DType::F16 | DType::BF16 => 16,
            DType::I8 | DType::F8E4M3 | DType::F8E5M2 => 8,
            DType::I4 => 4,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::I8 | DType::I4 | DType::F8E4M3 | DType::F8E5M2)
    }

    pub fn bytes_per_element(&self) -> f64 {
        self.bits() as f64 / 8.0
    }
}

/// Group size for block-wise / group-wise quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantGroupSize {
    None,
    G32,
    G64,
    G128,
    G256,
}

impl QuantGroupSize {
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            QuantGroupSize::None => None,
            QuantGroupSize::G32 => Some(32),
            QuantGroupSize::G64 => Some(64),
            QuantGroupSize::G128 => Some(128),
            QuantGroupSize::G256 => Some(256),
        }
    }

    pub fn from_usize(val: usize) -> Option<Self> {
        match val {
            0 => Some(QuantGroupSize::None),
            32 => Some(QuantGroupSize::G32),
            64 => Some(QuantGroupSize::G64),
            128 => Some(QuantGroupSize::G128),
            256 => Some(QuantGroupSize::G256),
            _ => None,
        }
    }
}

/// Dimensional shape of a tensor
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    pub fn numel(&self) -> usize {
        if self.0.is_empty() {
            0
        } else {
            self.0.iter().product()
        }
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            self.0
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Metadata descriptor for a tensor in a model
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
    pub group_size: QuantGroupSize,
    pub data_offset: u64,
    pub data_len: u64,
    pub scale_offset: u64,
    pub scale_len: u64,
    pub outlier_offset: u64,
    pub outlier_len: u64,
    pub sha256: String,
}

impl TensorMeta {
    pub fn total_bytes(&self) -> u64 {
        self.data_len + self.scale_len + self.outlier_len
    }
}
