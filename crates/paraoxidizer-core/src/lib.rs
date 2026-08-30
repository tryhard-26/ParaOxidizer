//! Core types, hardware detection, error definitions, and configurations for ParaOxidizer.

pub mod arch;
pub mod config;
pub mod error;
pub mod hardware;
pub mod tensor;

pub use arch::{ComponentType, ModelArchitecture, ModelConfig};
pub use config::ParaOxidizerConfig;
pub use error::{PoxError, Result};
pub use hardware::HardwareInfo;
pub use tensor::{DType, QuantGroupSize, Shape, TensorMeta};
