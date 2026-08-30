use thiserror::Error;

pub type Result<T> = std::result::Result<T, PoxError>;

#[derive(Error, Debug)]
pub enum PoxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Format error: {0}")]
    Format(String),

    #[error("Invalid magic bytes: expected {expected:?}, found {found:?}")]
    InvalidMagic { expected: [u8; 4], found: [u8; 4] },

    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Optimizer error: {0}")]
    Optimizer(String),

    #[error("Calibration error: {0}")]
    Calibration(String),

    #[error("Security violation: {0}")]
    Security(String),

    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),

    #[error("Cryptographic signature verification failed: {0}")]
    SignatureVerificationFailed(String),

    #[error("Integrity hash mismatch for tensor '{tensor}': expected {expected}, calculated {calculated}")]
    IntegrityHashMismatch {
        tensor: String,
        expected: String,
        calculated: String,
    },

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Model architecture mismatch: {0}")]
    Architecture(String),

    #[error("Config error: {0}")]
    Config(String),
}
