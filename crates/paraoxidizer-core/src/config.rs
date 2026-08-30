use serde::{Deserialize, Serialize};

/// Root configuration file representation (paraoxidizer.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParaOxidizerConfig {
    pub model: ModelSection,
    #[serde(default)]
    pub calibration: Option<CalibrationSection>,
    #[serde(default)]
    pub quantization: Option<QuantizationSection>,
    #[serde(default)]
    pub optimization: Option<OptimizationSection>,
    #[serde(default)]
    pub security: Option<SecuritySection>,
    #[serde(default)]
    pub runtime: Option<RuntimeSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSection {
    pub source: String,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSection {
    pub dataset: String,
    #[serde(default = "default_samples")]
    pub samples: usize,
    #[serde(default = "default_seq_len")]
    pub sequence_length: usize,
    #[serde(default)]
    pub profile: Option<String>,
}

fn default_samples() -> usize {
    256
}

fn default_seq_len() -> usize {
    512
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationSection {
    #[serde(default = "default_algorithm")]
    pub algorithm: String,
    #[serde(default = "default_allowed_precisions")]
    pub allowed_precisions: Vec<String>,
    #[serde(default = "default_group_sizes")]
    pub group_sizes: Vec<usize>,
    #[serde(default = "default_outlier_policy")]
    pub outlier_policy: String,
}

fn default_algorithm() -> String {
    "min-max".to_string()
}

fn default_allowed_precisions() -> Vec<String> {
    vec!["INT4".to_string(), "INT8".to_string(), "FP16".to_string()]
}

fn default_group_sizes() -> Vec<usize> {
    vec![32, 64, 128, 256]
}

fn default_outlier_policy() -> String {
    "automatic".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSection {
    #[serde(default)]
    pub memory_limit: Option<String>,
    #[serde(default)]
    pub latency_limit: Option<String>,
    #[serde(default)]
    pub quality_floor: Option<f64>,
    #[serde(default = "default_target_hardware")]
    pub target_hardware: String,
}

fn default_target_hardware() -> String {
    "auto".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySection {
    #[serde(default = "default_true")]
    pub verify_source: bool,
    #[serde(default)]
    pub require_signature: bool,
    #[serde(default)]
    pub signature_pubkey: Option<String>,
    #[serde(default)]
    pub memory_limit: Option<String>,
    #[serde(default = "default_true")]
    pub artifact_integrity: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSection {
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_context_length")]
    pub context: usize,
    #[serde(default = "default_batch_size")]
    pub batching: usize,
}

fn default_backend() -> String {
    "auto".to_string()
}

fn default_context_length() -> usize {
    2048
}

fn default_batch_size() -> usize {
    1
}
