use paraoxidizer_core::error::{PoxError, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
};

pub const POXCAL_MAGIC: &[u8; 4] = b"PXCL";
pub const POXCAL_VERSION: u32 = 1;

/// Statistical summary of layer activations collected during calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerActivationStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub variance: f32,
    pub p99: f32,
    pub p99_9: f32,
    pub outlier_ratio: f32,
    pub num_samples: usize,
}

/// The .poxcal calibration artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoxCalArtifact {
    pub magic: String,
    pub version: u32,
    pub dataset_sha256: String,
    pub profile_name: String,
    pub sample_count: usize,
    pub sequence_length: usize,
    pub timestamp_utc: u64,
    pub layer_stats: HashMap<String, LayerActivationStats>,
}

impl PoxCalArtifact {
    pub fn new(
        dataset_sha256: String,
        profile_name: String,
        sample_count: usize,
        sequence_length: usize,
    ) -> Self {
        Self {
            magic: "PXCL".to_string(),
            version: POXCAL_VERSION,
            dataset_sha256,
            profile_name,
            sample_count,
            sequence_length,
            timestamp_utc: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            layer_stats: HashMap::new(),
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_vec_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(&json)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;
        let artifact: PoxCalArtifact = serde_json::from_slice(&contents)?;
        if artifact.magic != "PXCL" {
            return Err(PoxError::Calibration("Invalid .poxcal magic".into()));
        }
        Ok(artifact)
    }
}
