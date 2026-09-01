use paraoxidizer_core::arch::ComponentType;
use paraoxidizer_format::poxcal::PoxCalArtifact;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sensitivity category of a model parameter or layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SensitivityLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for SensitivityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SensitivityLevel::Low => write!(f, "LOW"),
            SensitivityLevel::Medium => write!(f, "MEDIUM"),
            SensitivityLevel::High => write!(f, "HIGH"),
            SensitivityLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Sensitivity assessment for a single tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSensitivity {
    pub name: String,
    pub component: String,
    pub score: f32,
    pub level: SensitivityLevel,
    pub recommended_precision: String,
    pub recommended_group_size: usize,
}

/// Overall model sensitivity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSensitivityReport {
    pub tensors: Vec<TensorSensitivity>,
    pub summary_counts: HashMap<String, usize>,
}

pub struct SensitivityEngine;

impl SensitivityEngine {
    pub fn analyze(
        tensor_names: &[String],
        calibration: Option<&PoxCalArtifact>,
    ) -> ModelSensitivityReport {
        let mut results = Vec::new();
        let mut counts = HashMap::new();
        counts.insert("CRITICAL".into(), 0);
        counts.insert("HIGH".into(), 0);
        counts.insert("MEDIUM".into(), 0);
        counts.insert("LOW".into(), 0);

        for name in tensor_names {
            let comp = ComponentType::classify_tensor_name(name);
            let cal_factor = if let Some(cal) = calibration {
                if let Some(stats) = cal.layer_stats.get(name) {
                    1.0 + (stats.variance * 2.0) + (stats.outlier_ratio * 20.0)
                } else {
                    1.0
                }
            } else {
                1.0
            };

            let (base_score, _level, rec_prec, rec_group) = match comp {
                ComponentType::Norm => (9.5, SensitivityLevel::Critical, "FP16", 0),
                ComponentType::LmHead => (8.5, SensitivityLevel::Critical, "INT8", 0),
                ComponentType::Embedding => (7.8, SensitivityLevel::High, "INT8", 0),
                ComponentType::AttentionK | ComponentType::AttentionV => {
                    (6.8, SensitivityLevel::High, "INT8", 64)
                }
                ComponentType::MlpDown => (6.2, SensitivityLevel::High, "INT4", 64),
                ComponentType::AttentionOutput | ComponentType::MlpGate => {
                    (4.5, SensitivityLevel::Medium, "INT4", 128)
                }
                ComponentType::AttentionQ | ComponentType::MlpUp => {
                    (2.5, SensitivityLevel::Low, "INT4", 128)
                }
                ComponentType::Other => (3.0, SensitivityLevel::Medium, "INT4", 128),
            };

            let final_score = (base_score * cal_factor).min(10.0);

            // Re-evaluate level if calibration pushed it higher
            let effective_level = if final_score >= 8.5 {
                SensitivityLevel::Critical
            } else if final_score >= 6.0 {
                SensitivityLevel::High
            } else if final_score >= 3.5 {
                SensitivityLevel::Medium
            } else {
                SensitivityLevel::Low
            };

            let level_str = effective_level.to_string();
            *counts.entry(level_str).or_insert(0) += 1;

            results.push(TensorSensitivity {
                name: name.clone(),
                component: format!("{:?}", comp),
                score: final_score,
                level: effective_level,
                recommended_precision: rec_prec.to_string(),
                recommended_group_size: rec_group,
            });
        }

        ModelSensitivityReport {
            tensors: results,
            summary_counts: counts,
        }
    }
}
