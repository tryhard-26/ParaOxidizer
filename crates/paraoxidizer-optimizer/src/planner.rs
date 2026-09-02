use crate::pareto::{compute_pareto_frontier, ParetoPoint};
use paraoxidizer_calibration::sensitivity::{ModelSensitivityReport, SensitivityLevel};
use paraoxidizer_core::hardware::HardwareInfo;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Constraints provided by the user for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraints {
    pub max_memory_gb: Option<f64>,
    pub max_latency_ms: Option<f64>,
    pub min_quality_pct: Option<f64>,
    pub target_hardware: String,
}

impl Default for OptimizationConstraints {
    fn default() -> Self {
        Self {
            max_memory_gb: None,
            max_latency_ms: None,
            min_quality_pct: Some(98.0),
            target_hardware: "auto".to_string(),
        }
    }
}

/// The final planned quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationPlan {
    pub run_id: String,
    pub selected_point: ParetoPoint,
    pub pareto_frontier: Vec<ParetoPoint>,
    pub layer_precisions: HashMap<String, String>,
    pub layer_group_sizes: HashMap<String, usize>,
    pub outlier_policy: String,
}

pub struct OptimizationPlanner;

impl OptimizationPlanner {
    pub fn plan(
        total_params: u64,
        sensitivity: &ModelSensitivityReport,
        constraints: &OptimizationConstraints,
        hw: &HardwareInfo,
    ) -> QuantizationPlan {
        let mut rng = rand::thread_rng();
        let hex_run: String = (0..6)
            .map(|_| format!("{:x}", rng.gen_range(0..16)))
            .collect();
        let run_id = format!("pox-run-{}", hex_run);

        // Baseline FP16 parameters size in GB: 2 bytes per param
        let base_fp16_gb = (total_params as f64 * 2.0) / (1024.0 * 1024.0 * 1024.0);

        // Bandwidth cost model for latency
        let base_bandwidth_gbps = if hw.unified_memory {
            100.0 // M-series memory bandwidth
        } else if hw.simd_avx512 {
            65.0
        } else {
            45.0
        };

        // Candidate 1: INT4 All (group 128)
        let mem_c1 = base_fp16_gb * 0.28;
        let lat_c1 = (mem_c1 / base_bandwidth_gbps) * 1000.0 + 8.0;
        let qual_c1 = 96.8;

        // Candidate 2: Adaptive Mixed Precision (INT4 low/med, INT8 high, FP16 critical)
        let mem_c2 = base_fp16_gb * 0.35;
        let lat_c2 = (mem_c2 / base_bandwidth_gbps) * 1000.0 + 7.5;
        let qual_c2 = 98.7;

        // Candidate 3: Mixed High Quality (INT4-g64 low, INT8 med/high, FP16 critical)
        let mem_c3 = base_fp16_gb * 0.42;
        let lat_c3 = (mem_c3 / base_bandwidth_gbps) * 1000.0 + 7.0;
        let qual_c3 = 99.3;

        // Candidate 4: INT8 All
        let mem_c4 = base_fp16_gb * 0.52;
        let lat_c4 = (mem_c4 / base_bandwidth_gbps) * 1000.0 + 5.5; // Faster kernel than INT4 on scalar
        let qual_c4 = 99.6;

        // Candidate 5: FP16 Uncompressed Baseline
        let mem_c5 = base_fp16_gb;
        let lat_c5 = (mem_c5 / base_bandwidth_gbps) * 1000.0;
        let qual_c5 = 100.0;

        let candidates = vec![
            ParetoPoint {
                name: "INT4-Compact".into(),
                memory_gb: (mem_c1 * 10.0).round() / 10.0,
                latency_ms: (lat_c1 * 10.0).round() / 10.0,
                quality_pct: qual_c1,
                is_pareto_optimal: true,
                description: "Full INT4 with group size 128 for maximum memory reduction".into(),
                default_precision: "INT4".into(),
                group_size: 128,
                outlier_strategy: "disabled".into(),
            },
            ParetoPoint {
                name: "Adaptive-Mixed".into(),
                memory_gb: (mem_c2 * 10.0).round() / 10.0,
                latency_ms: (lat_c2 * 10.0).round() / 10.0,
                quality_pct: qual_c2,
                is_pareto_optimal: true,
                description: "Adaptive mixed INT4/INT8 with outlier protection".into(),
                default_precision: "INT4".into(),
                group_size: 128,
                outlier_strategy: "automatic".into(),
            },
            ParetoPoint {
                name: "Selective-INT8".into(),
                memory_gb: (mem_c3 * 10.0).round() / 10.0,
                latency_ms: (lat_c3 * 10.0).round() / 10.0,
                quality_pct: qual_c3,
                is_pareto_optimal: true,
                description: "Higher precision INT8 on Attention K/V and MLP Down".into(),
                default_precision: "INT4".into(),
                group_size: 64,
                outlier_strategy: "conservative".into(),
            },
            ParetoPoint {
                name: "INT8-Fast".into(),
                memory_gb: (mem_c4 * 10.0).round() / 10.0,
                latency_ms: (lat_c4 * 10.0).round() / 10.0,
                quality_pct: qual_c4,
                is_pareto_optimal: true,
                description: "Uniform INT8 symmetric quantization".into(),
                default_precision: "INT8".into(),
                group_size: 0,
                outlier_strategy: "disabled".into(),
            },
            ParetoPoint {
                name: "FP16-Baseline".into(),
                memory_gb: (mem_c5 * 10.0).round() / 10.0,
                latency_ms: (lat_c5 * 10.0).round() / 10.0,
                quality_pct: qual_c5,
                is_pareto_optimal: false,
                description: "Unquantized half-precision reference baseline".into(),
                default_precision: "FP16".into(),
                group_size: 0,
                outlier_strategy: "disabled".into(),
            },
        ];

        let frontier = compute_pareto_frontier(candidates);

        // Select best candidate matching constraints
        let mut selected = frontier[1].clone(); // Default to Adaptive-Mixed

        for pt in frontier.iter().rev() {
            let mem_ok = constraints
                .max_memory_gb
                .map(|m| pt.memory_gb <= m)
                .unwrap_or(true);
            let lat_ok = constraints
                .max_latency_ms
                .map(|l| pt.latency_ms <= l)
                .unwrap_or(true);
            let qual_ok = constraints
                .min_quality_pct
                .map(|q| pt.quality_pct >= q)
                .unwrap_or(true);

            if mem_ok && lat_ok && qual_ok {
                selected = pt.clone();
                break;
            }
        }

        // Generate per-layer assignments based on selected strategy and sensitivity
        let mut layer_precisions = HashMap::new();
        let mut layer_group_sizes = HashMap::new();

        for t in &sensitivity.tensors {
            match selected.name.as_str() {
                "INT4-Compact" => {
                    layer_precisions.insert(t.name.clone(), "INT4".into());
                    layer_group_sizes.insert(t.name.clone(), 128);
                }
                "INT8-Fast" => {
                    layer_precisions.insert(t.name.clone(), "INT8".into());
                    layer_group_sizes.insert(t.name.clone(), 0);
                }
                "FP16-Baseline" => {
                    layer_precisions.insert(t.name.clone(), "FP16".into());
                    layer_group_sizes.insert(t.name.clone(), 0);
                }
                _ => {
                    // Adaptive Mixed
                    let (prec, group) = match t.level {
                        SensitivityLevel::Critical => ("FP16", 0),
                        SensitivityLevel::High => ("INT8", 0),
                        SensitivityLevel::Medium => ("INT4", 64),
                        SensitivityLevel::Low => ("INT4", 128),
                    };
                    layer_precisions.insert(t.name.clone(), prec.into());
                    layer_group_sizes.insert(t.name.clone(), group);
                }
            }
        }

        QuantizationPlan {
            run_id,
            selected_point: selected,
            pareto_frontier: frontier,
            layer_precisions,
            layer_group_sizes,
            outlier_policy: "automatic".into(),
        }
    }
}
