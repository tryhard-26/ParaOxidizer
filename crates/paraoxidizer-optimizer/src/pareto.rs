use serde::{Deserialize, Serialize};

/// A single configuration candidate evaluated in the multi-objective search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub name: String,
    pub memory_gb: f64,
    pub latency_ms: f64,
    pub quality_pct: f64,
    pub is_pareto_optimal: bool,
    pub description: String,
    pub default_precision: String,
    pub group_size: usize,
    pub outlier_strategy: String,
}

impl ParetoPoint {
    /// Returns true if `self` dominates `other` in the multi-objective Pareto sense:
    /// min(memory), min(latency), max(quality)
    pub fn dominates(&self, other: &ParetoPoint) -> bool {
        let mem_better_or_equal = self.memory_gb <= other.memory_gb;
        let lat_better_or_equal = self.latency_ms <= other.latency_ms;
        let qual_better_or_equal = self.quality_pct >= other.quality_pct;

        let at_least_one_strictly_better = self.memory_gb < other.memory_gb
            || self.latency_ms < other.latency_ms
            || self.quality_pct > other.quality_pct;

        mem_better_or_equal
            && lat_better_or_equal
            && qual_better_or_equal
            && at_least_one_strictly_better
    }
}

/// Compute the non-dominated Pareto frontier from a list of evaluated points
pub fn compute_pareto_frontier(mut points: Vec<ParetoPoint>) -> Vec<ParetoPoint> {
    let n = points.len();
    for i in 0..n {
        let mut dominated = false;
        for j in 0..n {
            if i != j && points[j].dominates(&points[i]) {
                dominated = true;
                break;
            }
        }
        points[i].is_pareto_optimal = !dominated;
    }

    // Sort by memory ascending
    points.sort_by(|a, b| a.memory_gb.partial_cmp(&b.memory_gb).unwrap());
    points
}
