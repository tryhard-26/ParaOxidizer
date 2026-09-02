//! Adaptive mixed-precision optimizer and Pareto frontier solver.

pub mod pareto;
pub mod planner;

pub use pareto::{compute_pareto_frontier, ParetoPoint};
pub use planner::{OptimizationConstraints, OptimizationPlanner, QuantizationPlan};
