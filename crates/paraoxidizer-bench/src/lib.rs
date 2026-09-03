//! Benchmarking suite for ParaOxidizer inference speed, latency, kernels, and system metrics.

pub mod fidelity;
pub mod microbench;
pub mod runner;
pub mod system;

pub use fidelity::{run_fidelity_benchmarks, FidelityBenchmarkResult};
pub use microbench::{run_dot_product_benchmarks, run_gemv_benchmarks, DotProductBenchmarkResult, GemvBenchmarkResult};
pub use runner::{BenchmarkHarness, BenchmarkResult};
pub use system::{run_system_benchmarks, SystemBenchmarkResult};

