//! # ParaOxidizer
//!
//! **Parameter Oxidizer** — a Rust-native LLM quantization, optimization, verification,
//! and inference toolkit.
//!
//! ## Core Modules
//! - [`core`]: Common types, model architectures, hardware detection, error types, and configurations.
//! - [`format`]: File formats including the native zero-copy `.pox` container and `.poxcal` format.
//! - [`quant`]: Quantization kernels (FP32, FP16, BF16, INT8, INT4 with groups, FP8, outlier tables, SIMD).
//! - [`calibration`]: Activation recording, statistics, and workload profiles (including agentic traces).
//! - [`optimizer`]: Adaptive mixed-precision optimizer solving the memory-latency-quality Pareto frontier.
//! - [`security`]: Cryptographic integrity (SHA-256 manifests, Ed25519 signatures, secure parser limits).
//! - [`runtime`]: Standalone quantized Transformer inference engine, KV cache, tokenizer, and sampler.
//! - [`serve`]: OpenAI-compatible HTTP API server with streaming SSE and Prometheus metrics.
//! - [`bench`]: Standardized inference and quality benchmarking harness.
//! - [`cli`]: Complete CLI commands for the `pox` / `paraoxidizer` binary.

pub use paraoxidizer_core as core;
pub use paraoxidizer_format as format;
pub use paraoxidizer_quant as quant;
pub use paraoxidizer_calibration as calibration;
pub use paraoxidizer_optimizer as optimizer;
pub use paraoxidizer_security as security;
pub use paraoxidizer_runtime as runtime;
pub use paraoxidizer_serve as serve;
pub use paraoxidizer_bench as bench;
pub use paraoxidizer_cli as cli;
