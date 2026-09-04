use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "pox",
    bin_name = "pox",
    version,
    about = "ParaOxidizer — a Rust-native LLM quantization, optimization, verification, and inference toolkit",
    long_about = "ParaOxidizer turns oversized neural-network parameters into compact, hardware-optimized, verifiable inference artifacts."
)]
pub struct Cli {
    #[arg(long, global = true, default_value = "text", help = "Output format: text, json, jsonl")]
    pub format: String,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    #[command(about = "Deep inspection of SafeTensors, GGUF, Hugging Face, or .pox models")]
    Inspect {
        #[arg(help = "Path to model file or Hugging Face directory")]
        path: String,
    },

    #[command(about = "Report host hardware, SIMD features, accelerators, and recommendations")]
    Hardware,

    #[command(about = "Calibrate model on workload traces and generate .poxcal artifact")]
    Calibrate {
        #[arg(long, help = "Source model path (SafeTensors, HF dir, or .pox)")]
        model: String,
        #[arg(long, help = "Path to JSONL / text calibration dataset file")]
        dataset: Option<String>,
        #[arg(long, default_value = "general", help = "Workload profile: general, coding, reasoning, agentic, chat, long-context")]
        profile: String,
        #[arg(long, default_value = "256", help = "Number of calibration samples")]
        samples: usize,
        #[arg(long, default_value = "calibration.poxcal", help = "Output .poxcal file path")]
        output: String,
    },

    #[command(about = "Analyze parameter and layer sensitivity to quantization")]
    Analyze {
        #[arg(long, help = "Source model path")]
        model: String,
        #[arg(long, help = "Path to .poxcal calibration file")]
        calibration: Option<String>,
    },

    #[command(about = "Directly quantize model to .pox format (INT4, INT8, FP16)")]
    Quantize {
        #[arg(long, help = "Source model path")]
        model: String,
        #[arg(long, default_value = "4", help = "Target bit precision: 4 or 8")]
        bits: usize,
        #[arg(long, default_value = "128", help = "Group size: 32, 64, 128, 256")]
        group_size: usize,
        #[arg(long, default_value = "automatic", help = "Outlier policy: disabled, automatic, conservative, aggressive")]
        outlier: String,
        #[arg(long, default_value = "min-max", help = "Quantization algorithm: min-max, awq, gptq")]
        algorithm: String,
        #[arg(long, default_value = "model.pox", help = "Output .pox file path")]
        output: String,
    },

    #[command(about = "Flagship adaptive mixed-precision optimizer with Pareto frontier search")]
    Optimize {
        #[arg(help = "Source model path (SafeTensors, HF directory, or .pox)")]
        model: String,
        #[arg(long, help = "Maximum memory limit, e.g. 6GB, 4.5GB")]
        memory: Option<String>,
        #[arg(long, help = "Maximum latency target, e.g. 50ms")]
        latency: Option<String>,
        #[arg(long, help = "Minimum relative quality floor, e.g. 98.5%")]
        quality: Option<f64>,
        #[arg(long, help = "Path to .poxcal calibration file")]
        calibration: Option<String>,
        #[arg(long, default_value = "auto", help = "Target hardware profile: auto, cpu, metal, cuda")]
        hardware: String,
        #[arg(long, default_value = "model.pox", help = "Output .pox file path")]
        output: String,
    },

    #[command(about = "Check numerical validity, structure, and decompression of a .pox artifact")]
    Validate {
        #[arg(help = "Path to .pox model file")]
        model: String,
    },

    #[command(about = "Verify cryptographic supply chain, SHA-256 hashes, and Ed25519 signature")]
    Verify {
        #[arg(help = "Path to .pox model file")]
        model: String,
        #[arg(long, help = "Trusted Ed25519 public key in hex (optional)")]
        pubkey: Option<String>,
    },

    #[command(about = "Benchmark TTFT, throughput (tok/s), latency percentiles, memory, or run hardware suite")]
    Benchmark {
        #[arg(help = "Path to .pox model file (optional if --suite is specified)")]
        model: Option<String>,
        #[arg(long, help = "Execute the comprehensive hardware SIMD, GEMV, fidelity, and system microbenchmark suite")]
        suite: bool,
        #[arg(long, default_value = "Explain the architecture of a modern transformer.", help = "Benchmark prompt")]
        prompt: String,
        #[arg(long, default_value = "64", help = "Number of tokens to decode")]
        tokens: usize,
    },

    #[command(about = "Compare multiple .pox artifacts side-by-side")]
    Compare {
        #[arg(help = "Paths to .pox model files to compare", num_args = 1..)]
        models: Vec<String>,
    },

    #[command(about = "Execute interactive prompt inference with streaming token output")]
    Run {
        #[arg(help = "Path to .pox model file")]
        model: String,
        #[arg(long, default_value = "Explain the significance of zero-copy tensor memory mapping.", help = "Prompt text")]
        prompt: String,
        #[arg(long, default_value = "128", help = "Maximum tokens to generate")]
        max_tokens: usize,
        #[arg(long, default_value = "0.7", help = "Sampling temperature")]
        temperature: f32,
        #[arg(long, help = "Optional draft model path for speculative decoding")]
        draft: Option<String>,
        #[arg(long, default_value = "3", help = "Speculative lookahead K candidate tokens")]
        lookahead: usize,
    },

    #[command(about = "Launch OpenAI-compatible HTTP server (/v1/chat/completions, /metrics)")]
    Serve {
        #[arg(help = "Path to .pox model file")]
        model: String,
        #[arg(long, default_value = "127.0.0.1", help = "Bind host")]
        host: String,
        #[arg(long, default_value = "8080", help = "Bind port")]
        port: u16,
        #[arg(long, help = "Optional draft model path for speculative decoding")]
        draft: Option<String>,
    },

    #[command(about = "Launch interactive terminal TUI dashboard for real-time inference telemetry and memory")]
    Monitor {
        #[arg(help = "Optional path to .pox model file")]
        model: Option<String>,
        #[arg(long, default_value = "500", help = "Refresh interval in milliseconds")]
        interval_ms: u64,
    },

    #[command(about = "Sign a .pox artifact with an Ed25519 private key")]
    Sign {
        #[arg(help = "Path to .pox model file")]
        model: String,
        #[arg(long, help = "Ed25519 private key in hex")]
        key: String,
        #[arg(long, help = "Output path for signed artifact (defaults to in-place replacement)")]
        output: Option<String>,
    },

    #[command(about = "Inspect optimization provenance and reproducible build metadata")]
    InspectRun {
        #[arg(help = "Optimization Run ID (e.g. pox-run-7f31a9) or path to .pox file")]
        run_id: String,
    },

    #[command(about = "Reproduce an optimization run and verify artifact hash reproducibility")]
    Reproduce {
        #[arg(help = "Run ID or path to .pox file")]
        run_id: String,
    },

    #[command(about = "Generate or view a standardized workload profile (e.g. coding-agent)")]
    Workload {
        #[arg(help = "Profile name: coding-agent, general, reasoning, long-context, chat")]
        profile: String,
        #[arg(long, help = "Output path to save JSONL trace samples")]
        output: Option<String>,
    },

    #[command(about = "Compare parameters, precision shifts, and size between two .pox models")]
    Diff {
        #[arg(help = "Path to baseline model-a.pox")]
        model_a: String,
        #[arg(help = "Path to optimized model-b.pox")]
        model_b: String,
    },

    #[command(about = "Build and optimize model driven by a TOML configuration file")]
    Build {
        #[arg(help = "Path to paraoxidizer.toml configuration")]
        config: String,
    },

    #[command(about = "Generate a fresh Ed25519 signing keypair for model supply-chain security")]
    Keygen {
        #[arg(long, default_value = "pox_key", help = "Prefix for private (.key) and public (.pub) files")]
        output: String,
    },
}
