# ParaOxidizer (`paraoxidizer` / `pox`)

[![Crates.io](https://img.shields.io/badge/crates.io-paraoxidizer-orange.svg)](https://crates.io/crates/paraoxidizer)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-brightgreen.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20|%20Linux%20|%20Windows-lightgrey.svg)]()

ParaOxidizer is a standalone Rust library and CLI toolkit for quantizing, packaging, cryptographically verifying, and executing Transformer LLM weights without Python or external C++ runtimes.

It converts SafeTensors (single and multi-file sharded), GGUF files, or remote Hugging Face Hub repositories into `.pox` (ParaOxidizer Optimized Executable)—a zero-copy, 64-byte aligned, tamper-evident binary container.

```text
  [Hugging Face / SafeTensors / GGUF]
                 │
                 ▼
     [Model Ingestion & Parser]
                 │
                 ▼
  [Calibration & Sensitivity Analysis]  ── (Identifies outlier channels ≥ 3.5σ)
                 │
                 ▼
  [Adaptive Mixed-Precision Optimizer]  ── (Solves Pareto frontier: min Memory + Latency)
                 │
                 ▼
   [Quantization Kernel Dispatch]       ── (INT4 Affine Group-wise + INT8 Sym + FP16 Outliers)
                 │
                 ▼
     [Cryptographic Seal & Sign]        ── (SHA-256 Merkle Hash Tree + Ed25519 Signature)
                 │
                 ▼
        [output_model.pox]
          │             │
          ▼             ▼
   [Local Engine]  [OpenAI API Server]
```

---

## Technical Highlights

- **Pure Rust Architecture**: Compiles to a single static binary (`pox` / `paraoxidizer`) with zero Python, PyTorch, CUDA toolkit, or C/C++ compiler requirements.
- **Zero-Copy Memory-Mapped Layout (`.pox`)**: 128-byte header with 64-byte payload padding; maps weight tensors directly into virtual memory via `memmap2` with microsecond-level cold start overhead.
- **Group-Wise Affine INT4 Quantization**: Block-quantized INT4 packing (`min_offset + q * scale`, with $q \in [0, 15]$) across group sizes 32, 64, 128, and 256.
- **Sparse Outlier Coordinate Tables**: Automatically extracts activation/weight spikes ($\ge 3.5\sigma$) into FP16 coordinate tables, eliminating range inflation in low-bit channels.
- **Supply-Chain Verification**: Every tensor is tracked in a SHA-256 manifest. Containers support native Ed25519 digital signing and strict cryptographic integrity verification.
- **Hardware SIMD Vectorization**: Native ARM NEON paths on Apple Silicon (`vfmaq_f32`) and unrolled fallback paths for AVX2/x86_64.
- **Workload Profiling**: Built-in calibration profiles (`agentic`, `coding`, `reasoning`, `long-context`, `chat`, `general`) generate portable `.poxcal` calibration profiles.
- **Inference Server**: Embedded Axum HTTP server offering OpenAI-compatible `/v1/chat/completions` (with Server-Sent Events streaming) and Prometheus `/metrics`.

---

## Hardware Benchmark Results

Measurements captured on **Apple Silicon M4** (10-core CPU, ARM NEON, unified memory, Darwin arm64) using `cargo run --release -p paraoxidizer-bench --bin pox-bench`:

### 1. Vector Dot-Product Microbenchmarks (ARM NEON vs Scalar)

Evaluates the inner product kernel used during dequantized projection passes:

| Vector Dimension ($N$) | ARM NEON SIMD | Scalar Fallback | SIMD Speedup |
| :--- | :--- | :--- | :--- |
| **512** | 33.9 ns | 192.2 ns | **5.67×** |
| **1024** | 76.4 ns | 677.2 ns | **8.86×** |
| **2048** | 209.3 ns | 1768.0 ns | **8.45×** |
| **4096** | 742.0 ns | 2777.0 ns | **3.74×** |
| **8192** | 1149.8 ns | 6184.5 ns | **5.38×** |

### 2. Quantized Matrix-Vector Multiplication (GEMV)

Evaluates $y = Wx$ across Transformer layer dimensions:

| Kernel Configuration | Dimensions ($M \times K$) | Latency | Bandwidth | Compute | Speedup vs FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Projection INT4 G32** | $1024 \times 1024$ | 516.1 µs | 1.29 GB/s | 4.06 GFLOPS | **1.14×** |
| **Projection INT4 G128** | $1024 \times 1024$ | 527.0 µs | 1.07 GB/s | 3.98 GFLOPS | **1.11×** |
| **Projection INT8 Sym** | $1024 \times 1024$ | 545.2 µs | 1.94 GB/s | 3.85 GFLOPS | **1.08×** |
| **Attention INT4 G32** | $2048 \times 2048$ | 2108.7 µs | 1.25 GB/s | 3.98 GFLOPS | **1.01×** |
| **Attention INT4 G128** | $2048 \times 2048$ | 2150.1 µs | 1.04 GB/s | 3.90 GFLOPS | **0.99×** |
| **FFN Layer INT4 G32** | $4096 \times 4096$ | 8577.7 µs | 1.23 GB/s | 3.91 GFLOPS | **1.06×** |
| **FFN Layer INT4 G128** | $4096 \times 4096$ | 8976.2 µs | 1.00 GB/s | 3.74 GFLOPS | **1.01×** |

### 3. Quantization Compression & Reconstruction Error (Fidelity)

Evaluated on 1,000,000 parameter samples from a normal distribution with pathological outlier spikes ($> 4.0\sigma$):

| Quantization Scheme | Compression Ratio | Quantization Rate | Mean Squared Error | Cosine Similarity | SQNR (dB) | Outliers Preserved |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **INT8 Symmetric** | 2.00× | 1496.9 MB/s | $1.50 \times 10^{-5}$ | 0.996385 | 21.39 dB | 0 |
| **INT4 Group-128** | 3.76× | 1373.5 MB/s | $2.59 \times 10^{-5}$ | 0.993796 | 19.02 dB | 0 |
| **INT4 Group-64** | 3.56× | 1272.2 MB/s | $1.83 \times 10^{-5}$ | 0.995613 | 20.54 dB | 0 |
| **INT4 Group-32** | 3.20× | 1277.9 MB/s | $1.35 \times 10^{-5}$ | 0.996758 | 21.86 dB | 0 |
| **INT4 G128 + Outliers** | 3.69× | 683.8 MB/s | $2.06 \times 10^{-5}$ | 0.995064 | 20.02 dB | 1,815 |

*Note: Extracting outliers into a sparse FP16 table recovers dynamic range in low-bit channels, boosting SQNR from 19.02 dB to 20.02 dB.*

### 4. Container I/O & Supply-Chain Operations

| Operation | Payload Size | Latency | Bandwidth | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`.pox` Zero-Copy mmap** | 10.0 MB | **27.54 µs** | **380.78 GB/s** | Virtual memory mapping of tensor tables |
| **Eager File Buffer Read** | 10.0 MB | 1.32 ms | 7.97 GB/s | Standard heap allocation & file copy baseline |
| **SHA-256 Merkle Hash** | 25.0 MB | 45.53 ms | 0.58 GB/s | Streaming cryptographic integrity check |
| **Ed25519 Sign** | 32 B | 9.01 µs | — | Digital signature generation per artifact |
| **Ed25519 Verify** | 32 B | 21.15 µs | — | Strict signature verification |

---

## Installation & Build

### From Source
```bash
git clone https://github.com/tryhard/ParaOxidizer.git
cd ParaOxidizer
cargo build --release
```
The compiled binaries will be at `target/release/pox` and `target/release/paraoxidizer`.

### From Crates.io
```bash
cargo install paraoxidizer
```

### Library Dependency
Add to your `Cargo.toml`:
```toml
[dependencies]
paraoxidizer = "0.1.0"
```

---

## CLI Reference

### Ingestion & Inspection
```bash
# Inspect local Hugging Face directory or SafeTensors file
pox inspect path/to/hf_model

# Ingest and inspect directly from Hugging Face Hub (cached to ~/.cache/paraoxidizer/hub)
pox inspect hf-internal-testing/tiny-random-LlamaForCausalLM

# Inspect existing .pox container with JSON output
pox inspect model.pox --format json

# Probe host CPU, cache topology, SIMD extensions, and memory bandwidth
pox hardware
```

### Quantization & Optimization
```bash
# Direct group-wise INT4 quantization
pox quantize \
    --model path/to/hf_model \
    --bits 4 \
    --group-size 128 \
    --outlier automatic \
    --output model_int4.pox

# Generate calibration profile for a specific task domain
pox calibrate \
    --model path/to/hf_model \
    --profile agentic \
    --samples 512 \
    --output agentic.poxcal

# Parameter sensitivity analysis (identifies layers critical to downstream accuracy)
pox analyze \
    --model path/to/hf_model \
    --calibration agentic.poxcal

# Constrained mixed-precision optimization (Pareto frontier solver)
pox optimize \
    --model path/to/hf_model \
    --memory 6GB \
    --latency 40ms \
    --quality 98.0 \
    --calibration agentic.poxcal \
    --output model_pareto.pox
```

### Validation, Signing & Verification
```bash
# Validate internal structure, header offsets, and tensor ranges
pox validate model.pox

# Generate Ed25519 cryptographic keypair
pox keygen --output release_key

# Cryptographically sign container
pox sign model.pox --key release_key.key --output model_signed.pox

# Verify digital signature and bit-flip tamper resistance
pox verify model_signed.pox --pubkey $(cat release_key.pub)

# Compute structural drift and bitwidth diff between two models
pox diff baseline.pox optimized.pox
```

### Inference & Benchmarking
```bash
# Run local interactive prompt inference
pox run model.pox --prompt "Write a concurrent worker pool in Rust." --max-tokens 128

# Measure decode throughput (tok/s), TTFT, and latency percentiles
pox benchmark model.pox --tokens 64

# Run the complete hardware microbenchmark suite
pox benchmark --suite
```

### HTTP Server (OpenAI-Compatible)
```bash
pox serve model.pox --host 127.0.0.1 --port 8080
```
Query via `curl`:
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain zero-copy deserialization."}],
    "stream": true
  }'
```

---

## Configuration-Driven Pipeline (`paraoxidizer.toml`)

Automate full optimization workflows reproducibly:

```toml
[model]
source = "hf-internal-testing/tiny-random-LlamaForCausalLM"
architecture = "Llama"
output = "artifacts/llama_optimized.pox"

[calibration]
profile = "agentic"
samples = 256
sequence_length = 512

[quantization]
algorithm = "min-max"
allowed_precisions = ["INT4", "INT8"]
group_sizes = [32, 64, 128]
outlier_policy = "automatic"

[optimization]
memory_limit = "4GB"
latency_limit = "45ms"
quality_floor = 98.0
target_hardware = "auto"

[security]
verify_source = true
require_signature = false

[runtime]
backend = "auto"
context = 2048
batching = 1
```

Run via:
```bash
pox build paraoxidizer.toml
```

---

## The `.pox` Container Binary Layout

All multi-byte numeric values are encoded in little-endian format. Data segments are padded to 64-byte alignment boundaries for direct AVX-512 and ARM NEON cache-line streaming.

```text
+---------------------------------------------------------------+
| POX MAGIC: b"POX\x01" (4 bytes)                               |
+---------------------------------------------------------------+
| 128-BYTE HEADER (64-byte aligned):                            |
|   Format version, flags, 64-bit offsets and segment lengths   |
+---------------------------------------------------------------+
| MODEL CONFIGURATION (JSON):                                   |
|   Architecture, layer counts, dimensions, attention heads     |
+---------------------------------------------------------------+
| QUANTIZATION PLAN (JSON):                                     |
|   Layer-by-layer bitwidths, group sizes, outlier policies     |
+---------------------------------------------------------------+
| TENSOR DATA PAYLOAD (64-byte aligned blocks):                 |
|   - Quantized weight matrices (packed INT4 nibbles or INT8)   |
|   - Scales & min-offsets (FP16 arrays)                        |
|   - Sparse outlier coordinate tables (FP16 values + U32 idx)  |
+---------------------------------------------------------------+
| TENSOR INDEX:                                                 |
|   Directory of tensor descriptors, shapes, dtypes, hashes     |
+---------------------------------------------------------------+
| INTEGRITY MANIFEST:                                           |
|   Per-tensor SHA-256 checksums, toolchain build info, run ID  |
+---------------------------------------------------------------+
| CRYPTOGRAPHIC FOOTER:                                         |
|   Ed25519 Public Key (32 bytes) + Digital Signature (64 bytes)|
+---------------------------------------------------------------+
```

---

## Rust Library API Example

```rust
use paraoxidizer::format::PoxFile;
use paraoxidizer::runtime::{engine::PoxEngine, sampler::SamplerConfig};

fn main() -> anyhow::Result<()> {
    // Zero-copy virtual memory map
    let pox = PoxFile::open("model_opt.pox")?;

    // Verify SHA-256 Merkle integrity across all tensors
    pox.verify_integrity()?;

    // Instantiate inference engine
    let engine = PoxEngine::new(pox);

    // Streaming autoregressive generation
    let sampler = SamplerConfig::default();
    engine.generate_stream(
        "Explain memory safety in Rust.",
        64,
        sampler,
        |token| {
            print!("{token}");
            true
        },
    )?;

    Ok(())
}
```

---

## Workspace Crates

| Crate | Path | Role |
| :--- | :--- | :--- |
| `paraoxidizer-core` | `crates/paraoxidizer-core` | Architecture abstractions, tensor shapes, hardware probing |
| `paraoxidizer-format` | `crates/paraoxidizer-format` | `.pox` format reader/writer, SafeTensors/GGUF/HF ingestion |
| `paraoxidizer-quant` | `crates/paraoxidizer-quant` | Affine INT4 & symmetric INT8 kernels, SIMD dot products, outliers |
| `paraoxidizer-calibration` | `crates/paraoxidizer-calibration` | Task profile generators, parameter sensitivity classification |
| `paraoxidizer-optimizer` | `crates/paraoxidizer-optimizer` | Multi-objective Pareto frontier mixed-precision solver |
| `paraoxidizer-security` | `crates/paraoxidizer-security` | Merkle hash trees, Ed25519 signing/verification, quotas |
| `paraoxidizer-runtime` | `crates/paraoxidizer-runtime` | Autoregressive Transformer decoder, KV cache, tokenizer |
| `paraoxidizer-serve` | `crates/paraoxidizer-serve` | Axum HTTP server (`/v1/chat/completions`, `/metrics`) |
| `paraoxidizer-bench` | `crates/paraoxidizer-bench` | Microbenchmarks (SIMD, GEMV, fidelity, system, TTFT, tok/s) |
| `paraoxidizer-cli` | `crates/paraoxidizer-cli` | Unified CLI interface (`pox` / `paraoxidizer`) |

---

## License

Dual-licensed under either of:
- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
