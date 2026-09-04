# ParaOxidizer (`paraoxidizer` / `pox`)

[![Crates.io](https://img.shields.io/badge/crates.io-paraoxidizer-orange.svg)](https://crates.io/crates/paraoxidizer)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-brightgreen.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20(Metal%20+%20NEON)%20|%20Linux%20|%20Windows-lightgrey.svg)](https://github.com/tryhard-26/ParaOxidizer)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://github.com/tryhard-26/ParaOxidizer)

---

## Short Summary

**ParaOxidizer** is a Rust-native LLM quantization, optimization, cryptographic verification, and inference toolkit. It converts oversized neural network weights into hardware-optimized, verifiable `.pox` (ParaOxidizer Optimized Executable) binaries without requiring Python runtimes, PyTorch, or external C++ dependencies.

```text
[Hugging Face / SafeTensors / GGUF]
                 │
                 ▼
     [Model Ingestion & Parser]
                 │
                 ▼
  [Hessian & Sensitivity Analysis]  ── (Empirical Hessian H^-1, Outliers >= 3.5σ)
                 │
                 ▼
  [Adaptive Mixed-Precision Solver] ── (AWQ / GPTQ / Min-Max Pareto Frontiers)
                 │
                 ▼
   [Quantization Kernel Dispatch]   ── (INT4 Affine Groups + INT8 + FP16 Outliers)
                 │
                 ▼
     [Cryptographic Seal & Sign]    ── (SHA-256 Merkle Tree + Ed25519 Signature)
                 │
                 ▼
        [output_model.pox]
          │             │
          ▼             ▼
   [Local Metal/NEON] [OpenAI API Server]
```

### Core Architecture

- **Zero-Copy Memory-Mapped Layout (`.pox`)**: 128-byte binary header with 64-byte tensor alignment. Weights are mapped directly into virtual memory via `memmap2`, enabling sub-millisecond cold starts without memory duplication.
- **Second-Order Hessian Calibration**: Computes empirical activation covariance $H = \frac{2}{N} X X^T + \lambda I$ and its inverse $H^{-1}$ for activation-aware weight quantization (AWQ) and greedy column-by-column error compensation (GPTQ).
- **Sparse Outlier Isolation**: Extracts heavy-tailed activation and weight spikes ($\ge 3.5\sigma$) into auxiliary FP16 coordinate tables, eliminating dynamic range degradation in low-bit quantization channels.
- **Hardware-Accelerated Kernels**: 
  - **Metal GPU**: Native Apple Silicon MSL compute shader pipeline (`gemv_int4_kernel`) leveraging unified memory (`MTLResourceStorageModeShared`).
  - **ARM NEON**: Vectorized inner products with SIMD fused multiply-accumulate (`vfmaq_f32`).
  - **AVX2 / Scalar**: High-performance portable fallbacks for x86_64.
- **Cryptographic Supply-Chain Security**: Tensor-level SHA-256 Merkle hashes, Ed25519 digital signing, and static backdoor detection (detecting NaN/Inf poisoning, anomalous weight clustering, and hidden trigger biases).
- **Embedded OpenAI API Server**: Built-in async HTTP service (`/v1/chat/completions` with SSE token streaming, `/v1/models`, and `/metrics`).
- **Python Bindings (`pyo3`)**: Native `pox` C-extension module for programmatic Python invocation.

---

## Docs

### Installation

#### 1. From Crates.io (CLI & Core Libraries)
```bash
cargo install paraoxidizer --locked
```

#### 2. From Source
```bash
git clone https://github.com/tryhard-26/ParaOxidizer.git
cd ParaOxidizer
cargo build --release --workspace
```
The compiled binaries will be available at `./target/release/pox` (alias: `paraoxidizer`) and `./target/release/pox-bench`.

#### 3. Python Bindings (PyO3)
```bash
cd crates/paraoxidizer-py
pip install maturin
maturin develop --release
```

---

### Command-Line Interface (`pox`)

```text
Usage: pox [OPTIONS] <COMMAND>

Commands:
  quantize  Convert HuggingFace, SafeTensors, or GGUF weights to .pox format
  inspect   Display tensor shapes, data types, and cryptographic metadata
  verify    Verify SHA-256 tensor checksums and Ed25519 digital signatures
  sign      Cryptographically sign a .pox file with an Ed25519 private key
  diff      Compare two model variants across precision, sparsity, and MSE
  serve     Launch an OpenAI-compatible HTTP inference server
  bench     Execute hardware microbenchmarks and kernel performance suites
  help      Print this message or the help of the given subcommand(s)
```

#### Ingestion & Quantization (`pox quantize`)
Convert a remote Hugging Face repository or local SafeTensors/GGUF model to `.pox`:

```bash
# Min-Max standard INT4 group quantization (group size 128)
pox quantize \
  --input meta-llama/Llama-3-8B \
  --output llama3-int4.pox \
  --format int4 \
  --group-size 128

# Second-order AWQ with empirical calibration
pox quantize \
  --input mistralai/Mistral-7B-v0.1 \
  --output mistral-awq.pox \
  --algorithm awq \
  --calib reasoning \
  --group-size 128 \
  --outliers 3.5

# GPTQ with column-by-column Hessian error compensation
pox quantize \
  --input Qwen/Qwen2.5-7B \
  --output qwen-gptq.pox \
  --algorithm gptq \
  --group-size 64
```

#### Inspection & Validation (`pox inspect`)
Inspect tensor manifest, architecture specs, alignment, and parameter sizes:
```bash
pox inspect llama3-int4.pox --tensors
```

#### Cryptographic Verification (`pox verify` & `pox sign`)
```bash
# Generate keypair and sign model container
pox sign llama3-int4.pox --key secret_key.ed25519

# Verify tamper-evidence and digital signature
pox verify llama3-int4.pox --public-key public_key.ed25519
```

#### Differential Analysis (`pox diff`)
Evaluate quantization error, mean-squared error (MSE), and tensor drift between baseline and quantized models:
```bash
pox diff fp16_model.pox llama3-int4.pox
```

#### OpenAI-Compatible HTTP Server (`pox serve`)
Host the quantized model locally with Server-Sent Events (SSE) streaming:
```bash
pox serve \
  --model llama3-int4.pox \
  --host 127.0.0.1 \
  --port 8080 \
  --api-key "secret-token"
```
Test with `curl`:
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-int4.pox",
    "messages": [{"role": "user", "content": "Explain zero-copy memory mapping."}],
    "stream": true
  }'
```

---

### Declarative Pipeline (`paraoxidizer.toml`)

Quantization pipelines can be managed declaratively:

```toml
[model]
source = "meta-llama/Llama-3-8B"
output = "models/llama-3-8b-optimized.pox"

[quantization]
format = "int4"
algorithm = "awq"
group_size = 128
outlier_threshold = 3.5

[calibration]
dataset = "coding"
samples = 128
max_seq_len = 2048

[security]
sign = true
private_key_path = "keys/ed25519.sk"
verify_on_load = true

[runtime]
backend = "metal"          # "metal" (Apple Silicon) or "neon" or "cpu"
context_length = 4096
```
Execute with:
```bash
pox quantize --config paraoxidizer.toml
```

---

### Rust Library API

```rust
use paraoxidizer_core::precision::Precision;
use paraoxidizer_format::PoxFile;
use paraoxidizer_runtime::{PoxEngine, SamplerConfig};

fn main() -> anyhow::Result<()> {
    // Zero-copy virtual memory mapping
    let pox = PoxFile::open("llama3-int4.pox")?;
    println!("Loaded model with {} tensors", pox.header.tensor_count);

    // Initialize inference engine (auto-probes Metal GPU or ARM NEON)
    let engine = PoxEngine::new(pox);

    // Streaming autoregressive generation
    let sampler = SamplerConfig::default();
    engine.generate_stream(
        "Explain memory safety in Rust.",
        128,
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

### Python API (`import pox`)

```python
import pox

# Quantize directly from Hugging Face or local path
pox.quantize(
    input_path="meta-llama/Llama-3-8B",
    output_path="llama3-int4.pox",
    algorithm="awq",
    group_size=128,
    outliers=3.5,
)

# Inspect container metadata and verify supply-chain integrity
metadata = pox.inspect("llama3-int4.pox")
print("Tensors:", metadata["tensor_count"])

is_valid = pox.verify("llama3-int4.pox")
assert is_valid, "Tampering detected in model weights!"
```

---

## Benchmarks

Measurements conducted on **Apple Silicon M4** (10-core CPU, 10-core GPU, ARM NEON, unified memory, Darwin arm64) using `cargo run --release -p paraoxidizer-bench --bin pox-bench`:

### 1. Vector Dot-Product Microbenchmarks (ARM NEON vs Scalar)

Evaluates dequantized projection passes across varying vector dimensions:

| Vector Dimension ($N$) | ARM NEON SIMD | Scalar Fallback | SIMD Speedup |
| :--- | :--- | :--- | :--- |
| **$N = 512$** | 0.088 µs (5.83 GB/s) | 0.287 µs (1.78 GB/s) | **3.26x** |
| **$N = 1024$** | 0.175 µs (5.85 GB/s) | 0.584 µs (1.75 GB/s) | **3.34x** |
| **$N = 2048$** | 0.354 µs (5.79 GB/s) | 1.189 µs (1.72 GB/s) | **3.36x** |
| **$N = 4096$** | 0.697 µs (5.88 GB/s) | 2.433 µs (1.68 GB/s) | **3.49x** |
| **$N = 8192$** | 1.488 µs (5.51 GB/s) | 5.250 µs (1.56 GB/s) | **3.53x** |

### 2. INT4 GEMV Throughput (Apple M4 Unified Memory)

Matrix-vector multiplication ($M=1, K=4096, N=4096$) comparing FP16 baseline against ParaOxidizer INT4 group quantization:

| Kernel Variant | Latency | Bandwidth | Memory Footprint | vs FP16 Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **FP16 Baseline** | 227.12 µs | 147.54 GB/s | 33.55 MB | 1.00x |
| **INT4 Group-128 (CPU NEON)** | 88.42 µs | 379.43 GB/s (eff) | 9.44 MB | **2.57x** |
| **INT4 Group-128 (Metal GPU)** | 41.15 µs | 815.31 GB/s (eff) | 9.44 MB | **5.52x** |
| **INT4 Group-64 (Metal GPU)** | 44.20 µs | 759.05 GB/s (eff) | 10.49 MB | **5.14x** |

### 3. Quantization Algorithm Fidelity & Drift (Perplexity Proxy)

Evaluated across 1,000 synthetic calibration activations with heavy-tailed channel perturbations:

| Algorithm | Bit Width | Outlier Separation | Cosine Similarity | MSE Loss | Max Channel Drift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Min-Max Uniform** | INT4 (g128) | None | 0.9942 | $1.82 \times 10^{-3}$ | 0.412 |
| **Min-Max + Outliers** | INT4 (g128) | $\ge 3.5\sigma$ (FP16) | 0.9978 | $4.18 \times 10^{-4}$ | 0.089 |
| **AWQ (Hessian Salience)** | INT4 (g128) | Salient Scaled | 0.9984 | $2.95 \times 10^{-4}$ | 0.061 |
| **GPTQ ($H^{-1}$ Damped)** | INT4 (g128) | Error Compensation | **0.9986** | **$2.41 \times 10^{-4}$** | **0.052** |

### 4. Memory Footprint & Ingestion Scaling

| Model Architecture | Parameter Count | SafeTensors (FP16) | `.pox` INT4 (g128) | Compression Ratio | Cold-Start mmap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama-3-8B** | 8.03 B | 16.06 GB | **4.62 GB** | **3.48x** | 11.2 µs |
| **Mistral-7B-v0.1** | 7.24 B | 14.48 GB | **4.18 GB** | **3.46x** | 10.8 µs |
| **Qwen-2.5-7B** | 7.61 B | 15.22 GB | **4.39 GB** | **3.47x** | 11.0 µs |

---

## Technical Details

### `.pox` File Format Specification

The `.pox` container is structured specifically for zero-copy memory-mapped operations:

```text
┌────────────────────────────────────────────────────────┐
│ Header (128 bytes)                                     │
│  - Magic: "POX\0" (0x50, 0x4F, 0x58, 0x00)             │
│  - Version: u16 (Major=1, Minor=0)                     │
│  - Tensor Count: u32                                   │
│  - Metadata Length: u64                                │
│  - Checksum: [u8; 32] (SHA-256 of header + metadata)   │
│  - Alignment: 64 bytes                                 │
├────────────────────────────────────────────────────────┤
│ Metadata Section (JSON / FlexBuffers)                  │
│  - Architecture specification (Llama, Mistral, Qwen)   │
│  - Quantization config (scales, group size, algorithm) │
│  - Cryptographic signatures (Ed25519)                  │
├────────────────────────────────────────────────────────┤
│ Tensor Index Table                                     │
│  - Tensor Name, Shape, DataType, Group Size            │
│  - 64-byte aligned Payload Offset & Byte Length        │
├────────────────────────────────────────────────────────┤
│ Payload Section (64-byte aligned SIMD buffers)         │
│  - Packed INT4 nibble blocks [q0 | q1]                 │
│  - FP16 scale factors & zero offsets                   │
│  - Sparse outlier coordinate indices                   │
└────────────────────────────────────────────────────────┘
```

### Second-Order Hessian Calibration Math

Standard uniform quantization minimizes the parameter error $\arg\min_{\hat{W}} \|W - \hat{W}\|_F^2$. However, neural network layer degradation is governed by output activation drift:

$$E = \|W X - \hat{W} X\|_2^2 = (W - \hat{W}) X X^T (W - \hat{W})^T$$

ParaOxidizer estimates the empirical Hessian matrix $H$:

$$H = \frac{2}{N} X X^T + \lambda I, \quad \lambda = 0.01 \cdot \text{mean}(\text{diag}(H))$$

1. **AWQ (Activation-Aware Weight Quantization)**:
   Channel importance $s_j$ is derived from average activation magnitudes $s_j = \frac{1}{N} \sum_{i} |X_{i, j}|$. Salient channels ($top \ 1\%$) are scaled by $s^{\alpha}$ ($\alpha \approx 0.5$) before uniform quantization, shielding them from precision collapse.
2. **GPTQ (Generalized Post-Training Quantization)**:
   Inverts the damped Hessian $H^{-1}$ using Cholesky or Gauss-Jordan decomposition. Quantizes weights column-by-column ($w_q = \text{round}(w / \Delta) \cdot \Delta$) and immediately compensates all remaining unquantized weights in row $W$ via:

$$W_{:, >j} \leftarrow W_{:, >j} - \frac{w_j - \hat{w}_j}{[H^{-1}]_{j, j}} \cdot [H^{-1}]_{j, >j}$$

### Apple Silicon Metal GPU Execution

For unified memory GPUs, ParaOxidizer implements a Metal compute pipeline (`crates/paraoxidizer-runtime/src/metal_backend.rs`):
- Allocates unified memory buffers using `MTLResourceStorageModeShared`, avoiding discrete PCIe host-to-device memory copies.
- Unrolls 4-bit nibble decoding directly inside GPU threadgroup memory (`q0 = byte & 0x0F`, `q1 = (byte >> 4) & 0x0F`).
- Fuses scale adjustment `dequant = min_offset + q * scale` with 32-bit floating-point multiply-accumulate operations in parallel across compute units.

### Cryptographic Security Model

- **Static Scanner**: Scans model tensors prior to execution for structural anomalies, out-of-range floating point values (e.g. latent NaNs or Infs), and malicious weight trigger patterns.
- **Merkle Tree Supply-Chain**: Tensor hashes are organized into a binary SHA-256 Merkle tree. Modifying even a single nibble invalidates the root digest.
- **Ed25519 Signatures**: Model weights can be signed by vendors and cryptographically validated on boot. Unsigned or tampered models are rejected prior to execution.

---

## Workspace Structure

| Crate | Path | Description |
| :--- | :--- | :--- |
| `paraoxidizer-core` | `crates/paraoxidizer-core` | Architecture abstractions, tensor representations, hardware probing |
| `paraoxidizer-format` | `crates/paraoxidizer-format` | `.pox` zero-copy binary format, SafeTensors/GGUF/HF ingestion |
| `paraoxidizer-quant` | `crates/paraoxidizer-quant` | Affine INT4 & symmetric INT8 kernels, AWQ/GPTQ dispatch, SIMD dot |
| `paraoxidizer-calibration` | `crates/paraoxidizer-calibration` | Empirical Hessian computation, task profiles, sensitivity analysis |
| `paraoxidizer-optimizer` | `crates/paraoxidizer-optimizer` | Multi-objective Pareto frontier mixed-precision solver |
| `paraoxidizer-security` | `crates/paraoxidizer-security` | SHA-256 Merkle trees, Ed25519 signing/verification, backdoor scan |
| `paraoxidizer-runtime` | `crates/paraoxidizer-runtime` | Transformer decoder, KV cache, tokenizer, Metal GPU & NEON engines |
| `paraoxidizer-serve` | `crates/paraoxidizer-serve` | Axum HTTP server (`/v1/chat/completions` with SSE, `/metrics`) |
| `paraoxidizer-bench` | `crates/paraoxidizer-bench` | Hardware benchmarks (SIMD, GEMV, fidelity, system, throughput) |
| `paraoxidizer-cli` | `crates/paraoxidizer-cli` | Unified CLI binary (`pox` / `paraoxidizer`) |
| `paraoxidizer-py` | `crates/paraoxidizer-py` | Python native C-extension module (`import pox`) |

---

## Author

**Adriteyo Das**
- GitHub: [@tryhard-26](https://github.com/tryhard-26)
- Email: [das.adriteyo26@gmail.com](mailto:das.adriteyo26@gmail.com)

---

## License

This project is licensed under the **GNU General Public License v3.0 or later** ([GPL-3.0-or-later](LICENSE)).
