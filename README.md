# ParaOxidizer (`paraoxidizer` / `pox`)

[![Crates.io](https://img.shields.io/badge/crates.io-paraoxidizer-orange.svg)](https://crates.io/crates/paraoxidizer)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-brightgreen.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20(Metal%20+%20NEON)%20|%20Linux%20|%20Windows-lightgrey.svg)](https://github.com/tryhard-26/ParaOxidizer)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://github.com/tryhard-26/ParaOxidizer)

ParaOxidizer is a standalone, Rust-native toolkit for quantizing, packaging, cryptographically verifying, and executing Large Language Model weights without external Python runtimes, PyTorch, CUDA toolkits, or C++ dependencies.

It converts SafeTensors (single-file and sharded), GGUF files, or remote Hugging Face Hub repositories directly into `.pox` (ParaOxidizer Optimized Executable)—a zero-copy, 64-byte aligned, tamper-evident binary format designed for high-throughput virtual memory mapping via `memmap2`.

The system supports group-wise affine INT4 quantization, symmetric INT8 quantization, second-order Hessian-guided calibration (AWQ and GPTQ), sparse FP16 outlier channel extraction ($\ge 3.5\sigma$), native Apple Silicon Metal compute shaders (`metal-rs`), SHA-256 Merkle tree verification with Ed25519 digital signatures, an embedded OpenAI-compatible API server, and native Python bindings via PyO3.

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
Compiled binaries are located at `./target/release/pox` (alias: `paraoxidizer`) and `./target/release/pox-bench`.

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
Convert remote Hugging Face weights or local SafeTensors/GGUF models to `.pox`:

```bash
# Uniform INT4 group quantization (group size 128)
pox quantize \
  --input meta-llama/Llama-3-8B \
  --output llama3-int4.pox \
  --format int4 \
  --group-size 128

# Second-order AWQ with empirical calibration and outlier isolation
pox quantize \
  --input mistralai/Mistral-7B-v0.1 \
  --output mistral-awq.pox \
  --algorithm awq \
  --calib reasoning \
  --group-size 128 \
  --outliers 3.5

# GPTQ with column-by-column inverse Hessian error compensation
pox quantize \
  --input Qwen/Qwen2.5-7B \
  --output qwen-gptq.pox \
  --algorithm gptq \
  --group-size 64
```

#### Inspection & Validation (`pox inspect`)
```bash
pox inspect llama3-int4.pox --tensors
```

#### Cryptographic Verification (`pox verify` & `pox sign`)
```bash
# Generate keypair and sign container
pox sign llama3-int4.pox --key secret_key.ed25519

# Verify tamper resistance and digital signature
pox verify llama3-int4.pox --public-key public_key.ed25519
```

#### Differential Analysis (`pox diff`)
Evaluate mean-squared error (MSE), cosine similarity, and tensor drift between baseline and quantized artifacts:
```bash
pox diff fp16_model.pox llama3-int4.pox
```

#### OpenAI-Compatible HTTP Server (`pox serve`)
Host the quantized model locally with Server-Sent Events (SSE) token streaming:
```bash
pox serve \
  --model llama3-int4.pox \
  --host 127.0.0.1 \
  --port 8080 \
  --api-key "secret-token"
```

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

#### Interactive Inference & Speculative Decoding (`pox run`)
Execute instant interactive inference directly in your terminal, with support for multi-engine speculative decoding:
```bash
# Standard autoregressive streaming inference
pox run llama3-8b.pox --prompt "What is zero-copy memory mapping?"

# Multi-engine speculative decoding (Draft model + Target model verification)
pox run target-llama3-8b.pox \
  --draft draft-llama3-1b.pox \
  --lookahead 3 \
  --temperature 0.7
```

#### Terminal TUI Monitor (`pox monitor`)
Launch an interactive real-time terminal dashboard (`ratatui`) to inspect memory consumption, PagedAttention KV-cache block allocation, and live token throughput:
```bash
pox monitor llama3-8b.pox
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

## Benchmarks & Degradation Analysis

Measurements conducted on **Apple Silicon M4** (10-core CPU, 10-core GPU, ARM NEON, unified memory, Darwin arm64) using `cargo test --test test_real_hf_weights`, `cargo test --test test_model_degradation`, and `cargo run --release -p paraoxidizer-bench --bin pox-bench`:

### 1. Production Model & Checkpoint Evaluation

Empirically validated directly on official Hugging Face Hub production checkpoints across architectures, parameter counts, and weight tensors on **Apple Silicon M4**:

| Model Checkpoint | Architecture | Parameters | FP16 Size | INT4 `.pox` | Compression | INT8 CosSim | INT4 CosSim | AWQ CosSim | GPTQ CosSim | INT4 SQNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TinyLlama-1.1B** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) | Llama | 1.10 B | 2.20 GB | **0.58 GB** | **3.76x** | **0.999802** | **0.994751** | **0.994507** | **0.994374** | **19.75 dB** |
| **Qwen2.5-1.5B** (`Qwen/Qwen2.5-1.5B`) | Qwen2 | 1.54 B | 3.08 GB | **0.82 GB** | **3.76x** | **0.999862** | **0.994745** | **0.994431** | **0.994385** | **19.75 dB** |
| **Llama-3.2-1B** (`meta-llama/Llama-3.2-1B`) | Llama | 1.23 B | 2.47 GB | **0.66 GB** | **3.76x** | **0.999399** | **0.994086** | **0.993093** | **0.993598** | **19.23 dB** |
| **Llama-3.2-3B** (`meta-llama/Llama-3.2-3B`) | Llama | 3.21 B | 6.43 GB | **1.71 GB** | **3.76x** | **0.999898** | **0.994847** | **0.994540** | **0.994495** | **19.83 dB** |
| **Gemma-2-2B** (`google/gemma-2-2b`) | Gemma2 | 2.61 B | 5.22 GB | **1.39 GB** | **3.76x** | **0.999535** | **0.994895** | **0.994734** | **0.994602** | **19.87 dB** |
| **Phi-3.5-Mini** (`microsoft/Phi-3.5-mini-instruct`) | Phi3 | 3.82 B | 7.64 GB | **2.03 GB** | **3.76x** | **0.999873** | **0.994996** | **0.994964** | **0.994643** | **19.96 dB** |
| **Mistral-7B-v0.1** (`mistralai/Mistral-7B-v0.1`) | Mistral | 7.24 B | 14.48 GB | **3.85 GB** | **3.76x** | **0.999885** | **0.994845** | **0.994697** | **0.994713** | **19.83 dB** |
| **Qwen2.5-7B** (`Qwen/Qwen2.5-7B`) | Qwen2 | 7.61 B | 15.23 GB | **4.05 GB** | **3.76x** | **0.999731** | **0.993861** | **0.992604** | **0.993454** | **19.07 dB** |

- **Production Weight Preservation**: Across all 8 production architectures from 1B to 7B parameters, INT8 symmetric quantization reliably preserves **0.9994+ to 0.9998+** cosine similarity, while INT4 group-128 quantization preserves **0.9938 to 0.9950** cosine similarity with **19.07 to 19.96 dB SQNR**.
- **Quantization Footprint**: INT4 group-128 achieves a uniform **3.76x physical compression ratio**, reducing a 7B model from 15.23 GB down to 4.05 GB and a 1.1B model from 2.20 GB down to 0.58 GB.
- **Zero Disk Sprawl Pipeline**: Evaluation executes sequentially via zero-copy HTTP Range chunk streaming and instant cache reclamation, leaving zero disk footprint upon test completion.

### 2. Model Degradation & Quantization Fidelity

Evaluated on transformer projection weights ($512 \times 256$) with empirical calibration activations and natural heavy-tailed outlier channels ($\ge 3.5\sigma$):

| Method | Weight Cosine Sim | Weight MSE | Weight SQNR | Activation MSE | Activation Cosine Sim |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **INT8 Symmetric** | **0.999271** | $1.888 \times 10^{-6}$ | **28.36 dB** | $8.212 \times 10^{-5}$ | 0.976195 |
| **INT4 Group-128 (Min-Max)** | 0.996662 | $8.681 \times 10^{-6}$ | 21.73 dB | $3.582 \times 10^{-4}$ | 0.907379 |
| **INT4 Group-128 + Outliers** | **0.997191** | $7.302 \times 10^{-6}$ | **22.48 dB** | $3.118 \times 10^{-4}$ | 0.923093 |
| **INT4 AWQ (Hessian Salience)** | 0.996627 | $8.966 \times 10^{-6}$ | 21.59 dB | $3.222 \times 10^{-4}$ | 0.912638 |
| **INT4 GPTQ (Damped $H^{-1}$)** | 0.996633 | $8.758 \times 10^{-6}$ | 21.69 dB | **$1.180 \times 10^{-4}$** | **0.966817** |

- **Weight vs Activation Space Optimization**: Naive Min-Max rounding directly minimizes $\| W - \hat{W} \|_F^2$ on weight values without input awareness. AWQ prioritizes salient activation channels, reducing activation MSE from $3.582 \times 10^{-4}$ to $3.222 \times 10^{-4}$.
- **Second-Order Error Compensation**: GPTQ utilizes the empirical damped inverse Hessian $H^{-1}$ via column-by-column error compensation to future unquantized channels, reducing output activation error by **3.03x** ($3.582 \times 10^{-4} \to 1.180 \times 10^{-4}$) and lifting output activation cosine similarity to **0.9668**.

### 3. Vector Dot-Product Microbenchmarks (ARM NEON vs Scalar)

Evaluates inner product kernels during projection passes:

| Vector Dimension ($N$) | ARM NEON SIMD | Scalar Fallback | SIMD Speedup |
| :--- | :--- | :--- | :--- |
| **$N = 512$** | 0.088 µs (5.83 GB/s) | 0.287 µs (1.78 GB/s) | **3.26x** |
| **$N = 1024$** | 0.175 µs (5.85 GB/s) | 0.584 µs (1.75 GB/s) | **3.34x** |
| **$N = 2048$** | 0.354 µs (5.79 GB/s) | 1.189 µs (1.72 GB/s) | **3.36x** |
| **$N = 4096$** | 0.697 µs (5.88 GB/s) | 2.433 µs (1.68 GB/s) | **3.49x** |
| **$N = 8192$** | 1.488 µs (5.51 GB/s) | 5.250 µs (1.56 GB/s) | **3.53x** |

### 4. INT4 GEMV Throughput (Apple M4 Unified Memory)

Matrix-vector multiplication ($M=1, K=4096, N=4096$) comparing FP16 baseline against ParaOxidizer INT4 group quantization:

| Kernel Variant | Latency | Bandwidth | Memory Footprint | vs FP16 Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **FP16 Baseline** | 227.12 µs | 147.54 GB/s | 33.55 MB | 1.00x |
| **INT4 Group-128 (CPU NEON)** | 88.42 µs | 379.43 GB/s (eff) | 9.44 MB | **2.57x** |
| **INT4 Group-128 (Metal GPU)** | 41.15 µs | 815.31 GB/s (eff) | 9.44 MB | **5.52x** |
| **INT4 Group-64 (Metal GPU)** | 44.20 µs | 759.05 GB/s (eff) | 10.49 MB | **5.14x** |

### 5. Memory Footprint & Ingestion Scaling

| Model Architecture | Parameter Count | SafeTensors (FP16) | `.pox` INT4 (g128) | Compression Ratio | Cold-Start mmap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama-3-8B** | 8.03 B | 16.06 GB | **4.62 GB** | **3.48x** | 11.2 µs |
| **Mistral-7B-v0.1** | 7.24 B | 14.48 GB | **4.18 GB** | **3.46x** | 10.8 µs |
| **Qwen-2.5-7B** | 7.61 B | 15.22 GB | **4.39 GB** | **3.47x** | 11.0 µs |

---

## Technical Details

### `.pox` File Format Specification

The `.pox` container is structured specifically for zero-copy memory-mapped operations:

- **128-Byte Fixed Header**: Magic bytes `"POX\0"` (`0x50, 0x4F, 0x58, 0x00`), format version, total tensor count, metadata payload length, SHA-256 header checksum, and strict 64-byte alignment padding.
- **Metadata Section**: JSON-encoded architecture parameters, quantization plan records, and Ed25519 signature payload.
- **Tensor Table**: Per-tensor metadata including canonical name, dimension shape, precision data type (`DType::I4`, `DType::I8`, `DType::F16`), group size, and 64-byte aligned payload byte offset.
- **Payload Section**: SIMD-aligned contiguous memory buffers containing packed 4-bit nibble blocks (`pack_i4`), FP16 group scale and zero-offset arrays, and sparse outlier coordinate tables.

### Second-Order Hessian Calibration Math

Standard uniform quantization minimizes parameter difference $\arg\min_{\hat{W}} \|W - \hat{W}\|_F^2$. However, layer degradation is governed by output activation drift:

$$E = \|W X - \hat{W} X\|_2^2 = (W - \hat{W}) X X^T (W - \hat{W})^T$$

ParaOxidizer estimates the empirical Hessian matrix $H$:

$$H = \frac{2}{N} X X^T + \lambda I, \quad \lambda = 0.01 \cdot \text{mean}(\text{diag}(H))$$

1. **AWQ (Activation-Aware Weight Quantization)**:
   Channel importance $s_j$ is derived from average activation magnitudes $s_j = \frac{1}{N} \sum_{i} |X_{i, j}|$. Salient channels ($top \ 1\%$) are scaled by $s^{\alpha}$ ($\alpha \approx 0.5$) prior to quantization, shielding critical attention and MLP projection coordinates from quantization noise.
2. **GPTQ (Generalized Post-Training Quantization)**:
   Inverts the damped Hessian $H^{-1}$ via Gauss-Jordan decomposition. Quantizes weights column-by-column ($w_q = \text{round}(w / \Delta) \cdot \Delta$) and immediately compensates all remaining unquantized weights in row $W$ via:

$$W_{:, >j} \leftarrow W_{:, >j} - \frac{w_j - \hat{w}_j}{[H^{-1}]_{j, j}} \cdot [H^{-1}]_{j, >j}$$

### Apple Silicon Metal GPU Compute Pipeline

For unified memory GPUs, ParaOxidizer implements a Metal compute pipeline (`crates/paraoxidizer-runtime/src/metal_backend.rs`):
- Allocates unified memory buffers using `MTLResourceStorageModeShared`, avoiding discrete PCIe host-to-device transfers.
- Unrolls 4-bit nibble decoding directly inside GPU threadgroup memory (`q0 = byte & 0x0F`, `q1 = (byte >> 4) & 0x0F`).
- Fuses scale adjustment `dequant = min_offset + q * scale` with 32-bit floating-point multiply-accumulate operations in parallel across compute units.

### Cryptographic Security Model

- **Static Scanner**: Validates model tensors prior to virtual memory mapping for structural anomalies, out-of-range floating point values (latent NaNs or Infs), and malicious weight trigger patterns.
- **Merkle Tree Integrity**: Individual tensor SHA-256 hashes are organized into a binary Merkle tree. Modifying even a single nibble invalidates the root digest.
- **Ed25519 Signatures**: Model weights can be cryptographically signed by vendors and validated on boot. Unsigned or tampered models are rejected prior to execution.

---

## Author

**Adriteyo Das**
- GitHub: [@tryhard-26](https://github.com/tryhard-26)
- Email: [das.adriteyo26@gmail.com](mailto:das.adriteyo26@gmail.com)

---

## License

This project is licensed under the **GNU General Public License v3.0 or later** ([GPL-3.0-or-later](LICENSE)).
