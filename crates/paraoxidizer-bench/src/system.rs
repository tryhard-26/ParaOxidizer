use paraoxidizer_core::tensor::{DType, QuantGroupSize, Shape};
use paraoxidizer_format::{PoxFile, PoxMetadata, PoxQuantPlanRecord, PoxWriter};
use paraoxidizer_security::signature::KeyPair;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use tempfile::NamedTempFile;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBenchmarkResult {
    pub benchmark_name: String,
    pub payload_size_mb: f64,
    pub latency_ms: f64,
    pub throughput_gb_s: f64,
    pub notes: String,
}

pub fn run_system_benchmarks() -> Vec<SystemBenchmarkResult> {
    let mut results = Vec::new();

    // 1. Zero-Copy mmap vs Eager Read (10MB payload)
    {
        let size = 10 * 1024 * 1024;
        let tmp_file = NamedTempFile::new().unwrap();
        let metadata = PoxMetadata {
            model_config: paraoxidizer_core::arch::ModelConfig {
                architecture: paraoxidizer_core::arch::ModelArchitecture::Llama,
                hidden_size: 4096,
                intermediate_size: 11008,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                num_key_value_heads: 32,
                vocab_size: 32000,
                max_position_embeddings: 4096,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                tie_word_embeddings: false,
                eos_token_id: 2,
                bos_token_id: 1,
            },
            total_parameters: 5_000_000,
            quantized_by: "paraoxidizer-bench".to_string(),
            timestamp_utc: 0,
            original_format: "bench".to_string(),
            base_model_name: "bench-model".to_string(),
        };
        let quant_plan = PoxQuantPlanRecord {
            default_precision: "INT4".to_string(),
            group_size: 128,
            outlier_strategy: "automatic".to_string(),
            layer_assignments: HashMap::new(),
        };

        let mut writer = PoxWriter::new(metadata, quant_plan, "bench-run".to_string());
        let dummy_data = vec![0xABu8; size];
        writer.add_tensor(
            "bench.tensor".to_string(),
            Shape::new(vec![5_000_000]),
            DType::I4,
            QuantGroupSize::G128,
            &dummy_data,
            None,
            None,
        );
        writer.write_to_file(tmp_file.path()).unwrap();

        // Measure zero-copy PoxFile::open (mmap)
        let iters = 50;
        let start_mmap = Instant::now();
        for _ in 0..iters {
            let pox = PoxFile::open(tmp_file.path()).unwrap();
            std::hint::black_box(pox.tensors.len());
        }
        let mmap_latency_ms = (start_mmap.elapsed().as_secs_f64() * 1000.0) / (iters as f64);
        let mmap_bw = ((size as f64) / 1e9) / (mmap_latency_ms * 1e-3);

        results.push(SystemBenchmarkResult {
            benchmark_name: "POX Zero-Copy mmap Load".to_string(),
            payload_size_mb: 10.0,
            latency_ms: mmap_latency_ms,
            throughput_gb_s: mmap_bw,
            notes: "Zero-copy virtual memory mapping of tensor descriptors".to_string(),
        });

        // Measure standard eager file buffer read
        let start_read = Instant::now();
        for _ in 0..iters {
            let mut f = File::open(tmp_file.path()).unwrap();
            let mut buf = Vec::with_capacity(size);
            f.read_to_end(&mut buf).unwrap();
            std::hint::black_box(buf.len());
        }
        let read_latency_ms = (start_read.elapsed().as_secs_f64() * 1000.0) / (iters as f64);
        let read_bw = ((size as f64) / 1e9) / (read_latency_ms * 1e-3);

        results.push(SystemBenchmarkResult {
            benchmark_name: "Eager Buffer Read (Baseline)".to_string(),
            payload_size_mb: 10.0,
            latency_ms: read_latency_ms,
            throughput_gb_s: read_bw,
            notes: "Standard file I/O buffer allocation and byte copy".to_string(),
        });
    }

    // 2. Cryptographic SHA-256 Hashing Throughput (25MB payload)
    {
        let size = 25 * 1024 * 1024;
        let dummy_data = vec![0x42u8; size];

        let iters = 10;
        let start_hash = Instant::now();
        for _ in 0..iters {
            let mut hasher = Sha256::new();
            hasher.update(&dummy_data);
            let out = hasher.finalize();
            std::hint::black_box(out);
        }
        let hash_latency_ms = (start_hash.elapsed().as_secs_f64() * 1000.0) / (iters as f64);
        let hash_bw = ((size as f64) / 1e9) / (hash_latency_ms * 1e-3);

        results.push(SystemBenchmarkResult {
            benchmark_name: "SHA-256 Checksum Hashing".to_string(),
            payload_size_mb: 25.0,
            latency_ms: hash_latency_ms,
            throughput_gb_s: hash_bw,
            notes: "Hardware-accelerated cryptographic SHA-256 pass".to_string(),
        });
    }

    // 3. Ed25519 Signing & Verification
    {
        let keypair = KeyPair::generate();
        let message = [0x77u8; 32];

        let iters = 1000;
        let start_sign = Instant::now();
        for _ in 0..iters {
            let sig = keypair.sign_message(&message);
            std::hint::black_box(sig);
        }
        let sign_us = (start_sign.elapsed().as_micros() as f64) / (iters as f64);

        results.push(SystemBenchmarkResult {
            benchmark_name: "Ed25519 Digital Signing".to_string(),
            payload_size_mb: 0.0,
            latency_ms: sign_us / 1000.0,
            throughput_gb_s: 0.0,
            notes: format!("{:.1} µs per signature", sign_us),
        });

        let sig = keypair.sign_message(&message);
        let pubkey_hex = keypair.public_key_hex();
        let start_verify = Instant::now();
        for _ in 0..iters {
            let ok = paraoxidizer_security::verify_signature_hex(&message, &pubkey_hex, &sig).unwrap();
            std::hint::black_box(ok);
        }
        let verify_us = (start_verify.elapsed().as_micros() as f64) / (iters as f64);

        results.push(SystemBenchmarkResult {
            benchmark_name: "Ed25519 Signature Verification".to_string(),
            payload_size_mb: 0.0,
            latency_ms: verify_us / 1000.0,
            throughput_gb_s: 0.0,
            notes: format!("{:.1} µs per verification", verify_us),
        });
    }

    results
}
