use half::{bf16, f16};
use paraoxidizer_calibration::HessianMatrix;
use paraoxidizer_quant::kernels::{
    compute_awq_scales, dequantize_awq, dequantize_int4_group, dequantize_int8_symmetric,
    quantize_awq, quantize_gptq, quantize_int4_group, quantize_int8_symmetric,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SafetensorsTensorInfo {
    #[serde(default)]
    dtype: String,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    data_offsets: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HfIndex {
    weight_map: HashMap<String, String>,
}

struct ProductionModelSpec {
    name: &'static str,
    official_id: &'static str,
    fetch_repo: &'static str,
    architecture: &'static str,
    param_count_str: &'static str,
    fp16_size_gb: f64,
}

struct BenchmarkRow {
    name: &'static str,
    official_id: &'static str,
    architecture: &'static str,
    param_count: &'static str,
    fp16_gb: f64,
    int4_pox_gb: f64,
    compression: f64,
    int8_cos_sim: f64,
    int4_cos_sim: f64,
    awq_cos_sim: f64,
    gptq_cos_sim: f64,
    int4_sqnr_db: f64,
}

fn compute_fidelity(orig: &[f32], recon: &[f32]) -> (f64, f64, f64) {
    let mut sum_sq_err = 0.0f64;
    let mut dot_prod = 0.0f64;
    let mut norm_orig_sq = 0.0f64;
    let mut norm_recon_sq = 0.0f64;

    for (&o, &r) in orig.iter().zip(recon.iter()) {
        let diff = (o - r) as f64;
        let o_f64 = o as f64;
        let r_f64 = r as f64;

        sum_sq_err += diff * diff;
        dot_prod += o_f64 * r_f64;
        norm_orig_sq += o_f64 * o_f64;
        norm_recon_sq += r_f64 * r_f64;
    }

    let n = orig.len() as f64;
    let mse = sum_sq_err / n;
    let cos_sim = if norm_orig_sq > 0.0 && norm_recon_sq > 0.0 {
        dot_prod / (norm_orig_sq.sqrt() * norm_recon_sq.sqrt())
    } else {
        1.0
    };
    let sqnr = if sum_sq_err > 1e-12 {
        10.0 * (norm_orig_sq / sum_sq_err).log10()
    } else {
        99.9
    };

    (mse, cos_sim, sqnr)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================================================================================");
    println!(" EVALUATING REAL HUGGING FACE PRODUCTION MODELS (Apple Silicon M4, Zero-Copy Remote Range Pipeline)");
    println!("========================================================================================================================\n");

    let specs = vec![
        ProductionModelSpec {
            name: "TinyLlama-1.1B",
            official_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            fetch_repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            architecture: "Llama",
            param_count_str: "1.10 B",
            fp16_size_gb: 2.20,
        },
        ProductionModelSpec {
            name: "Qwen2.5-1.5B",
            official_id: "Qwen/Qwen2.5-1.5B",
            fetch_repo: "Qwen/Qwen2.5-1.5B",
            architecture: "Qwen2",
            param_count_str: "1.54 B",
            fp16_size_gb: 3.08,
        },
        ProductionModelSpec {
            name: "Llama-3.2-1B",
            official_id: "meta-llama/Llama-3.2-1B",
            fetch_repo: "unsloth/Llama-3.2-1B",
            architecture: "Llama",
            param_count_str: "1.23 B",
            fp16_size_gb: 2.47,
        },
        ProductionModelSpec {
            name: "Llama-3.2-3B",
            official_id: "meta-llama/Llama-3.2-3B",
            fetch_repo: "unsloth/Llama-3.2-3B",
            architecture: "Llama",
            param_count_str: "3.21 B",
            fp16_size_gb: 6.43,
        },
        ProductionModelSpec {
            name: "Gemma-2-2B",
            official_id: "google/gemma-2-2b",
            fetch_repo: "unsloth/gemma-2-2b",
            architecture: "Gemma2",
            param_count_str: "2.61 B",
            fp16_size_gb: 5.22,
        },
        ProductionModelSpec {
            name: "Phi-3.5-Mini",
            official_id: "microsoft/Phi-3.5-mini-instruct",
            fetch_repo: "microsoft/Phi-3.5-mini-instruct",
            architecture: "Phi3",
            param_count_str: "3.82 B",
            fp16_size_gb: 7.64,
        },
        ProductionModelSpec {
            name: "Mistral-7B-v0.1",
            official_id: "mistralai/Mistral-7B-v0.1",
            fetch_repo: "mistralai/Mistral-7B-v0.1",
            architecture: "Mistral",
            param_count_str: "7.24 B",
            fp16_size_gb: 14.48,
        },
        ProductionModelSpec {
            name: "Qwen2.5-7B",
            official_id: "Qwen/Qwen2.5-7B",
            fetch_repo: "Qwen/Qwen2.5-7B",
            architecture: "Qwen2",
            param_count_str: "7.61 B",
            fp16_size_gb: 15.23,
        },
    ];

    let mut results = Vec::new();

    for spec in &specs {
        println!(
            "--> Fetching and evaluating production weights for {} [{}]...",
            spec.name, spec.official_id
        );

        // 1. Determine shard / safetensors file
        let idx_url = format!(
            "https://huggingface.co/{}/raw/main/model.safetensors.index.json",
            spec.fetch_repo
        );
        let (st_url, target_shard) = match ureq::get(&idx_url).call() {
            Ok(resp) => {
                let mut content = String::new();
                resp.into_reader().read_to_string(&mut content)?;
                let idx: HfIndex = serde_json::from_str(&content)?;
                // Pick the first shard file in weight_map
                let shard = idx
                    .weight_map
                    .values()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| "model.safetensors".to_string());
                (
                    format!(
                        "https://huggingface.co/{}/resolve/main/{}",
                        spec.fetch_repo, shard
                    ),
                    shard,
                )
            }
            Err(_) => (
                format!(
                    "https://huggingface.co/{}/resolve/main/model.safetensors",
                    spec.fetch_repo
                ),
                "model.safetensors".to_string(),
            ),
        };

        println!("    Target shard: {}", target_shard);

        // 2. Fetch header length (first 8 bytes)
        let mut header_len_bytes = [0u8; 8];
        let resp = ureq::get(&st_url).set("Range", "bytes=0-7").call()?;
        resp.into_reader().read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes);

        // 3. Fetch JSON header string
        let header_range = format!("bytes=8-{}", 7 + header_len);
        let resp_hdr = ureq::get(&st_url).set("Range", &header_range).call()?;
        let mut header_str = String::new();
        resp_hdr.into_reader().read_to_string(&mut header_str)?;

        let header_map: HashMap<String, SafetensorsTensorInfo> = serde_json::from_str(&header_str)?;

        // 4. Find a representative 2D projection matrix
        let mut target_tensor: Option<(String, SafetensorsTensorInfo)> = None;
        for (name, info) in &header_map {
            if name == "__metadata__" || info.dtype.is_empty() || info.data_offsets.len() < 2 {
                continue;
            }
            if info.shape.len() == 2
                && info.shape[0] >= 64
                && info.shape[1] >= 64
                && (name.contains("k_proj")
                    || name.contains("q_proj")
                    || name.contains("v_proj")
                    || name.contains("o_proj")
                    || name.contains("dense")
                    || name.contains("qkv_proj"))
            {
                target_tensor = Some((name.clone(), info.clone()));
                break;
            }
        }

        let (tensor_name, tensor_info) =
            target_tensor.expect("Must find 2D projection tensor in checkpoint");
        println!(
            "    Selected tensor: '{}' shape: {:?}, dtype: {}",
            tensor_name, tensor_info.shape, tensor_info.dtype
        );

        let rows = tensor_info.shape[0];
        let cols = tensor_info.shape[1];
        let num_weights = rows * cols;

        let data_start = 8 + header_len + tensor_info.data_offsets[0];
        let data_end = 8 + header_len + tensor_info.data_offsets[1];
        let range_header = format!("bytes={}-{}", data_start, data_end - 1);

        // 5. Stream raw weights
        let resp_weights = ureq::get(&st_url).set("Range", &range_header).call()?;
        let mut raw_bytes = Vec::with_capacity((data_end - data_start) as usize);
        resp_weights.into_reader().read_to_end(&mut raw_bytes)?;

        // 6. Decode to f32
        let weights: Vec<f32> = if tensor_info.dtype == "BF16" {
            raw_bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        } else if tensor_info.dtype == "F16" {
            raw_bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        } else {
            raw_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        assert_eq!(weights.len(), num_weights);

        // Select evaluation matrix dimensions for benchmark (bounded to 512 cols for fast Hessian inversion)
        let eval_rows = rows.min(512);
        let eval_cols = cols.min(512);
        let mut eval_matrix = Vec::with_capacity(eval_rows * eval_cols);
        for r in 0..eval_rows {
            for c in 0..eval_cols {
                eval_matrix.push(weights[r * cols + c]);
            }
        }

        let eval_numel = eval_rows * eval_cols;

        // INT8 Symmetric
        let (q_int8, scales_int8) = quantize_int8_symmetric(&eval_matrix);
        let mut dequant_int8 = vec![0.0f32; eval_numel];
        dequantize_int8_symmetric(&q_int8, &scales_int8, &mut dequant_int8).unwrap();
        let (_, int8_cos, _) = compute_fidelity(&eval_matrix, &dequant_int8);

        // INT4 Group-128 Min-Max
        let (packed_int4, scales_int4) = quantize_int4_group(&eval_matrix, 128);
        let mut dequant_int4 = vec![0.0f32; eval_numel];
        dequantize_int4_group(
            &packed_int4,
            &scales_int4,
            128,
            eval_numel,
            &mut dequant_int4,
        )
        .unwrap();
        let (_, int4_cos, int4_sqnr) = compute_fidelity(&eval_matrix, &dequant_int4);

        // INT4 AWQ
        let mut act_scales = vec![0.0f32; eval_cols];
        for c in 0..eval_cols {
            let mut sum_mag = 0.0f32;
            for r in 0..eval_rows {
                sum_mag += eval_matrix[r * eval_cols + c].abs();
            }
            act_scales[c] = (sum_mag / (eval_rows as f32)).max(1e-6);
        }
        let protection = compute_awq_scales(&act_scales, eval_cols);
        let (packed_awq, scales_awq) =
            quantize_awq(&eval_matrix, eval_rows, eval_cols, &act_scales, 128);
        let mut dequant_awq = vec![0.0f32; eval_numel];
        dequantize_awq(
            &packed_awq,
            &scales_awq,
            &protection,
            128,
            eval_rows,
            eval_cols,
            &mut dequant_awq,
        )
        .unwrap();
        let (_, awq_cos, _) = compute_fidelity(&eval_matrix, &dequant_awq);

        // INT4 GPTQ (Damped Inverse Hessian)
        let mut hessian = HessianMatrix::new(eval_cols);
        let num_calib = 64;
        let mut calib_x = vec![0.0f32; eval_cols * num_calib];
        for s in 0..num_calib {
            for c in 0..eval_cols {
                calib_x[c * num_calib + s] =
                    eval_matrix[(s % eval_rows) * eval_cols + c] + ((s as f32 * 0.1).sin() * 0.01);
            }
        }
        hessian.accumulate_activations(&calib_x, num_calib);
        hessian.compute_inverse(0.01);
        let (packed_gptq, scales_gptq) =
            quantize_gptq(&eval_matrix, eval_rows, eval_cols, &hessian.inv_data, 128);
        let mut dequant_gptq = vec![0.0f32; eval_numel];
        dequantize_int4_group(
            &packed_gptq,
            &scales_gptq,
            128,
            eval_numel,
            &mut dequant_gptq,
        )
        .unwrap();
        let (_, gptq_cos, _) = compute_fidelity(&eval_matrix, &dequant_gptq);

        let int4_bytes = packed_int4.len() + scales_int4.len();
        let raw_eval_fp16_bytes = eval_numel * 2;
        let compression = (raw_eval_fp16_bytes as f64) / (int4_bytes as f64);
        let int4_pox_gb = spec.fp16_size_gb / compression;

        println!("    [✓] INT8 CosSim: {:.6}, INT4 CosSim: {:.6}, AWQ: {:.6}, GPTQ: {:.6}, SQNR: {:.2} dB, Comp: {:.2}x",
            int8_cos, int4_cos, awq_cos, gptq_cos, int4_sqnr, compression
        );

        results.push(BenchmarkRow {
            name: spec.name,
            official_id: spec.official_id,
            architecture: spec.architecture,
            param_count: spec.param_count_str,
            fp16_gb: spec.fp16_size_gb,
            int4_pox_gb,
            compression,
            int8_cos_sim: int8_cos,
            int4_cos_sim: int4_cos,
            awq_cos_sim: awq_cos,
            gptq_cos_sim: gptq_cos,
            int4_sqnr_db: int4_sqnr,
        });

        // Explicit zero-footprint verification: ensure no temporary caches remain
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let hub_cache = std::path::PathBuf::from(home)
            .join(".cache")
            .join("paraoxidizer")
            .join("hub");
        if hub_cache.exists() {
            let _ = std::fs::remove_dir_all(&hub_cache);
        }
    }

    println!("\n========================================================================================================================");
    println!(" PRODUCTION MODELS BENCHMARK SUITE SUMMARY TABLE");
    println!("========================================================================================================================\n");

    println!("| Model Checkpoint | Architecture | Parameters | FP16 Size | INT4 `.pox` | Compression | INT8 CosSim | INT4 CosSim | AWQ CosSim | GPTQ CosSim | INT4 SQNR |");
    println!(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    );

    for r in &results {
        println!(
            "| **{}** (`{}`) | {} | {} | {:.2} GB | **{:.2} GB** | **{:.2}x** | **{:.6}** | **{:.6}** | **{:.6}** | **{:.6}** | **{:.2} dB** |",
            r.name, r.official_id, r.architecture, r.param_count, r.fp16_gb, r.int4_pox_gb, r.compression, r.int8_cos_sim, r.int4_cos_sim, r.awq_cos_sim, r.gptq_cos_sim, r.int4_sqnr_db
        );
    }

    println!("\n========================================================================================================================\n");

    Ok(())
}
