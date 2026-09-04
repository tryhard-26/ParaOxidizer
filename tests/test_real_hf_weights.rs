use paraoxidizer::cli::commands::{run_quantize, run_validate, run_verify};
use paraoxidizer::format::{HfModel, PoxFile};
use paraoxidizer::quant::kernels::{
    dequantize_int4_group, dequantize_int8_symmetric, quantize_awq, quantize_gptq,
    quantize_int4_group, quantize_int8_symmetric,
};
use paraoxidizer::runtime::{KvCache, PoxEngine};
use tempfile::tempdir;

#[derive(Debug)]
struct RealModelBenchmarkResult {
    model_name: &'static str,
    original_size_bytes: usize,
    int8_size_bytes: usize,
    int4_size_bytes: usize,
    compression_ratio_int4: f64,
    int8_cos_sim: f64,
    int4_cos_sim: f64,
    awq_cos_sim: f64,
    gptq_cos_sim: f64,
    int4_sqnr_db: f64,
    int4_mse: f64,
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

#[test]
fn test_real_huggingface_models_and_weights() {
    println!("\n========================================================================================================================");
    println!(" EVALUATING REAL HUGGING FACE MODELS & WEIGHTS ON HARDWARE (Apple M4)");
    println!("========================================================================================================================");

    let real_repos = [
        ("Llama (Real HF SafeTensors)", "hf-internal-testing/tiny-random-LlamaForCausalLM"),
        ("Qwen2.5 (Real HF SafeTensors)", "yujiepan/qwen2.5-tiny-random"),
        ("Gemma (Real HF SafeTensors)", "yujiepan/gemma-tiny-random"),
    ];

    let tmp = tempdir().unwrap();
    let mut benchmark_results = Vec::new();

    for (name, repo_id) in &real_repos {
        println!("\n---> Loading real model: {} [{}]", name, repo_id);
        let hf_model = HfModel::load(repo_id).expect("Failed to load real HF model repository");
        println!(
            "Loaded {} tensors successfully (Arch: {:?}, Hidden: {}, Layers: {})",
            hf_model.tensors.len(),
            hf_model.model_config.architecture,
            hf_model.model_config.hidden_size,
            hf_model.model_config.num_hidden_layers
        );

        // Calculate total raw parameter count and original bytes
        let mut total_params = 0;
        let mut all_weights = Vec::new();

        for (_, (_, _, floats)) in &hf_model.tensors {
            total_params += floats.len();
            all_weights.extend_from_slice(floats);
        }

        let orig_fp16_bytes = total_params * 2;
        println!("Total Parameters: {} ({:.2} MB in FP16)", total_params, (orig_fp16_bytes as f64) / 1e6);

        // 1. Evaluate INT8 Symmetric Quantization on Real Weights
        let (q_int8, scales_int8) = quantize_int8_symmetric(&all_weights);
        let mut dequant_int8 = vec![0.0f32; total_params];
        dequantize_int8_symmetric(&q_int8, &scales_int8, &mut dequant_int8).unwrap();
        let (_, int8_cos, _) = compute_fidelity(&all_weights, &dequant_int8);
        let int8_bytes = q_int8.len() + scales_int8.len();

        // 2. Evaluate INT4 Group-128 Min-Max Quantization on Real Weights
        let (packed_int4, scales_int4) = quantize_int4_group(&all_weights, 128);
        let mut dequant_int4 = vec![0.0f32; total_params];
        dequantize_int4_group(&packed_int4, &scales_int4, 128, total_params, &mut dequant_int4).unwrap();
        let (int4_mse, int4_cos, int4_sqnr) = compute_fidelity(&all_weights, &dequant_int4);
        let int4_bytes = packed_int4.len() + scales_int4.len();

        // 3. Evaluate AWQ on Real Weights
        let cols = hf_model.model_config.hidden_size.max(64);
        let rows = total_params / cols;
        let mut act_scales = vec![0.0f32; cols];
        for c in 0..cols {
            let mut sum_mag = 0.0f32;
            for r in 0..rows {
                sum_mag += all_weights[r * cols + c].abs();
            }
            act_scales[c] = sum_mag / (rows as f32);
        }
        let (packed_awq, scales_awq) = quantize_awq(&all_weights[0..rows * cols], rows, cols, &act_scales, 128);
        let mut dequant_awq = vec![0.0f32; rows * cols];
        dequantize_int4_group(&packed_awq, &scales_awq, 128, rows * cols, &mut dequant_awq).unwrap();
        let (_, awq_cos, _) = compute_fidelity(&all_weights[0..rows * cols], &dequant_awq);

        // 4. Evaluate GPTQ on Real Weights
        let mut inv_h = vec![0.0f32; cols * cols];
        for i in 0..cols {
            inv_h[i * cols + i] = 1.0;
        }
        let (packed_gptq, scales_gptq) = quantize_gptq(&all_weights[0..rows * cols], rows, cols, &inv_h, 128);
        let mut dequant_gptq = vec![0.0f32; rows * cols];
        dequantize_int4_group(&packed_gptq, &scales_gptq, 128, rows * cols, &mut dequant_gptq).unwrap();
        let (_, gptq_cos, _) = compute_fidelity(&all_weights[0..rows * cols], &dequant_gptq);

        // 5. Test Full Packaging to .pox Container via CLI
        let safe_name = repo_id.replace('/', "_");
        let pox_path = tmp.path().join(format!("{}.pox", safe_name));
        run_quantize(
            repo_id,
            4,
            128,
            "automatic",
            "min-max",
            pox_path.to_str().unwrap(),
        )
        .expect("Failed to quantize real HF model to .pox");

        assert!(pox_path.exists());
        run_validate(pox_path.to_str().unwrap()).expect("Validation failed on real HF .pox file");
        run_verify(pox_path.to_str().unwrap(), None).expect("Verification failed on real HF .pox file");

        // 6. Test Inference Forward Pass on Real Weights
        let pox_file = PoxFile::open(&pox_path).expect("Failed to open .pox file");
        let engine = PoxEngine::new(pox_file);
        let mut kv = KvCache::new(2, 2, 32, 128);
        let logits = engine.forward_token(12, 0, &mut kv).expect("Failed forward token pass on real weights");
        assert!(!logits.is_empty());
        assert!(!logits.iter().any(|v| v.is_nan() || v.is_infinite()), "Logits contained NaN/Inf");

        let compression = (orig_fp16_bytes as f64) / (int4_bytes as f64);

        benchmark_results.push(RealModelBenchmarkResult {
            model_name: name,
            original_size_bytes: orig_fp16_bytes,
            int8_size_bytes: int8_bytes,
            int4_size_bytes: int4_bytes,
            compression_ratio_int4: compression,
            int8_cos_sim: int8_cos,
            int4_cos_sim: int4_cos,
            awq_cos_sim: awq_cos,
            gptq_cos_sim: gptq_cos,
            int4_sqnr_db: int4_sqnr,
            int4_mse,
        });
    }

    println!("\n{:<30} | {:<10} | {:<10} | {:<12} | {:<12} | {:<12} | {:<12} | {:<10}", 
        "HF Model", "FP16 (MB)", "INT4 (MB)", "Compression", "INT8 CosSim", "INT4 CosSim", "AWQ CosSim", "INT4 SQNR");
    println!("{:-<30}-|-{:-<10}-|-{:-<10}-|-{:-<12}-|-{:-<12}-|-{:-<12}-|-{:-<12}-|-{:-<10}", 
        "", "", "", "", "", "", "", "");

    for r in &benchmark_results {
        println!("{:<30} | {:<10.2} | {:<10.2} | {:<12.2}x | {:<12.6} | {:<12.6} | {:<12.6} | {:<10.2}",
            r.model_name,
            (r.original_size_bytes as f64) / 1e6,
            (r.int4_size_bytes as f64) / 1e6,
            r.compression_ratio_int4,
            r.int8_cos_sim,
            r.int4_cos_sim,
            r.awq_cos_sim,
            r.int4_sqnr_db,
        );

        // Strict assertions on real weights:
        assert!(r.int8_cos_sim >= 0.9900, "Real weight INT8 cosine similarity degraded: {}", r.int8_cos_sim);
        assert!(r.int4_cos_sim >= 0.9950, "Real weight INT4 cosine similarity degraded: {}", r.int4_cos_sim);
        assert!(r.awq_cos_sim >= 0.9950, "Real weight AWQ cosine similarity degraded: {}", r.awq_cos_sim);
        assert!(r.compression_ratio_int4 >= 3.3, "Real weight compression ratio too low: {}", r.compression_ratio_int4);
    }

    println!("========================================================================================================================\n");
}
