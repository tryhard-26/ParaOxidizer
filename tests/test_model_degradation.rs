mod common;

use common::create_hf_llama_model;
use paraoxidizer::cli::commands::run_quantize;
use paraoxidizer::format::PoxFile;
use paraoxidizer::quant::kernels::{
    dequantize_int4_group, dequantize_int8_symmetric, quantize_awq, quantize_gptq,
    quantize_int4_group, quantize_int8_symmetric,
};
use paraoxidizer::quant::outlier::{OutlierPolicy, SparseOutlierTable};
use paraoxidizer::runtime::{KvCache, PoxEngine};
use tempfile::tempdir;

#[derive(Debug)]
struct FidelityMetrics {
    method: &'static str,
    weight_mse: f64,
    weight_cos_sim: f64,
    weight_sqnr_db: f64,
    act_mse: f64,
    act_cos_sim: f64,
}

fn compute_metrics(orig: &[f32], recon: &[f32]) -> (f64, f64, f64, f64) {
    let mut sum_sq_err = 0.0f64;
    let mut sum_abs_err = 0.0f64;
    let mut dot_prod = 0.0f64;
    let mut norm_orig_sq = 0.0f64;
    let mut norm_recon_sq = 0.0f64;

    for (&o, &r) in orig.iter().zip(recon.iter()) {
        let diff = (o - r).abs() as f64;
        let o_f64 = o as f64;
        let r_f64 = r as f64;

        sum_sq_err += diff * diff;
        sum_abs_err += diff;
        dot_prod += o_f64 * r_f64;
        norm_orig_sq += o_f64 * o_f64;
        norm_recon_sq += r_f64 * r_f64;
    }

    let n = orig.len() as f64;
    let mse = sum_sq_err / n;
    let mae = sum_abs_err / n;
    let cosine_sim = if norm_orig_sq > 0.0 && norm_recon_sq > 0.0 {
        dot_prod / (norm_orig_sq.sqrt() * norm_recon_sq.sqrt())
    } else {
        1.0
    };
    let sqnr_db = if sum_sq_err > 1e-12 {
        10.0 * (norm_orig_sq / sum_sq_err).log10()
    } else {
        99.9
    };

    (mse, mae, cosine_sim, sqnr_db)
}

#[test]
fn test_layer_quantization_degradation() {
    println!("\n==========================================================================================");
    println!(" EVALUATING LAYER QUANTIZATION FIDELITY & DEGRADATION (Apple M4)");
    println!("==========================================================================================");

    let rows = 512;
    let cols = 256;
    let numel = rows * cols;

    // Synthesize realistic transformer projection weights (hidden_size=256 -> intermediate=512)
    let mut weights = Vec::with_capacity(numel);
    for i in 0..numel {
        let u1 = ((i as f64 * 31415.92).sin().abs() * 0.999 + 0.0001).min(0.9999);
        let u2 = ((i as f64 * 27182.81).cos().abs() * 0.999 + 0.0001).min(0.9999);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let mut val = (z * 0.04) as f32; // Gaussian std dev = 0.04
        if i % 1500 == 0 {
            val *= 3.5; // Natural heavy-tailed outlier channel (~3.5 std dev)
        }
        weights.push(val);
    }

    // Input activation vectors representing realistic batch passes
    let num_tokens = 64;
    let mut activations = Vec::with_capacity(num_tokens * cols);
    for i in 0..(num_tokens * cols) {
        let val = ((i as f32 * 0.37).sin() * 0.5) + ((i as f32 * 0.11).cos() * 0.2);
        activations.push(val);
    }

    // Baseline FP16 matrix-vector multiplication Y = W * X
    let mut baseline_y = vec![0.0f32; num_tokens * rows];
    for t in 0..num_tokens {
        let x_offset = t * cols;
        let y_offset = t * rows;
        for r in 0..rows {
            let mut sum = 0.0f32;
            let w_offset = r * cols;
            for c in 0..cols {
                sum += weights[w_offset + c] * activations[x_offset + c];
            }
            baseline_y[y_offset + r] = sum;
        }
    }

    let mut metrics_table = Vec::new();

    // 1. INT8 Symmetric
    {
        let (q_weights, scales) = quantize_int8_symmetric(&weights);
        let mut dequant = vec![0.0f32; numel];
        dequantize_int8_symmetric(&q_weights, &scales, &mut dequant).unwrap();

        let (w_mse, _, w_cos, w_sqnr) = compute_metrics(&weights, &dequant);

        let mut test_y = vec![0.0f32; num_tokens * rows];
        for t in 0..num_tokens {
            let x_offset = t * cols;
            let y_offset = t * rows;
            for r in 0..rows {
                let mut sum = 0.0f32;
                let w_offset = r * cols;
                for c in 0..cols {
                    sum += dequant[w_offset + c] * activations[x_offset + c];
                }
                test_y[y_offset + r] = sum;
            }
        }
        let (act_mse, _, act_cos, _) = compute_metrics(&baseline_y, &test_y);

        metrics_table.push(FidelityMetrics {
            method: "INT8 Symmetric",
            weight_mse: w_mse,
            weight_cos_sim: w_cos,
            weight_sqnr_db: w_sqnr,
            act_mse,
            act_cos_sim: act_cos,
        });
    }

    // 2. INT4 Group-128 (Standard Min-Max)
    {
        let (packed, scales) = quantize_int4_group(&weights, 128);
        let mut dequant = vec![0.0f32; numel];
        dequantize_int4_group(&packed, &scales, 128, numel, &mut dequant).unwrap();

        let (w_mse, _, w_cos, w_sqnr) = compute_metrics(&weights, &dequant);

        let mut test_y = vec![0.0f32; num_tokens * rows];
        for t in 0..num_tokens {
            let x_offset = t * cols;
            let y_offset = t * rows;
            for r in 0..rows {
                let mut sum = 0.0f32;
                let w_offset = r * cols;
                for c in 0..cols {
                    sum += dequant[w_offset + c] * activations[x_offset + c];
                }
                test_y[y_offset + r] = sum;
            }
        }
        let (act_mse, _, act_cos, _) = compute_metrics(&baseline_y, &test_y);

        metrics_table.push(FidelityMetrics {
            method: "INT4 Group-128 (Min-Max)",
            weight_mse: w_mse,
            weight_cos_sim: w_cos,
            weight_sqnr_db: w_sqnr,
            act_mse,
            act_cos_sim: act_cos,
        });
    }

    // 3. INT4 Group-128 + Sparse Outliers (>= 3.5 std dev)
    {
        let mut weights_copy = weights.clone();
        let outliers = SparseOutlierTable::extract_and_zero_outliers(&mut weights_copy, OutlierPolicy::Automatic)
            .unwrap_or_default();
        let (packed, scales) = quantize_int4_group(&weights_copy, 128);
        let mut dequant = vec![0.0f32; numel];
        dequantize_int4_group(&packed, &scales, 128, numel, &mut dequant).unwrap();
        for (&idx, &val) in outliers.indices.iter().zip(outliers.values.iter()) {
            dequant[idx as usize] = val.to_f32();
        }

        let (w_mse, _, w_cos, w_sqnr) = compute_metrics(&weights, &dequant);

        let mut test_y = vec![0.0f32; num_tokens * rows];
        for t in 0..num_tokens {
            let x_offset = t * cols;
            let y_offset = t * rows;
            for r in 0..rows {
                let mut sum = 0.0f32;
                let w_offset = r * cols;
                for c in 0..cols {
                    sum += dequant[w_offset + c] * activations[x_offset + c];
                }
                test_y[y_offset + r] = sum;
            }
        }
        let (act_mse, _, act_cos, _) = compute_metrics(&baseline_y, &test_y);

        metrics_table.push(FidelityMetrics {
            method: "INT4 Group-128 + Outliers",
            weight_mse: w_mse,
            weight_cos_sim: w_cos,
            weight_sqnr_db: w_sqnr,
            act_mse,
            act_cos_sim: act_cos,
        });
    }

    // 4. INT4 AWQ (Activation-Aware Weight Quantization)
    {
        let mut act_scales = vec![0.0f32; cols];
        for c in 0..cols {
            let mut sum_mag = 0.0f32;
            for t in 0..num_tokens {
                sum_mag += activations[t * cols + c].abs();
            }
            act_scales[c] = sum_mag / (num_tokens as f32);
        }

        let (packed, scales) = quantize_awq(&weights, rows, cols, &act_scales, 128);
        let mut dequant = vec![0.0f32; numel];
        dequantize_int4_group(&packed, &scales, 128, numel, &mut dequant).unwrap();

        let (w_mse, _, w_cos, w_sqnr) = compute_metrics(&weights, &dequant);

        let mut test_y = vec![0.0f32; num_tokens * rows];
        for t in 0..num_tokens {
            let x_offset = t * cols;
            let y_offset = t * rows;
            for r in 0..rows {
                let mut sum = 0.0f32;
                let w_offset = r * cols;
                for c in 0..cols {
                    sum += dequant[w_offset + c] * activations[x_offset + c];
                }
                test_y[y_offset + r] = sum;
            }
        }
        let (act_mse, _, act_cos, _) = compute_metrics(&baseline_y, &test_y);

        metrics_table.push(FidelityMetrics {
            method: "INT4 AWQ (Hessian Salience)",
            weight_mse: w_mse,
            weight_cos_sim: w_cos,
            weight_sqnr_db: w_sqnr,
            act_mse,
            act_cos_sim: act_cos,
        });
    }

    // 5. INT4 GPTQ (Optimal Brain Quantization with H^-1 error compensation)
    {
        let mut hessian = vec![0.0f32; cols * cols];
        for t in 0..num_tokens {
            let x_tok = &activations[t * cols..(t + 1) * cols];
            for i in 0..cols {
                for j in 0..cols {
                    hessian[i * cols + j] += x_tok[i] * x_tok[j];
                }
            }
        }
        for v in &mut hessian {
            *v *= 2.0 / (num_tokens as f32);
        }
        let mut inv_h = vec![0.0f32; cols * cols];
        for i in 0..cols {
            inv_h[i * cols + i] = 1.0 / (hessian[i * cols + i] + 0.01);
        }

        let (packed, scales) = quantize_gptq(&weights, rows, cols, &inv_h, 128);
        let mut dequant = vec![0.0f32; numel];
        dequantize_int4_group(&packed, &scales, 128, numel, &mut dequant).unwrap();

        let (w_mse, _, w_cos, w_sqnr) = compute_metrics(&weights, &dequant);

        let mut test_y = vec![0.0f32; num_tokens * rows];
        for t in 0..num_tokens {
            let x_offset = t * cols;
            let y_offset = t * rows;
            for r in 0..rows {
                let mut sum = 0.0f32;
                let w_offset = r * cols;
                for c in 0..cols {
                    sum += dequant[w_offset + c] * activations[x_offset + c];
                }
                test_y[y_offset + r] = sum;
            }
        }
        let (act_mse, _, act_cos, _) = compute_metrics(&baseline_y, &test_y);

        metrics_table.push(FidelityMetrics {
            method: "INT4 GPTQ (Damped H^-1)",
            weight_mse: w_mse,
            weight_cos_sim: w_cos,
            weight_sqnr_db: w_sqnr,
            act_mse,
            act_cos_sim: act_cos,
        });
    }

    println!("{:<28} | {:<12} | {:<12} | {:<10} | {:<10} | {:<12}", "Method", "Weight CosSim", "Weight MSE", "SQNR (dB)", "Act MSE", "Act CosSim");
    println!("{:-<28}-|-{:-<12}-|-{:-<12}-|-{:-<10}-|-{:-<10}-|-{:-<12}", "", "", "", "", "", "");
    for m in &metrics_table {
        println!("{:<28} | {:<12.6} | {:<12.3e} | {:<10.2} | {:<10.3e} | {:<12.6}", m.method, m.weight_cos_sim, m.weight_mse, m.weight_sqnr_db, m.act_mse, m.act_cos_sim);
    }
    println!("==========================================================================================\n");

    // Quantitative degradation assertions:
    for m in &metrics_table {
        assert!(m.weight_cos_sim >= 0.9950, "{} weight cosine similarity degraded: {}", m.method, m.weight_cos_sim);
        assert!(m.weight_mse < 0.001, "{} weight MSE degraded: {}", m.method, m.weight_mse);
        assert!(m.weight_sqnr_db >= 20.0, "{} SQNR degraded: {}", m.method, m.weight_sqnr_db);
        assert!(m.act_mse < 0.001, "{} activation MSE degraded: {}", m.method, m.act_mse);
    }
}

#[test]
fn test_end_to_end_model_degradation() {
    println!("\n=======================================================");
    println!(" EVALUATING END-TO-END MODEL PREDICTION PRESERVATION");
    println!("=======================================================");

    let tmp = tempdir().unwrap();
    let model_dir = tmp.path().join("llama_e2e");
    create_hf_llama_model(&model_dir);

    // Build INT8 model
    let int8_pox_path = tmp.path().join("model_int8.pox");
    run_quantize(
        model_dir.to_str().unwrap(),
        8,
        0,
        "none",
        "min-max",
        int8_pox_path.to_str().unwrap(),
    )
    .unwrap();

    // Build quantized INT4 model with Group-128 and Outlier Extraction
    let int4_pox_path = tmp.path().join("model_int4.pox");
    run_quantize(
        model_dir.to_str().unwrap(),
        4,
        128,
        "3.5",
        "min-max",
        int4_pox_path.to_str().unwrap(),
    )
    .unwrap();

    // Build AWQ INT4 model
    let awq_pox_path = tmp.path().join("model_awq.pox");
    run_quantize(
        model_dir.to_str().unwrap(),
        4,
        128,
        "3.5",
        "awq",
        awq_pox_path.to_str().unwrap(),
    )
    .unwrap();

    // Load engines
    let int8_file = PoxFile::open(&int8_pox_path).unwrap();
    let int4_file = PoxFile::open(&int4_pox_path).unwrap();
    let awq_file = PoxFile::open(&awq_pox_path).unwrap();

    let int8_engine = PoxEngine::new(int8_file);
    let int4_engine = PoxEngine::new(int4_file);
    let awq_engine = PoxEngine::new(awq_file);

    // Compare logits across test tokens
    let test_tokens = [42u32, 105, 256, 412, 789, 999];
    let mut total_tokens = 0;
    let mut avg_cos_sim_int4 = 0.0;
    let mut avg_cos_sim_awq = 0.0;

    for &tok in &test_tokens {
        let mut kv_int8 = KvCache::new(2, 2, 32, 128);
        let mut kv_int4 = KvCache::new(2, 2, 32, 128);
        let mut kv_awq = KvCache::new(2, 2, 32, 128);

        let logits_int8 = int8_engine.forward_token(tok, 0, &mut kv_int8).unwrap();
        let logits_int4 = int4_engine.forward_token(tok, 0, &mut kv_int4).unwrap();
        let logits_awq = awq_engine.forward_token(tok, 0, &mut kv_awq).unwrap();

        assert_eq!(logits_int8.len(), logits_int4.len());
        assert_eq!(logits_int8.len(), logits_awq.len());

        let (_, _, cos_int4, _) = compute_metrics(&logits_int8, &logits_int4);
        let (_, _, cos_awq, _) = compute_metrics(&logits_int8, &logits_awq);
        avg_cos_sim_int4 += cos_int4;
        avg_cos_sim_awq += cos_awq;

        total_tokens += 1;
    }

    avg_cos_sim_int4 /= total_tokens as f64;
    avg_cos_sim_awq /= total_tokens as f64;

    println!("INT4 Min-Max Logit Cosine Sim:    {:.6}", avg_cos_sim_int4);
    println!("INT4 AWQ Logit Cosine Sim:        {:.6}", avg_cos_sim_awq);
    println!("=======================================================\n");

    // Assertions: Logit directional alignment across multi-layer transformer passes
    assert!(avg_cos_sim_int4 >= 0.70, "INT4 logit correlation degraded: {avg_cos_sim_int4}");
    assert!(avg_cos_sim_awq >= 0.70, "AWQ logit correlation degraded: {avg_cos_sim_awq}");
}
