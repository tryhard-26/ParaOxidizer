use paraoxidizer_quant::{
    kernels::{
        compute_awq_scales, dequantize_awq, dequantize_int4_group, dequantize_int8_symmetric,
        quantize_awq, quantize_gptq, quantize_int4_group, quantize_int8_symmetric,
    },
    outlier::{OutlierPolicy, SparseOutlierTable},
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityBenchmarkResult {
    pub format_name: String,
    pub original_bytes: usize,
    pub quantized_bytes: usize,
    pub compression_ratio: f64,
    pub quant_throughput_mb_s: f64,
    pub mse: f64,
    pub mae: f64,
    pub cosine_similarity: f64,
    pub sqnr_db: f64,
    pub outlier_count: usize,
}

pub fn run_fidelity_benchmarks() -> Vec<FidelityBenchmarkResult> {
    // Generate synthetic LLM weight distribution: Normal distribution with synthetic outlier spikes
    let numel = 1_000_000; // 1M parameters (~4MB in FP32, ~2MB in FP16)
    let mut weights = Vec::with_capacity(numel);

    // Box-Muller transform for normal distribution
    for i in 0..numel {
        let u1 = ((i as f64 * 12345.67).sin().abs() * 0.999 + 0.0001).min(0.9999);
        let u2 = ((i as f64 * 76543.21).cos().abs() * 0.999 + 0.0001).min(0.9999);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let mut val = (z * 0.05) as f32; // std dev = 0.05

        // Introduce pathological activation/weight outliers every 2000 elements (> 4.0 std dev)
        if i % 2000 == 0 {
            val *= 8.0;
        }
        weights.push(val);
    }

    let original_fp16_bytes = numel * 2;
    let mut results = Vec::new();

    // 1. INT8 Symmetric
    {
        let start = Instant::now();
        let (q_weights, scales) = quantize_int8_symmetric(&weights);
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = ((original_fp16_bytes as f64) / 1e6) / elapsed;

        let mut dequant = vec![0.0f32; numel];
        dequantize_int8_symmetric(&q_weights, &scales, &mut dequant).unwrap();

        let (mse, mae, cos_sim, sqnr) = compute_metrics(&weights, &dequant);
        let compressed_bytes = q_weights.len() + scales.len();

        results.push(FidelityBenchmarkResult {
            format_name: "INT8 Symmetric".to_string(),
            original_bytes: original_fp16_bytes,
            quantized_bytes: compressed_bytes,
            compression_ratio: (original_fp16_bytes as f64) / (compressed_bytes as f64),
            quant_throughput_mb_s: throughput,
            mse,
            mae,
            cosine_similarity: cos_sim,
            sqnr_db: sqnr,
            outlier_count: 0,
        });
    }

    // 2. INT4 Group-128 (Standard)
    {
        let start = Instant::now();
        let (packed, scales) = quantize_int4_group(&weights, 128);
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = ((original_fp16_bytes as f64) / 1e6) / elapsed;

        let mut dequant = vec![0.0f32; numel];
        dequantize_int4_group(&packed, &scales, 128, numel, &mut dequant).unwrap();

        let (mse, mae, cos_sim, sqnr) = compute_metrics(&weights, &dequant);
        let compressed_bytes = packed.len() + scales.len();

        results.push(FidelityBenchmarkResult {
            format_name: "INT4 Group-128".to_string(),
            original_bytes: original_fp16_bytes,
            quantized_bytes: compressed_bytes,
            compression_ratio: (original_fp16_bytes as f64) / (compressed_bytes as f64),
            quant_throughput_mb_s: throughput,
            mse,
            mae,
            cosine_similarity: cos_sim,
            sqnr_db: sqnr,
            outlier_count: 0,
        });
    }

    // 3. INT4 Group-64
    {
        let start = Instant::now();
        let (packed, scales) = quantize_int4_group(&weights, 64);
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = ((original_fp16_bytes as f64) / 1e6) / elapsed;

        let mut dequant = vec![0.0f32; numel];
        dequantize_int4_group(&packed, &scales, 64, numel, &mut dequant).unwrap();

        let (mse, mae, cos_sim, sqnr) = compute_metrics(&weights, &dequant);
        let compressed_bytes = packed.len() + scales.len();

        results.push(FidelityBenchmarkResult {
            format_name: "INT4 Group-64".to_string(),
            original_bytes: original_fp16_bytes,
            quantized_bytes: compressed_bytes,
            compression_ratio: (original_fp16_bytes as f64) / (compressed_bytes as f64),
            quant_throughput_mb_s: throughput,
            mse,
            mae,
            cosine_similarity: cos_sim,
            sqnr_db: sqnr,
            outlier_count: 0,
        });
    }

    // 4. INT4 Group-32 (High Precision Low-Bit)
    {
        let start = Instant::now();
        let (packed, scales) = quantize_int4_group(&weights, 32);
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = ((original_fp16_bytes as f64) / 1e6) / elapsed;

        let mut dequant = vec![0.0f32; numel];
        dequantize_int4_group(&packed, &scales, 32, numel, &mut dequant).unwrap();

        let (mse, mae, cos_sim, sqnr) = compute_metrics(&weights, &dequant);
        let compressed_bytes = packed.len() + scales.len();

        results.push(FidelityBenchmarkResult {
            format_name: "INT4 Group-32".to_string(),
            original_bytes: original_fp16_bytes,
            quantized_bytes: compressed_bytes,
            compression_ratio: (original_fp16_bytes as f64) / (compressed_bytes as f64),
            quant_throughput_mb_s: throughput,
            mse,
            mae,
            cosine_similarity: cos_sim,
            sqnr_db: sqnr,
            outlier_count: 0,
        });
    }

    // 5. INT4 Group-128 + Sparse Outlier Table (≥ 3.5 std dev)
    {
        let start = Instant::now();
        let mut weights_copy = weights.clone();
        let outliers = SparseOutlierTable::extract_and_zero_outliers(
            &mut weights_copy,
            OutlierPolicy::Automatic,
        )
        .unwrap_or_default();
        let (packed, scales) = quantize_int4_group(&weights_copy, 128);
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = ((original_fp16_bytes as f64) / 1e6) / elapsed;

        let mut dequant = vec![0.0f32; numel];
        dequantize_int4_group(&packed, &scales, 128, numel, &mut dequant).unwrap();
        // Restore outliers
        for (&idx, &val) in outliers.indices.iter().zip(outliers.values.iter()) {
            dequant[idx as usize] = val.to_f32();
        }

        let (mse, mae, cos_sim, sqnr) = compute_metrics(&weights, &dequant);
        let compressed_bytes = packed.len()
            + scales.len()
            + (outliers.indices.len() * 4)
            + (outliers.values.len() * 2);

        results.push(FidelityBenchmarkResult {
            format_name: "INT4 Group-128 + Outliers".to_string(),
            original_bytes: original_fp16_bytes,
            quantized_bytes: compressed_bytes,
            compression_ratio: (original_fp16_bytes as f64) / (compressed_bytes as f64),
            quant_throughput_mb_s: throughput,
            mse,
            mae,
            cosine_similarity: cos_sim,
            sqnr_db: sqnr,
            outlier_count: outliers.indices.len(),
        });
    }

    // 6. INT4 AWQ (Salience-scaled Group-128)
    {
        let rows = 1000;
        let cols = 1000;
        let mut act_scales = vec![0.0f32; cols];
        for c in 0..cols {
            let channel_weight_l1: f32 = (0..rows).map(|r| weights[r * cols + c].abs()).sum();
            act_scales[c] =
                (channel_weight_l1 / rows as f32) * (1.0 + ((c as f32 * 0.1).sin() * 0.5));
        }

        let start = Instant::now();
        let (packed, scales) = quantize_awq(&weights, rows, cols, &act_scales, 128);
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = ((original_fp16_bytes as f64) / 1e6) / elapsed;

        let protection = compute_awq_scales(&act_scales, cols);
        let mut dequant = vec![0.0f32; numel];
        dequantize_awq(&packed, &scales, &protection, 128, rows, cols, &mut dequant).unwrap();

        let (mse, mae, cos_sim, sqnr) = compute_metrics(&weights, &dequant);
        let compressed_bytes = packed.len() + scales.len();

        results.push(FidelityBenchmarkResult {
            format_name: "INT4 AWQ (Salience-scaled)".to_string(),
            original_bytes: original_fp16_bytes,
            quantized_bytes: compressed_bytes,
            compression_ratio: (original_fp16_bytes as f64) / (compressed_bytes as f64),
            quant_throughput_mb_s: throughput,
            mse,
            mae,
            cosine_similarity: cos_sim,
            sqnr_db: sqnr,
            outlier_count: 0,
        });
    }

    // 7. INT4 GPTQ (Damped H^-1 Error Compensation)
    {
        let rows = 1000;
        let cols = 1000;
        let num_samples = 64;
        let mut hessian = paraoxidizer_calibration::HessianMatrix::new(cols);
        let mut calib_x = vec![0.0f32; cols * num_samples];
        for s in 0..num_samples {
            for c in 0..cols {
                calib_x[c * num_samples + s] = ((s as f32 * 0.3 + c as f32 * 0.05).sin()) * 0.1;
            }
        }
        hessian.accumulate_activations(&calib_x, num_samples);
        hessian.compute_inverse(0.01);

        let start = Instant::now();
        let (packed, scales) = quantize_gptq(&weights, rows, cols, &hessian.inv_data, 128);
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = ((original_fp16_bytes as f64) / 1e6) / elapsed;

        let mut dequant = vec![0.0f32; numel];
        dequantize_int4_group(&packed, &scales, 128, numel, &mut dequant).unwrap();

        let (mse, mae, cos_sim, sqnr) = compute_metrics(&weights, &dequant);
        let compressed_bytes = packed.len() + scales.len();

        results.push(FidelityBenchmarkResult {
            format_name: "INT4 GPTQ (Damped H^-1)".to_string(),
            original_bytes: original_fp16_bytes,
            quantized_bytes: compressed_bytes,
            compression_ratio: (original_fp16_bytes as f64) / (compressed_bytes as f64),
            quant_throughput_mb_s: throughput,
            mse,
            mae,
            cosine_similarity: cos_sim,
            sqnr_db: sqnr,
            outlier_count: 0,
        });
    }

    results
}

fn compute_metrics(orig: &[f32], recon: &[f32]) -> (f64, f64, f64, f64) {
    let mut sum_sq_err = 0.0;
    let mut sum_abs_err = 0.0;
    let mut dot_prod = 0.0;
    let mut norm_orig_sq = 0.0;
    let mut norm_recon_sq = 0.0;

    for (&o, &r) in orig.iter().zip(recon.iter()) {
        let diff = (o - r) as f64;
        let o_f64 = o as f64;
        let r_f64 = r as f64;

        sum_sq_err += diff * diff;
        sum_abs_err += diff.abs();
        dot_prod += o_f64 * r_f64;
        norm_orig_sq += o_f64 * o_f64;
        norm_recon_sq += r_f64 * r_f64;
    }

    let n = orig.len() as f64;
    let mse = sum_sq_err / n;
    let mae = sum_abs_err / n;
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

    (mse, mae, cos_sim, sqnr)
}
