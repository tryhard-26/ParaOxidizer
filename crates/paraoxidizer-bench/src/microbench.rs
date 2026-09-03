#![allow(clippy::needless_range_loop)]

use paraoxidizer_quant::kernels::{dot_product_simd, gemv_int4, gemv_int8, quantize_int4_group, quantize_int8_symmetric};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvBenchmarkResult {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub bitwidth: u8,
    pub group_size: usize,
    pub latency_us: f64,
    pub bandwidth_gb_s: f64,
    pub gflops: f64,
    pub speedup_vs_fp32: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DotProductBenchmarkResult {
    pub vector_size: usize,
    pub simd_latency_ns: f64,
    pub scalar_latency_ns: f64,
    pub speedup: f64,
}

/// Naive scalar dot product for comparison against SIMD
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

pub fn run_dot_product_benchmarks() -> Vec<DotProductBenchmarkResult> {
    let sizes = [512, 1024, 2048, 4096, 8192];
    let mut results = Vec::new();

    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|i| ((i % 100) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..size).map(|i| (((i + 50) % 100) as f32) * 0.01).collect();

        // Warmup
        for _ in 0..50 {
            std::hint::black_box(dot_product_simd(&a, &b));
            std::hint::black_box(dot_product_scalar(&a, &b));
        }

        // Measure SIMD
        let iters = 2000;
        let start_simd = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(dot_product_simd(&a, &b));
        }
        let simd_duration = start_simd.elapsed();
        let simd_ns = (simd_duration.as_nanos() as f64) / (iters as f64);

        // Measure Scalar
        let start_scalar = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(dot_product_scalar(&a, &b));
        }
        let scalar_duration = start_scalar.elapsed();
        let scalar_ns = (scalar_duration.as_nanos() as f64) / (iters as f64);

        let speedup = if simd_ns > 0.0 { scalar_ns / simd_ns } else { 1.0 };

        results.push(DotProductBenchmarkResult {
            vector_size: size,
            simd_latency_ns: simd_ns,
            scalar_latency_ns: scalar_ns,
            speedup,
        });
    }

    results
}

pub fn run_gemv_benchmarks() -> Vec<GemvBenchmarkResult> {
    let configs = [
        ("Projection (1024x1024)", 1024, 1024),
        ("Attention (2048x2048)", 2048, 2048),
        ("FFN Layer (4096x4096)", 4096, 4096),
    ];

    let mut results = Vec::new();

    for (name, rows, cols) in configs {
        let total_elements = rows * cols;
        let weights: Vec<f32> = (0..total_elements)
            .map(|i| (i as f32 * 0.001).sin() * 0.2)
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).cos()).collect();
        let mut y = vec![0.0f32; rows];

        let ops = 2.0 * (rows as f64) * (cols as f64);

        // 1. Baseline FP32 Scalar GEMV
        let iters_fp32 = 20;
        let start_fp32 = Instant::now();
        for _ in 0..iters_fp32 {
            for r in 0..rows {
                let row_start = r * cols;
                let mut sum = 0.0f32;
                for c in 0..cols {
                    sum += weights[row_start + c] * x[c];
                }
                y[r] = sum;
            }
        }
        let fp32_latency_us = (start_fp32.elapsed().as_micros() as f64) / (iters_fp32 as f64);

        // 2. INT8 GEMV
        let (int8_q, int8_scale) = quantize_int8_symmetric(&weights);
        let iters_int8 = 30;
        for _ in 0..5 {
            let _ = gemv_int8(&int8_q, &int8_scale, rows, cols, &x, &mut y);
        }
        let start_int8 = Instant::now();
        for _ in 0..iters_int8 {
            let _ = gemv_int8(&int8_q, &int8_scale, rows, cols, &x, &mut y);
        }
        let int8_latency_us = (start_int8.elapsed().as_micros() as f64) / (iters_int8 as f64);
        let int8_bytes = (int8_q.len() + int8_scale.len() + (cols * 4) + (rows * 4)) as f64;
        let int8_bw = (int8_bytes / (int8_latency_us * 1e-6)) / 1e9;
        let int8_gflops = (ops / (int8_latency_us * 1e-6)) / 1e9;

        results.push(GemvBenchmarkResult {
            name: format!("{} INT8 Sym", name),
            rows,
            cols,
            bitwidth: 8,
            group_size: 0,
            latency_us: int8_latency_us,
            bandwidth_gb_s: int8_bw,
            gflops: int8_gflops,
            speedup_vs_fp32: fp32_latency_us / int8_latency_us,
        });

        // 3. INT4 Group-128
        let (int4_q, int4_scales) = quantize_int4_group(&weights, 128);
        let iters_int4 = 30;
        for _ in 0..5 {
            let _ = gemv_int4(&int4_q, &int4_scales, 128, rows, cols, None, &x, &mut y);
        }
        let start_int4 = Instant::now();
        for _ in 0..iters_int4 {
            let _ = gemv_int4(&int4_q, &int4_scales, 128, rows, cols, None, &x, &mut y);
        }
        let int4_latency_us = (start_int4.elapsed().as_micros() as f64) / (iters_int4 as f64);
        let int4_bytes = (int4_q.len() + int4_scales.len() + (cols * 4) + (rows * 4)) as f64;
        let int4_bw = (int4_bytes / (int4_latency_us * 1e-6)) / 1e9;
        let int4_gflops = (ops / (int4_latency_us * 1e-6)) / 1e9;

        results.push(GemvBenchmarkResult {
            name: format!("{} INT4 G128", name),
            rows,
            cols,
            bitwidth: 4,
            group_size: 128,
            latency_us: int4_latency_us,
            bandwidth_gb_s: int4_bw,
            gflops: int4_gflops,
            speedup_vs_fp32: fp32_latency_us / int4_latency_us,
        });

        // 4. INT4 Group-32
        let (int4_g32_q, int4_g32_scales) = quantize_int4_group(&weights, 32);
        let start_int4_g32 = Instant::now();
        for _ in 0..iters_int4 {
            let _ = gemv_int4(&int4_g32_q, &int4_g32_scales, 32, rows, cols, None, &x, &mut y);
        }
        let int4_g32_latency_us = (start_int4_g32.elapsed().as_micros() as f64) / (iters_int4 as f64);
        let int4_g32_bytes = (int4_g32_q.len() + int4_g32_scales.len() + (cols * 4) + (rows * 4)) as f64;
        let int4_g32_bw = (int4_g32_bytes / (int4_g32_latency_us * 1e-6)) / 1e9;
        let int4_g32_gflops = (ops / (int4_g32_latency_us * 1e-6)) / 1e9;

        results.push(GemvBenchmarkResult {
            name: format!("{} INT4 G32", name),
            rows,
            cols,
            bitwidth: 4,
            group_size: 32,
            latency_us: int4_g32_latency_us,
            bandwidth_gb_s: int4_g32_bw,
            gflops: int4_g32_gflops,
            speedup_vs_fp32: fp32_latency_us / int4_g32_latency_us,
        });
    }

    results
}
