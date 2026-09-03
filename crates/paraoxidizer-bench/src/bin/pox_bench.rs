use comfy_table::{presets::UTF8_FULL, Cell, Color, Row, Table};
use paraoxidizer_bench::{
    fidelity::run_fidelity_benchmarks,
    microbench::{run_dot_product_benchmarks, run_gemv_benchmarks},
    system::run_system_benchmarks,
};

fn main() {
    println!("\n=======================================================");
    println!("  ParaOxidizer Advanced Engineering Benchmark Suite");
    println!("  Host: macOS Darwin (Apple Silicon) | Target: arm64");
    println!("=======================================================\n");

    // 1. SIMD Dot-Product Vector Acceleration
    println!("▶ [1/4] Running SIMD Vector Dot-Product Microbenchmarks...");
    let dot_results = run_dot_product_benchmarks();
    let mut dot_table = Table::new();
    dot_table.load_preset(UTF8_FULL);
    dot_table.set_header(vec![
        Cell::new("Vector Dimension (N)").fg(Color::Cyan),
        Cell::new("ARM NEON SIMD (ns)").fg(Color::Green),
        Cell::new("Scalar Fallback (ns)").fg(Color::Yellow),
        Cell::new("SIMD Speedup").fg(Color::Magenta),
    ]);
    for r in &dot_results {
        dot_table.add_row(Row::from(vec![
            Cell::new(r.vector_size.to_string()),
            Cell::new(format!("{:.1} ns", r.simd_latency_ns)),
            Cell::new(format!("{:.1} ns", r.scalar_latency_ns)),
            Cell::new(format!("{:.2}x", r.speedup)),
        ]));
    }
    println!("{}\n", dot_table);

    // 2. Quantized GEMV Matrix-Vector Multiplication
    println!("▶ [2/4] Running Quantized Matrix-Vector (GEMV) Kernels...");
    let gemv_results = run_gemv_benchmarks();
    let mut gemv_table = Table::new();
    gemv_table.load_preset(UTF8_FULL);
    gemv_table.set_header(vec![
        Cell::new("Kernel Configuration").fg(Color::Cyan),
        Cell::new("Shape [M × K]").fg(Color::Blue),
        Cell::new("Latency (µs)").fg(Color::Yellow),
        Cell::new("Bandwidth (GB/s)").fg(Color::Green),
        Cell::new("Compute (GFLOPS)").fg(Color::Green),
        Cell::new("Speedup vs FP32").fg(Color::Magenta),
    ]);
    for r in &gemv_results {
        gemv_table.add_row(Row::from(vec![
            Cell::new(&r.name),
            Cell::new(format!("{} × {}", r.rows, r.cols)),
            Cell::new(format!("{:.1} µs", r.latency_us)),
            Cell::new(format!("{:.2} GB/s", r.bandwidth_gb_s)),
            Cell::new(format!("{:.2} GFLOPS", r.gflops)),
            Cell::new(format!("{:.2}x", r.speedup_vs_fp32)),
        ]));
    }
    println!("{}\n", gemv_table);

    // 3. Quantization Compression & Reconstruction Error (Fidelity)
    println!("▶ [3/4] Running Quantization Fidelity & Reconstruction Error Analysis...");
    let fidelity_results = run_fidelity_benchmarks();
    let mut fid_table = Table::new();
    fid_table.load_preset(UTF8_FULL);
    fid_table.set_header(vec![
        Cell::new("Quantization Scheme").fg(Color::Cyan),
        Cell::new("Compression").fg(Color::Green),
        Cell::new("Throughput (MB/s)").fg(Color::Yellow),
        Cell::new("MSE").fg(Color::Blue),
        Cell::new("Cosine Sim").fg(Color::Green),
        Cell::new("SQNR (dB)").fg(Color::Magenta),
        Cell::new("Outliers Kept").fg(Color::White),
    ]);
    for r in &fidelity_results {
        fid_table.add_row(Row::from(vec![
            Cell::new(&r.format_name),
            Cell::new(format!("{:.2}x", r.compression_ratio)),
            Cell::new(format!("{:.1} MB/s", r.quant_throughput_mb_s)),
            Cell::new(format!("{:.2e}", r.mse)),
            Cell::new(format!("{:.6}", r.cosine_similarity)),
            Cell::new(format!("{:.2} dB", r.sqnr_db)),
            Cell::new(r.outlier_count.to_string()),
        ]));
    }
    println!("{}\n", fid_table);

    // 4. System & Container Deserialization
    println!("▶ [4/4] Running System & Container Deserialization Benchmarks...");
    let sys_results = run_system_benchmarks();
    let mut sys_table = Table::new();
    sys_table.load_preset(UTF8_FULL);
    sys_table.set_header(vec![
        Cell::new("Operation").fg(Color::Cyan),
        Cell::new("Payload").fg(Color::Blue),
        Cell::new("Latency").fg(Color::Yellow),
        Cell::new("Throughput (GB/s)").fg(Color::Green),
        Cell::new("Notes").fg(Color::White),
    ]);
    for r in &sys_results {
        let payload_str = if r.payload_size_mb > 0.0 {
            format!("{:.1} MB", r.payload_size_mb)
        } else {
            "-".to_string()
        };
        let latency_str = if r.latency_ms < 1.0 {
            format!("{:.2} µs", r.latency_ms * 1000.0)
        } else {
            format!("{:.2} ms", r.latency_ms)
        };
        let bw_str = if r.throughput_gb_s > 0.0 {
            format!("{:.2} GB/s", r.throughput_gb_s)
        } else {
            "-".to_string()
        };
        sys_table.add_row(Row::from(vec![
            Cell::new(&r.benchmark_name),
            Cell::new(payload_str),
            Cell::new(latency_str),
            Cell::new(bw_str),
            Cell::new(&r.notes),
        ]));
    }
    println!("{}\n", sys_table);

    println!("=======================================================");
    println!("  Benchmark Suite Completed Successfully");
    println!("=======================================================\n");
}
