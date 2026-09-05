use comfy_table::{presets::UTF8_FULL, Cell, Color, Row, Table};
use half::{bf16, f16};
use memmap2::Mmap;
use paraoxidizer_core::tensor::{DType, QuantGroupSize, Shape};
use paraoxidizer_format::{
    hf::HfConfigJson,
    pox::{PoxFile, PoxMetadata, PoxQuantPlanRecord, PoxWriter},
};
use paraoxidizer_quant::kernels::{
    dequantize_int4_group, dequantize_int8_symmetric, quantize_int4_group, quantize_int8_symmetric,
};
use safetensors::tensor::{Dtype as StDtype, SafeTensors};
use std::{collections::HashMap, fs::File, io::Read, path::Path, time::Instant};

#[derive(Default)]
struct CategoryStats {
    total_params: usize,
    sum_sq_err: f64,
    dot_prod: f64,
    norm_orig_sq: f64,
    norm_recon_sq: f64,
    tensor_count: usize,
}

impl CategoryStats {
    fn add(&mut self, orig: &[f32], recon: &[f32]) {
        self.total_params += orig.len();
        self.tensor_count += 1;
        for (&o, &r) in orig.iter().zip(recon.iter()) {
            let diff = (o - r) as f64;
            let o_f = o as f64;
            let r_f = r as f64;
            self.sum_sq_err += diff * diff;
            self.dot_prod += o_f * r_f;
            self.norm_orig_sq += o_f * o_f;
            self.norm_recon_sq += r_f * r_f;
        }
    }

    fn cos_sim(&self) -> f64 {
        if self.norm_orig_sq > 0.0 && self.norm_recon_sq > 0.0 {
            self.dot_prod / (self.norm_orig_sq.sqrt() * self.norm_recon_sq.sqrt())
        } else {
            1.0
        }
    }

    fn mse(&self) -> f64 {
        if self.total_params > 0 {
            self.sum_sq_err / (self.total_params as f64)
        } else {
            0.0
        }
    }

    fn sqnr(&self) -> f64 {
        if self.sum_sq_err > 1e-12 {
            10.0 * (self.norm_orig_sq / self.sum_sq_err).log10()
        } else {
            99.9
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = Path::new("scratch/tinyllama");
    let config_path = model_dir.join("config.json");
    let safetensors_path = model_dir.join("model.safetensors");
    let output_pox_path = model_dir.join("tinyllama-1.1b-int4.pox");

    if !safetensors_path.exists() {
        eprintln!(
            "Error: safetensors file not found at {}",
            safetensors_path.display()
        );
        std::process::exit(1);
    }

    println!("==================================================================================================");
    println!(" FULL-MODEL END-TO-END QUANTIZATION BENCHMARK: TinyLlama-1.1B-Chat-v1.0 (Apple Silicon M4)");
    println!("==================================================================================================\n");

    // 1. Ingest config.json
    let mut config_file = File::open(&config_path)?;
    let mut config_str = String::new();
    config_file.read_to_string(&mut config_str)?;
    let hf_config: HfConfigJson = serde_json::from_str(&config_str)?;
    let model_config = hf_config.to_model_config();

    println!("Architecture:           {}", model_config.architecture);
    println!("Hidden Layers:          {}", model_config.num_hidden_layers);
    println!("Hidden Size:            {}", model_config.hidden_size);
    println!(
        "Attention Heads:        {} (KV heads: {})",
        model_config.num_attention_heads, model_config.num_key_value_heads
    );
    println!("Vocab Size:             {}", model_config.vocab_size);

    // 2. Memory-map SafeTensors file
    let st_file = File::open(&safetensors_path)?;
    let st_file_size = st_file.metadata()?.len();
    println!(
        "Source SafeTensors Size: {:.2} MB ({:.3} GB)",
        st_file_size as f64 / (1024.0 * 1024.0),
        st_file_size as f64 / 1e9
    );

    let mmap = unsafe { Mmap::map(&st_file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let total_tensors = st.tensors().len();
    println!("Total Tensors to Quantize: {}\n", total_tensors);

    // 3. Setup PoxWriter
    let metadata = PoxMetadata {
        model_config: model_config.clone(),
        total_parameters: 1_100_048_384,
        quantized_by: format!("ParaOxidizer v{}", env!("CARGO_PKG_VERSION")),
        timestamp_utc: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        original_format: "SafeTensors (BF16)".into(),
        base_model_name: "TinyLlama-1.1B-Chat-v1.0".into(),
    };

    let quant_plan = PoxQuantPlanRecord {
        default_precision: "INT4 (Group-128)".into(),
        group_size: 128,
        outlier_strategy: "None".into(),
        layer_assignments: HashMap::new(),
    };

    let mut writer = PoxWriter::new(metadata, quant_plan, "full-model-eval".to_string());

    // Category accumulators
    let mut stats_attn = CategoryStats::default();
    let mut stats_mlp = CategoryStats::default();
    let mut stats_embed = CategoryStats::default();
    let mut stats_norm = CategoryStats::default();
    let mut stats_head = CategoryStats::default();
    let mut stats_global = CategoryStats::default();

    let quant_start = Instant::now();

    for (name, view) in st.tensors() {
        let shape_vec = view.shape().to_vec();
        let shape = Shape::new(shape_vec.clone());
        let raw_data = view.data();

        let floats: Vec<f32> = match view.dtype() {
            StDtype::BF16 => raw_data
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            StDtype::F16 => raw_data
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            StDtype::F32 => raw_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => panic!("Unexpected dtype in safetensors: {:?}", other),
        };

        let numel = floats.len();

        let is_norm = name.contains("layernorm") || name.contains("norm");
        let is_embed = name.contains("embed_tokens");
        let is_head = name.contains("lm_head");
        let is_attn = name.contains("self_attn");
        let is_mlp = name.contains("mlp");

        if is_norm {
            // Norm weights are kept in INT8 or high-precision
            let (q_int8, scales_int8) = quantize_int8_symmetric(&floats);
            let mut dequant = vec![0.0f32; numel];
            dequantize_int8_symmetric(&q_int8, &scales_int8, &mut dequant)?;

            stats_norm.add(&floats, &dequant);
            stats_global.add(&floats, &dequant);

            writer.add_tensor(
                name.to_string(),
                shape,
                DType::I8,
                QuantGroupSize::None,
                &q_int8,
                Some(&scales_int8),
                None,
            );
        } else {
            // Linear projection / embedding / lm_head weights quantized to INT4 Group-128
            let (packed_int4, scales_int4) = quantize_int4_group(&floats, 128);
            let mut dequant = vec![0.0f32; numel];
            dequantize_int4_group(&packed_int4, &scales_int4, 128, numel, &mut dequant)?;

            if is_embed {
                stats_embed.add(&floats, &dequant);
            } else if is_head {
                stats_head.add(&floats, &dequant);
            } else if is_attn {
                stats_attn.add(&floats, &dequant);
            } else if is_mlp {
                stats_mlp.add(&floats, &dequant);
            }
            stats_global.add(&floats, &dequant);

            writer.add_tensor(
                name.to_string(),
                shape,
                DType::I4,
                QuantGroupSize::G128,
                &packed_int4,
                Some(&scales_int4),
                None,
            );
        }
    }

    let quant_duration = quant_start.elapsed();
    println!(
        "All {} tensors quantized in {:.2}s",
        total_tensors,
        quant_duration.as_secs_f64()
    );

    // 4. Write full .pox container to disk
    println!(
        "Writing compiled .pox container to {}...",
        output_pox_path.display()
    );
    let write_start = Instant::now();
    writer.write_to_file(&output_pox_path)?;
    let write_duration = write_start.elapsed();

    let pox_file_size = std::fs::metadata(&output_pox_path)?.len();
    let compression_ratio = st_file_size as f64 / pox_file_size as f64;

    println!(
        "Wrote {:.2} MB in {:.2}s (Compression: {:.2}x)\n",
        pox_file_size as f64 / (1024.0 * 1024.0),
        write_duration.as_secs_f64(),
        compression_ratio
    );

    // 5. Verify .pox container via zero-copy mmap
    let mmap_start = Instant::now();
    let pox = PoxFile::open(&output_pox_path)?;
    let mmap_duration = mmap_start.elapsed();

    let verify_start = Instant::now();
    pox.verify_integrity()?;
    let verify_duration = verify_start.elapsed();

    println!("Container Verification:");
    println!(
        "  Cold-Start mmap Time: {:.2} µs",
        mmap_duration.as_micros()
    );
    println!(
        "  Full SHA-256 Checksum: Verified in {:.2} ms (All {} tensors bit-intact)\n",
        verify_duration.as_secs_f64() * 1000.0,
        pox.tensors.len()
    );

    // 6. Print Full Model Benchmark Table
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        Cell::new("Model Component / Layer Type").fg(Color::Cyan),
        Cell::new("Tensor Count").fg(Color::Yellow),
        Cell::new("Parameters").fg(Color::Yellow),
        Cell::new("Quantization").fg(Color::Green),
        Cell::new("Cosine Similarity").fg(Color::Green),
        Cell::new("MSE").fg(Color::Green),
        Cell::new("SQNR (dB)").fg(Color::Green),
    ]);

    table.add_row(Row::from(vec![
        Cell::new("Attention Projections (q, k, v, o)"),
        Cell::new(format!("{}", stats_attn.tensor_count)),
        Cell::new(format!("{:.2} M", stats_attn.total_params as f64 / 1e6)),
        Cell::new("INT4 (Group-128)"),
        Cell::new(format!("{:.6}", stats_attn.cos_sim())).fg(Color::Green),
        Cell::new(format!("{:.4e}", stats_attn.mse())),
        Cell::new(format!("{:.2} dB", stats_attn.sqnr())).fg(Color::Cyan),
    ]));

    table.add_row(Row::from(vec![
        Cell::new("Feed-Forward MLP (gate, up, down)"),
        Cell::new(format!("{}", stats_mlp.tensor_count)),
        Cell::new(format!("{:.2} M", stats_mlp.total_params as f64 / 1e6)),
        Cell::new("INT4 (Group-128)"),
        Cell::new(format!("{:.6}", stats_mlp.cos_sim())).fg(Color::Green),
        Cell::new(format!("{:.4e}", stats_mlp.mse())),
        Cell::new(format!("{:.2} dB", stats_mlp.sqnr())).fg(Color::Cyan),
    ]));

    table.add_row(Row::from(vec![
        Cell::new("Token Embeddings (embed_tokens)"),
        Cell::new(format!("{}", stats_embed.tensor_count)),
        Cell::new(format!("{:.2} M", stats_embed.total_params as f64 / 1e6)),
        Cell::new("INT4 (Group-128)"),
        Cell::new(format!("{:.6}", stats_embed.cos_sim())).fg(Color::Green),
        Cell::new(format!("{:.4e}", stats_embed.mse())),
        Cell::new(format!("{:.2} dB", stats_embed.sqnr())).fg(Color::Cyan),
    ]));

    table.add_row(Row::from(vec![
        Cell::new("Output LM Head (lm_head)"),
        Cell::new(format!("{}", stats_head.tensor_count)),
        Cell::new(format!("{:.2} M", stats_head.total_params as f64 / 1e6)),
        Cell::new("INT4 (Group-128)"),
        Cell::new(format!("{:.6}", stats_head.cos_sim())).fg(Color::Green),
        Cell::new(format!("{:.4e}", stats_head.mse())),
        Cell::new(format!("{:.2} dB", stats_head.sqnr())).fg(Color::Cyan),
    ]));

    table.add_row(Row::from(vec![
        Cell::new("Normalization Weights (rms_norm)"),
        Cell::new(format!("{}", stats_norm.tensor_count)),
        Cell::new(format!("{:.2} k", stats_norm.total_params as f64 / 1e3)),
        Cell::new("INT8 Symmetric"),
        Cell::new(format!("{:.6}", stats_norm.cos_sim())).fg(Color::Green),
        Cell::new(format!("{:.4e}", stats_norm.mse())),
        Cell::new(format!("{:.2} dB", stats_norm.sqnr())).fg(Color::Cyan),
    ]));

    table.add_row(Row::from(vec![
        Cell::new("TOTAL / WHOLE MODEL AGGREGATE").fg(Color::Yellow),
        Cell::new(format!("{}", stats_global.tensor_count)).fg(Color::Yellow),
        Cell::new(format!("{:.3} B", stats_global.total_params as f64 / 1e9)).fg(Color::Yellow),
        Cell::new("Mixed INT4/INT8").fg(Color::Yellow),
        Cell::new(format!("{:.6}", stats_global.cos_sim())).fg(Color::Green),
        Cell::new(format!("{:.4e}", stats_global.mse())),
        Cell::new(format!("{:.2} dB", stats_global.sqnr())).fg(Color::Cyan),
    ]));

    println!("{}", table);

    println!("\nPhysical Disk Footprint:");
    println!(
        "  Raw SafeTensors (BF16):  {:.2} MB ({:.3} GB)",
        st_file_size as f64 / (1024.0 * 1024.0),
        st_file_size as f64 / 1e9
    );
    println!(
        "  Compiled .pox (INT4/8):  {:.2} MB ({:.3} GB)",
        pox_file_size as f64 / (1024.0 * 1024.0),
        pox_file_size as f64 / 1e9
    );
    println!("  True Compression Ratio:  {:.2}x", compression_ratio);
    println!(
        "  Total Parameters:        {} (100% verified)",
        stats_global.total_params
    );

    Ok(())
}
