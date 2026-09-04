mod common;

use common::create_hf_llama_model;
use paraoxidizer::cli::commands;
use paraoxidizer::format::PoxFile;
use paraoxidizer::quant::{
    dequantize_int4_group, dequantize_int8_symmetric, dot_product_simd, quantize_int4_group,
    quantize_int8_symmetric, OutlierPolicy, SparseOutlierTable,
};
use paraoxidizer::security::{verify_signature_hex, KeyPair};
use tempfile::tempdir;

#[test]
fn test_int4_group_quantization_roundtrip() {
    let mut weights = Vec::new();
    for i in 0..512 {
        weights.push(((i as f32) * 0.05).sin() * 2.5);
    }

    for group_size in [32, 64, 128, 256] {
        let (packed, scales) = quantize_int4_group(&weights, group_size);
        assert_eq!(packed.len(), weights.len().div_ceil(2));

        let mut dequant = vec![0.0f32; weights.len()];
        dequantize_int4_group(&packed, &scales, group_size, weights.len(), &mut dequant).unwrap();

        // Calculate max error
        let mut max_err = 0.0f32;
        for (orig, recon) in weights.iter().zip(dequant.iter()) {
            let err = (orig - recon).abs();
            if err > max_err {
                max_err = err;
            }
        }

        // For INT4 with dynamic range 5.0, max step is ~5.0/15 = 0.33, so max error <= 0.25
        assert!(
            max_err < 0.25,
            "Group size {} had max error {}",
            group_size,
            max_err
        );
    }
}

#[test]
fn test_int8_quantization_roundtrip() {
    let mut weights = Vec::new();
    for i in 0..256 {
        weights.push(((i as f32) * 0.1).cos() * 1.8);
    }

    let (q_data, scales) = quantize_int8_symmetric(&weights);
    let mut dequant = vec![0.0f32; weights.len()];
    dequantize_int8_symmetric(&q_data, &scales, &mut dequant).unwrap();

    let mut max_err = 0.0f32;
    for (orig, recon) in weights.iter().zip(dequant.iter()) {
        let err = (orig - recon).abs();
        if err > max_err {
            max_err = err;
        }
    }

    // For INT8, max step is ~1.8/127 = 0.015
    assert!(max_err < 0.03, "INT8 max error too high: {}", max_err);
}

#[test]
fn test_outlier_extraction_and_restoration() {
    let mut weights: Vec<f32> = (0..200).map(|i| ((i as f32) * 0.1).sin() * 0.2).collect();
    weights[50] = 15.0;
    weights[120] = -18.0;
    let orig_weights = weights.clone();

    let table_opt =
        SparseOutlierTable::extract_and_zero_outliers(&mut weights, OutlierPolicy::Automatic);
    assert!(table_opt.is_some());
    let table = table_opt.unwrap();
    assert_eq!(table.len(), 2); // 15.0 and -18.0 extracted

    // The weights slice should now have 0.0 at indices 50 and 120
    assert_eq!(weights[50], 0.0);
    assert_eq!(weights[120], 0.0);

    // Test serialization roundtrip
    let bytes = table.to_bytes();
    let deserialized = SparseOutlierTable::from_bytes(&bytes).unwrap();
    assert_eq!(deserialized.len(), 2);

    // Restore on top of weights
    deserialized.apply_to(&mut weights);
    assert!((weights[50] - orig_weights[50]).abs() < 0.01);
    assert!((weights[120] - orig_weights[120]).abs() < 0.01);
}

#[test]
fn test_simd_dot_product() {
    let a: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.02).cos()).collect();

    let simd_res = dot_product_simd(&a, &b);

    // Scalar reference
    let mut scalar_res = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        scalar_res += x * y;
    }

    assert!((simd_res - scalar_res).abs() < 1e-4);
}

#[test]
fn test_cryptographic_supply_chain() {
    let keypair = KeyPair::generate();
    let pub_hex = keypair.public_key_hex();
    let message = b"Model artifact SHA-256 integrity hash payload";

    let signature_hex = keypair.sign_message(message);

    // Verify valid signature
    let valid = verify_signature_hex(message, &pub_hex, &signature_hex).unwrap();
    assert!(valid);

    // Verify invalid signature rejection
    let invalid = verify_signature_hex(b"Tampered payload", &pub_hex, &signature_hex).unwrap();
    assert!(!invalid);
}

#[test]
fn test_tamper_detection_in_pox_file() {
    let tmp = tempdir().unwrap();
    let model_dir = tmp.path().join("llama_tamper");
    create_hf_llama_model(&model_dir);

    let pox_path = tmp.path().join("clean.pox");
    commands::run_quantize(
        model_dir.to_str().unwrap(),
        4,
        128,
        "automatic",
        "min-max",
        pox_path.to_str().unwrap(),
    )
    .unwrap();

    // Verify clean artifact
    commands::run_validate(pox_path.to_str().unwrap()).unwrap();
    commands::run_verify(pox_path.to_str().unwrap(), None).unwrap();

    // Inject 1-byte tamper into tensor data payload
    let mut raw_bytes = std::fs::read(&pox_path).unwrap();
    let tamper_idx = raw_bytes.len() - 100; // byte inside payload
    raw_bytes[tamper_idx] ^= 0xFF; // Flip all bits
    let tampered_path = tmp.path().join("tampered.pox");
    std::fs::write(&tampered_path, raw_bytes).unwrap();

    // Verification must detect the bit-flip and fail
    let verify_res = commands::run_verify(tampered_path.to_str().unwrap(), None);
    assert!(
        verify_res.is_err(),
        "Cryptographic verification must fail on tampered bytes"
    );
}

#[test]
fn test_toml_build_pipeline() {
    let tmp = tempdir().unwrap();
    let model_dir = tmp.path().join("hf_model");
    create_hf_llama_model(&model_dir);

    let out_pox = tmp.path().join("built_model.pox");
    let toml_path = tmp.path().join("paraoxidizer.toml");

    let toml_content = format!(
        r#"
[model]
source = "{}"
output = "{}"

[quantization]
algorithm = "min-max"
outlier_policy = "automatic"

[optimization]
memory_limit = "8GB"
quality_floor = 97.0
target_hardware = "auto"
"#,
        model_dir.display(),
        out_pox.display()
    );

    std::fs::write(&toml_path, toml_content).unwrap();

    // Run build
    commands::run_build(toml_path.to_str().unwrap()).unwrap();
    assert!(out_pox.exists());

    // Validate
    let pox = PoxFile::open(&out_pox).unwrap();
    assert_eq!(
        pox.metadata.model_config.architecture,
        paraoxidizer::core::ModelArchitecture::Llama
    );
}

#[test]
fn test_workload_and_reproduce() {
    let tmp = tempdir().unwrap();
    let workload_out = tmp.path().join("coding_trace.jsonl");

    commands::run_workload("coding-agent", Some(workload_out.to_str().unwrap())).unwrap();
    assert!(workload_out.exists());

    let model_dir = tmp.path().join("model_hf");
    create_hf_llama_model(&model_dir);
    let pox_path = tmp.path().join("reproducible.pox");

    commands::run_quantize(
        model_dir.to_str().unwrap(),
        4,
        64,
        "automatic",
        "min-max",
        pox_path.to_str().unwrap(),
    )
    .unwrap();

    commands::run_inspect_run(pox_path.to_str().unwrap()).unwrap();
    commands::run_reproduce(pox_path.to_str().unwrap()).unwrap();
}

#[test]
fn test_awq_and_gptq_fidelity() {
    use paraoxidizer::calibration::HessianMatrix;
    use paraoxidizer::quant::{
        dequantize_int4_group, quantize_awq, quantize_gptq, quantize_int4_group,
    };

    let rows = 64;
    let cols = 64;
    let total = rows * cols;
    let weights: Vec<f32> = (0..total).map(|i| (i as f32 * 0.05).sin() * 0.2).collect();

    // 1. Min-Max Baseline
    let (base_packed, base_scales) = quantize_int4_group(&weights, 32);
    let mut base_dequant = vec![0.0f32; total];
    dequantize_int4_group(&base_packed, &base_scales, 32, total, &mut base_dequant).unwrap();
    let base_mse: f32 = weights
        .iter()
        .zip(base_dequant.iter())
        .map(|(w, q)| (w - q).powi(2))
        .sum::<f32>()
        / total as f32;

    // 2. AWQ Quantization with activation scales
    let act_scales: Vec<f32> = (0..cols)
        .map(|c| (c as f32 * 0.1).cos().abs() + 0.1)
        .collect();
    let (awq_packed, awq_scales) = quantize_awq(&weights, rows, cols, &act_scales, 32);
    let mut awq_dequant = vec![0.0f32; total];
    dequantize_int4_group(&awq_packed, &awq_scales, 32, total, &mut awq_dequant).unwrap();
    let awq_mse: f32 = weights
        .iter()
        .zip(awq_dequant.iter())
        .map(|(w, q)| (w - q).powi(2))
        .sum::<f32>()
        / total as f32;

    // 3. GPTQ Quantization with Hessian
    let mut hessian = HessianMatrix::new(cols);
    let dummy_acts: Vec<f32> = (0..cols * 32).map(|i| (i as f32 * 0.02).sin()).collect();
    hessian.accumulate_activations(&dummy_acts, 32);
    hessian.compute_inverse(0.01);
    let (gptq_packed, gptq_scales) = quantize_gptq(&weights, rows, cols, &hessian.inv_data, 32);
    let mut gptq_dequant = vec![0.0f32; total];
    dequantize_int4_group(&gptq_packed, &gptq_scales, 32, total, &mut gptq_dequant).unwrap();
    let gptq_mse: f32 = weights
        .iter()
        .zip(gptq_dequant.iter())
        .map(|(w, q)| (w - q).powi(2))
        .sum::<f32>()
        / total as f32;

    assert!(base_mse < 0.01, "Base MSE must be low: {}", base_mse);
    assert!(awq_mse < 0.01, "AWQ MSE must be low: {}", awq_mse);
    assert!(gptq_mse < 0.01, "GPTQ MSE must be low: {}", gptq_mse);
}
