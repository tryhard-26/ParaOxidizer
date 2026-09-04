mod common;

use common::{create_hf_llama_model, create_hf_mistral_model, create_hf_qwen_model, create_hf_sharded_model};
use paraoxidizer::cli::commands;
use paraoxidizer::format::HfModel;
use paraoxidizer::format::PoxFile;
use tempfile::tempdir;

#[test]
fn test_hf_llama_pipeline() {
    let tmp = tempdir().unwrap();
    let model_dir = tmp.path().join("llama_hf");
    create_hf_llama_model(&model_dir);

    // 1. Ingestion check
    let hf = HfModel::load(&model_dir).unwrap();
    assert_eq!(hf.model_config.architecture, paraoxidizer::core::ModelArchitecture::Llama);
    assert!(hf.tensors.len() >= 20);

    // 2. Inspection command
    commands::run_inspect(model_dir.to_str().unwrap(), "text").unwrap();
    commands::run_inspect(model_dir.to_str().unwrap(), "json").unwrap();

    // 3. Calibration on agentic trace profile
    let cal_path = tmp.path().join("llama_agentic.poxcal");
    commands::run_calibrate(
        model_dir.to_str().unwrap(),
        None,
        "agentic",
        64,
        cal_path.to_str().unwrap(),
    )
    .unwrap();
    assert!(cal_path.exists());

    // 4. Parameter sensitivity analysis
    commands::run_analyze(
        model_dir.to_str().unwrap(),
        Some(cal_path.to_str().unwrap()),
        "text",
    )
    .unwrap();

    // 5. Adaptive Mixed Precision Optimization to .pox
    let pox_path = tmp.path().join("llama_opt.pox");
    commands::run_optimize(
        model_dir.to_str().unwrap(),
        Some("4GB"),
        Some("40ms"),
        Some(98.0),
        Some(cal_path.to_str().unwrap()),
        "auto",
        pox_path.to_str().unwrap(),
        "text",
    )
    .unwrap();
    assert!(pox_path.exists());

    // 6. Validation of generated .pox
    commands::run_validate(pox_path.to_str().unwrap()).unwrap();

    // 7. Ed25519 signing & verification
    let keypair = paraoxidizer::security::KeyPair::generate();
    let signed_pox_path = tmp.path().join("llama_opt_signed.pox");
    commands::run_sign(
        pox_path.to_str().unwrap(),
        &keypair.private_key_hex(),
        Some(signed_pox_path.to_str().unwrap()),
    )
    .unwrap();

    commands::run_verify(
        signed_pox_path.to_str().unwrap(),
        Some(&keypair.public_key_hex()),
    )
    .unwrap();

    // 8. Inference run
    commands::run_inference(
        signed_pox_path.to_str().unwrap(),
        "Explain memory safety in Rust.",
        16,
        0.7,
    )
    .unwrap();

    // 9. Benchmark
    commands::run_benchmark(
        Some(signed_pox_path.to_str().unwrap()),
        false,
        "Benchmarking latency.",
        16,
        "text",
    )
    .unwrap();
}

#[test]
fn test_hf_qwen_and_mistral() {
    let tmp = tempdir().unwrap();

    // Qwen
    let qwen_dir = tmp.path().join("qwen_hf");
    create_hf_qwen_model(&qwen_dir);
    let qwen = HfModel::load(&qwen_dir).unwrap();
    assert_eq!(qwen.model_config.architecture, paraoxidizer::core::ModelArchitecture::Qwen);

    let qwen_pox = tmp.path().join("qwen.pox");
    commands::run_quantize(
        qwen_dir.to_str().unwrap(),
        4,
        128,
        "automatic",
        "awq",
        qwen_pox.to_str().unwrap(),
    )
    .unwrap();
    assert!(qwen_pox.exists());
    commands::run_validate(qwen_pox.to_str().unwrap()).unwrap();

    // Mistral
    let mistral_dir = tmp.path().join("mistral_hf");
    create_hf_mistral_model(&mistral_dir);
    let mistral = HfModel::load(&mistral_dir).unwrap();
    assert_eq!(mistral.model_config.architecture, paraoxidizer::core::ModelArchitecture::Mistral);

    let mistral_pox = tmp.path().join("mistral.pox");
    commands::run_quantize(
        mistral_dir.to_str().unwrap(),
        8,
        0,
        "disabled",
        "min-max",
        mistral_pox.to_str().unwrap(),
    )
    .unwrap();
    assert!(mistral_pox.exists());
    commands::run_validate(mistral_pox.to_str().unwrap()).unwrap();

    // Compare
    commands::run_compare(&[
        qwen_pox.to_str().unwrap().to_string(),
        mistral_pox.to_str().unwrap().to_string(),
    ])
    .unwrap();
}

#[test]
fn test_hf_sharded_model() {
    let tmp = tempdir().unwrap();
    let sharded_dir = tmp.path().join("sharded_hf");
    create_hf_sharded_model(&sharded_dir);

    // Verify index parsing and loading from multiple shards
    let hf = HfModel::load(&sharded_dir).unwrap();
    assert!(hf.tensors.contains_key("model.embed_tokens.weight"));
    assert!(hf.tensors.contains_key("model.layers.0.q_proj.weight") || hf.tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(hf.tensors.contains_key("lm_head.weight"));

    let out_pox = tmp.path().join("sharded.pox");
    commands::run_quantize(
        sharded_dir.to_str().unwrap(),
        4,
        64,
        "conservative",
        "gptq",
        out_pox.to_str().unwrap(),
    )
    .unwrap();

    let pox = PoxFile::open(&out_pox).unwrap();
    assert!(pox.tensors.len() >= 4);
    commands::run_validate(out_pox.to_str().unwrap()).unwrap();
}

#[test]
fn test_model_diff() {
    let tmp = tempdir().unwrap();
    let model_dir = tmp.path().join("diff_base_hf");
    create_hf_llama_model(&model_dir);

    let pox_int4 = tmp.path().join("model_int4.pox");
    commands::run_quantize(
        model_dir.to_str().unwrap(),
        4,
        128,
        "automatic",
        "min-max",
        pox_int4.to_str().unwrap(),
    )
    .unwrap();

    let pox_int8 = tmp.path().join("model_int8.pox");
    commands::run_quantize(
        model_dir.to_str().unwrap(),
        8,
        0,
        "disabled",
        "min-max",
        pox_int8.to_str().unwrap(),
    )
    .unwrap();

    // Run diff
    commands::run_diff(pox_int8.to_str().unwrap(), pox_int4.to_str().unwrap()).unwrap();
}

#[test]
fn test_huggingface_hub_direct_remote_fetch() {
    let tmp = tempdir().unwrap();
    let repo_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";

    // Ingest directly from Hugging Face Hub
    let hf = HfModel::load(repo_id).unwrap();
    assert_eq!(hf.model_config.architecture, paraoxidizer::core::ModelArchitecture::Llama);
    assert_eq!(hf.tensors.len(), 21);

    // Run CLI inspect on remote repo ID
    commands::run_inspect(repo_id, "text").unwrap();

    // Quantize remote HF model to .pox
    let out_pox = tmp.path().join("remote_hf.pox");
    commands::run_quantize(
        repo_id,
        4,
        32,
        "automatic",
        "min-max",
        out_pox.to_str().unwrap(),
    )
    .unwrap();

    // Validate and verify
    commands::run_validate(out_pox.to_str().unwrap()).unwrap();
    commands::run_verify(out_pox.to_str().unwrap(), None).unwrap();
}
