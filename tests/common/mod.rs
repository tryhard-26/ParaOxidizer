#![allow(dead_code)]

use half::f16;
use safetensors::tensor::{Dtype, TensorView};
use std::{collections::HashMap, fs::File, io::Write, path::Path};

/// Helper to serialize and write a SafeTensors file
pub fn write_safetensors_file<P: AsRef<Path>>(path: P, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    let mut views = HashMap::new();
    let mut raw_bytes_storage: Vec<Vec<u8>> = Vec::new();

    for (_name, _shape, floats) in tensors {
        // Encode as F16
        let mut raw = Vec::with_capacity(floats.len() * 2);
        for &f in floats {
            raw.extend_from_slice(&f16::from_f32(f).to_le_bytes());
        }
        raw_bytes_storage.push(raw);
    }

    for (i, (name, shape, _)) in tensors.iter().enumerate() {
        let view = TensorView::new(Dtype::F16, shape.clone(), &raw_bytes_storage[i]).unwrap();
        views.insert(name.to_string(), view);
    }

    let serialized = safetensors::serialize(&views, &None).unwrap();
    let mut file = File::create(path).unwrap();
    file.write_all(&serialized).unwrap();
}

/// Synthesize a realistic Hugging Face Llama model directory
pub fn create_hf_llama_model<P: AsRef<Path>>(dir: P) {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).unwrap();

    let config_json = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 1000,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "tie_word_embeddings": false,
        "bos_token_id": 1,
        "eos_token_id": 2
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config_json).unwrap(),
    )
    .unwrap();

    let weights = vec![
        (
            "model.embed_tokens.weight",
            vec![1000, 256],
            generate_weights(1000 * 256),
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![256],
            vec![1.0; 256],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![256, 256],
            generate_weights(256 * 256),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![64, 256],
            generate_weights(64 * 256),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![64, 256],
            generate_weights(64 * 256),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![256, 256],
            generate_weights(256 * 256),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![256],
            vec![1.0; 256],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![512, 256],
            generate_weights(512 * 256),
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![512, 256],
            generate_weights(512 * 256),
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![256, 512],
            generate_weights(256 * 512),
        ),
        (
            "model.layers.1.input_layernorm.weight",
            vec![256],
            vec![1.0; 256],
        ),
        (
            "model.layers.1.self_attn.q_proj.weight",
            vec![256, 256],
            generate_weights(256 * 256),
        ),
        (
            "model.layers.1.self_attn.k_proj.weight",
            vec![64, 256],
            generate_weights(64 * 256),
        ),
        (
            "model.layers.1.self_attn.v_proj.weight",
            vec![64, 256],
            generate_weights(64 * 256),
        ),
        (
            "model.layers.1.self_attn.o_proj.weight",
            vec![256, 256],
            generate_weights(256 * 256),
        ),
        (
            "model.layers.1.post_attention_layernorm.weight",
            vec![256],
            vec![1.0; 256],
        ),
        (
            "model.layers.1.mlp.gate_proj.weight",
            vec![512, 256],
            generate_weights(512 * 256),
        ),
        (
            "model.layers.1.mlp.up_proj.weight",
            vec![512, 256],
            generate_weights(512 * 256),
        ),
        (
            "model.layers.1.mlp.down_proj.weight",
            vec![256, 512],
            generate_weights(256 * 512),
        ),
        ("model.norm.weight", vec![256], vec![1.0; 256]),
        (
            "lm_head.weight",
            vec![1000, 256],
            generate_weights(1000 * 256),
        ),
    ];

    write_safetensors_file(dir.join("model.safetensors"), &weights);
}

/// Synthesize a Qwen-2.5 style Hugging Face model directory
pub fn create_hf_qwen_model<P: AsRef<Path>>(dir: P) {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).unwrap();

    let config_json = serde_json::json!({
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": true
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config_json).unwrap(),
    )
    .unwrap();

    let weights = vec![
        (
            "model.embed_tokens.weight",
            vec![1000, 256],
            generate_weights(1000 * 256),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![256, 256],
            generate_weights(256 * 256),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![64, 256],
            generate_weights(64 * 256),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![64, 256],
            generate_weights(64 * 256),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![256, 256],
            generate_weights(256 * 256),
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![256, 768],
            generate_weights(256 * 768),
        ),
        ("model.norm.weight", vec![256], vec![1.0; 256]),
    ];

    write_safetensors_file(dir.join("model.safetensors"), &weights);
}

/// Synthesize a Mistral style Hugging Face model directory
pub fn create_hf_mistral_model<P: AsRef<Path>>(dir: P) {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).unwrap();

    let config_json = serde_json::json!({
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 32000,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-5
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config_json).unwrap(),
    )
    .unwrap();

    let weights = vec![
        (
            "model.embed_tokens.weight",
            vec![500, 256],
            generate_weights(500 * 256),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![256, 256],
            generate_weights(256 * 256),
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![256, 512],
            generate_weights(256 * 512),
        ),
        ("model.norm.weight", vec![256], vec![1.0; 256]),
        (
            "lm_head.weight",
            vec![500, 256],
            generate_weights(500 * 256),
        ),
    ];

    write_safetensors_file(dir.join("model.safetensors"), &weights);
}

/// Synthesize a sharded Hugging Face model directory
pub fn create_hf_sharded_model<P: AsRef<Path>>(dir: P) {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).unwrap();

    let config_json = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 1000
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config_json).unwrap(),
    )
    .unwrap();

    let shard1_weights = vec![
        (
            "model.embed_tokens.weight",
            vec![1000, 256],
            generate_weights(1000 * 256),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![256, 256],
            generate_weights(256 * 256),
        ),
    ];
    let shard2_weights = vec![
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![256, 512],
            generate_weights(256 * 512),
        ),
        ("model.norm.weight", vec![256], vec![1.0; 256]),
        (
            "lm_head.weight",
            vec![1000, 256],
            generate_weights(1000 * 256),
        ),
    ];

    write_safetensors_file(
        dir.join("model-00001-of-00002.safetensors"),
        &shard1_weights,
    );
    write_safetensors_file(
        dir.join("model-00002-of-00002.safetensors"),
        &shard2_weights,
    );

    let index_json = serde_json::json!({
        "metadata": { "total_size": 1048576 },
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors"
        }
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index_json).unwrap(),
    )
    .unwrap();
}

fn generate_weights(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            // Introduce occasional extreme outliers to test outlier preservation
            if i % 1000 == 999 {
                12.5f32
            } else {
                (i as f32 * 0.013).sin() * 0.4
            }
        })
        .collect()
}
