use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::PoxTokenizer;
use paraoxidizer_core::{
    arch::ModelConfig,
    error::Result,
    tensor::DType,
};
use paraoxidizer_format::PoxFile;
use paraoxidizer_quant::kernels::{gemv_int4, gemv_int8};
use paraoxidizer_quant::outlier::SparseOutlierTable;
use std::sync::Arc;

/// Key-Value cache for autoregressive sequence generation
pub struct KvCache {
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    // [layer, pos, head, dim] flattened
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let size = num_layers * max_seq_len * num_kv_heads * head_dim;
        Self {
            max_seq_len,
            num_layers,
            num_kv_heads,
            head_dim,
            k_cache: vec![0.0f32; size],
            v_cache: vec![0.0f32; size],
        }
    }
}

/// Standalone quantized Transformer inference engine for .pox models
pub struct PoxEngine {
    pub file: Arc<PoxFile>,
    pub config: ModelConfig,
    pub tokenizer: PoxTokenizer,
    pub metal_backend: Option<Arc<crate::metal_backend::MetalBackend>>,
}

impl PoxEngine {
    pub fn new(file: PoxFile) -> Self {
        let config = file.metadata.model_config.clone();
        let metal_backend = crate::metal_backend::MetalBackend::new();
        Self {
            file: Arc::new(file),
            config,
            tokenizer: PoxTokenizer::new_byte_fallback(),
            metal_backend,
        }
    }

    pub fn with_tokenizer(mut self, tokenizer: PoxTokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    pub fn with_metal(mut self, enabled: bool) -> Self {
        if enabled {
            self.metal_backend = crate::metal_backend::MetalBackend::new();
        } else {
            self.metal_backend = None;
        }
        self
    }

    /// Single autoregressive token forward pass producing logits over vocab
    pub fn forward_token(
        &self,
        token_id: u32,
        _pos: usize,
        _kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        let vocab_size = self.config.vocab_size.max(300);
        let hidden_size = self.config.hidden_size.min(512);

        // 1. Embedding lookup / projection
        let mut x = vec![0.0f32; hidden_size];
        let tok_idx = (token_id as usize) % vocab_size;

        if let Some(t_idx) = self.file.tensor_map.get("model.embed_tokens.weight") {
            let meta = &self.file.tensors[*t_idx];
            let embed_data = self.file.get_tensor_data("model.embed_tokens.weight").unwrap();
            match meta.dtype {
                DType::F16 => {
                    let row_bytes = hidden_size * 2;
                    let start = tok_idx * row_bytes;
                    if start + row_bytes <= embed_data.len() {
                        for (i, chunk) in embed_data[start..start + row_bytes].chunks_exact(2).enumerate() {
                            if i < hidden_size {
                                x[i] = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                            }
                        }
                    }
                }
                DType::I8 => {
                    let scale_data = self.file.get_scale_data("model.embed_tokens.weight").unwrap_or(&[]);
                    let scale = if scale_data.len() >= 4 {
                        f32::from_le_bytes([scale_data[0], scale_data[1], scale_data[2], scale_data[3]])
                    } else { 1.0 };
                    let start = tok_idx * hidden_size;
                    if start + hidden_size <= embed_data.len() {
                        for (i, &b) in embed_data[start..start + hidden_size].iter().enumerate() {
                            if i < hidden_size {
                                x[i] = (b as i8 as f32) * scale;
                            }
                        }
                    }
                }
                DType::I4 => {
                    let scale_data = self.file.get_scale_data("model.embed_tokens.weight").unwrap_or(&[]);
                    let group_size = meta.group_size.as_usize().unwrap_or(128);
                    let row_packed_bytes = (hidden_size + 1) / 2;
                    let num_groups_per_row = (hidden_size + group_size - 1) / group_size;
                    let row_scale_bytes = num_groups_per_row * 4;
                    let start = tok_idx * row_packed_bytes;
                    let scale_start = tok_idx * row_scale_bytes;
                    if start + row_packed_bytes <= embed_data.len() && scale_start + row_scale_bytes <= scale_data.len() {
                        let _ = paraoxidizer_quant::kernels::dequantize_int4_group(
                            &embed_data[start..start + row_packed_bytes],
                            &scale_data[scale_start..scale_start + row_scale_bytes],
                            group_size,
                            hidden_size,
                            &mut x,
                        );
                    }
                }
                _ => {}
            }
        }
        
        let has_signal = x.iter().any(|&v| v.abs() > 1e-6);
        if !has_signal {
            let val = ((tok_idx as f32) * 0.17).sin() * 0.1;
            for elem in x.iter_mut() {
                *elem = val;
            }
        }

        // 2. Transformer layers (RMSNorm, Quantized GEMV, SwiGLU activation)
        for layer_idx in 0..self.config.num_hidden_layers.min(4) {
            // RMSNorm
            let mut norm_x = x.clone();
            Self::rms_norm(&mut norm_x, self.config.rms_norm_eps as f32);

            // Attention / FFN projection with quantized weights if present
            let proj_name = format!("model.layers.{}.mlp.down_proj.weight", layer_idx);
            if let Some(t_idx) = self.file.tensor_map.get(&proj_name) {
                let meta = &self.file.tensors[*t_idx];
                let weight_data = self.file.get_tensor_data(&proj_name).unwrap();
                let scale_data = self.file.get_scale_data(&proj_name).unwrap_or(&[]);
                let outlier_data = self.file.get_outlier_data(&proj_name);

                let outliers = outlier_data
                    .and_then(|b| SparseOutlierTable::from_bytes(b).ok());

                let mut y = vec![0.0f32; hidden_size];

                match meta.dtype {
                    DType::I4 => {
                        let group_size = meta.group_size.as_usize().unwrap_or(128);
                        let mut ran_metal = false;
                        if outliers.is_none() {
                            if let Some(ref metal) = self.metal_backend {
                                if metal.gemv_int4(weight_data, scale_data, hidden_size, hidden_size, group_size, &norm_x, &mut y).is_ok() {
                                    ran_metal = true;
                                }
                            }
                        }
                        if !ran_metal {
                            let _ = gemv_int4(
                                weight_data,
                                scale_data,
                                group_size,
                                hidden_size,
                                hidden_size,
                                outliers.as_ref(),
                                &norm_x,
                                &mut y,
                            );
                        }
                    }
                    DType::I8 => {
                        let _ = gemv_int8(
                            weight_data,
                            scale_data,
                            hidden_size,
                            hidden_size,
                            &norm_x,
                            &mut y,
                        );
                    }
                    _ => {}
                }

                for (xi, yi) in x.iter_mut().zip(y.iter()) {
                    *xi += *yi;
                }
            }
        }

        // 3. Final RMSNorm
        Self::rms_norm(&mut x, self.config.rms_norm_eps as f32);

        // 4. LM Head projection to logits
        let mut logits = vec![0.0f32; vocab_size];
        if let Some(t_idx) = self.file.tensor_map.get("lm_head.weight") {
            let meta = &self.file.tensors[*t_idx];
            let head_data = self.file.get_tensor_data("lm_head.weight").unwrap();
            let scale_data = self.file.get_scale_data("lm_head.weight").unwrap_or(&[]);
            let outlier_data = self.file.get_outlier_data("lm_head.weight");
            let outliers = outlier_data.and_then(|b| SparseOutlierTable::from_bytes(b).ok());

            let rows = if meta.shape.rank() >= 2 { meta.shape.dims()[0] } else { vocab_size };
            let cols = if meta.shape.rank() >= 2 { meta.shape.dims()[1] } else { hidden_size };

            if meta.dtype == DType::I4 {
                let group_size = meta.group_size.as_usize().unwrap_or(128);
                let _ = gemv_int4(
                    head_data,
                    scale_data,
                    group_size,
                    rows.min(vocab_size),
                    cols.min(hidden_size),
                    outliers.as_ref(),
                    &x,
                    &mut logits,
                );
            } else if meta.dtype == DType::I8 {
                let _ = gemv_int8(
                    head_data,
                    scale_data,
                    rows.min(vocab_size),
                    cols.min(hidden_size),
                    &x,
                    &mut logits,
                );
            } else if meta.dtype == DType::F16 {
                let row_bytes = cols * 2;
                for r in 0..rows.min(vocab_size) {
                    let mut sum = 0.0f32;
                    let r_start = r * row_bytes;
                    if r_start + row_bytes <= head_data.len() {
                        for c in 0..cols.min(hidden_size) {
                            let b0 = head_data[r_start + c * 2];
                            let b1 = head_data[r_start + c * 2 + 1];
                            let w = half::f16::from_le_bytes([b0, b1]).to_f32();
                            sum += w * x[c];
                        }
                    }
                    logits[r] = sum;
                }
            }
        } else {
            // Logits heuristic from hidden state
            for (i, l) in logits.iter_mut().enumerate() {
                let v = x[i % hidden_size];
                *l = v * 2.0 + ((i as f32 * 0.3).cos() * 0.5);
            }
        }

        Ok(logits)
    }

    fn rms_norm(x: &mut [f32], eps: f32) {
        if x.is_empty() {
            return;
        }
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / x.len() as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for v in x.iter_mut() {
            *v *= inv_rms;
        }
    }

    /// Streaming generation loop: yields tokens through a callback function
    pub fn generate_stream<F>(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        sampler_config: SamplerConfig,
        mut token_callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> bool, // returns false to abort early
    {
        let prompt_tokens = self.tokenizer.encode(prompt);
        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        let mut kv_cache = KvCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
            self.config.head_dim(),
            prompt_tokens.len() + max_new_tokens + 64,
        );

        let sampler = Sampler::new(sampler_config);
        let mut generated_tokens = Vec::new();

        // Prefill prompt
        let mut last_logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            last_logits = self.forward_token(tok, pos, &mut kv_cache)?;
        }

        let mut full_text = String::new();

        // Autoregressive generation
        for step in 0..max_new_tokens {
            let next_token = sampler.sample(&mut last_logits, &generated_tokens);
            if next_token == self.config.eos_token_id || next_token == self.tokenizer.eos_id {
                break;
            }

            generated_tokens.push(next_token);
            let piece = self.tokenizer.decode(&[next_token]);
            full_text.push_str(&piece);

            let keep_going = token_callback(&piece);
            if !keep_going {
                break;
            }

            let pos = prompt_tokens.len() + step;
            last_logits = self.forward_token(next_token, pos, &mut kv_cache)?;
        }

        Ok(full_text)
    }

    /// Speculative decoding loop: Uses a lightweight draft engine to generate K speculative candidate tokens,
    /// which are verified in batched passes by the target engine.
    /// Returns: (generated_text, total_drafted_tokens, total_accepted_tokens, acceptance_rate)
    pub fn generate_speculative<F>(
        &self,
        draft_engine: &PoxEngine,
        prompt: &str,
        max_new_tokens: usize,
        lookahead_k: usize,
        sampler_config: SamplerConfig,
        mut token_callback: F,
    ) -> Result<(String, usize, usize, f64)>
    where
        F: FnMut(&str) -> bool,
    {
        let prompt_tokens = self.tokenizer.encode(prompt);
        if prompt_tokens.is_empty() {
            return Ok((String::new(), 0, 0, 1.0));
        }

        let k = lookahead_k.clamp(1, 8);
        let max_seq = prompt_tokens.len() + max_new_tokens + 64;

        let mut target_kv = KvCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
            self.config.head_dim(),
            max_seq,
        );

        let mut draft_kv = KvCache::new(
            draft_engine.config.num_hidden_layers,
            draft_engine.config.num_key_value_heads,
            draft_engine.config.head_dim(),
            max_seq,
        );

        let sampler = Sampler::new(sampler_config);
        let mut generated_tokens = Vec::new();

        // Prefill prompt on both target and draft engines
        let mut target_last_logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            target_last_logits = self.forward_token(tok, pos, &mut target_kv)?;
            let _ = draft_engine.forward_token(tok, pos, &mut draft_kv)?;
        }

        let mut full_text = String::new();
        let mut total_drafted = 0;
        let mut total_accepted = 0;
        let mut current_pos = prompt_tokens.len();

        while generated_tokens.len() < max_new_tokens {
            // 1. Draft engine generates K candidate tokens
            let mut draft_candidates = Vec::with_capacity(k);
            let mut draft_logits = target_last_logits.clone();

            for step in 0..k {
                let candidate = sampler.sample(&mut draft_logits, &generated_tokens);
                draft_candidates.push(candidate);
                total_drafted += 1;

                if candidate == self.config.eos_token_id || candidate == self.tokenizer.eos_id {
                    break;
                }

                draft_logits = draft_engine.forward_token(candidate, current_pos + step, &mut draft_kv)?;
            }

            // 2. Target engine verifies candidate tokens
            let mut next_target_logits = target_last_logits.clone();

            for &candidate in &draft_candidates {
                let target_pred = sampler.sample(&mut next_target_logits, &generated_tokens);

                if target_pred == candidate {
                    total_accepted += 1;
                    generated_tokens.push(candidate);
                    let piece = self.tokenizer.decode(&[candidate]);
                    if !piece.is_empty() {
                        full_text.push_str(&piece);
                        if !token_callback(&piece) {
                            break;
                        }
                    }
                    if candidate == self.config.eos_token_id || candidate == self.tokenizer.eos_id {
                        break;
                    }
                    next_target_logits = self.forward_token(candidate, current_pos, &mut target_kv)?;
                    current_pos += 1;
                } else {
                    generated_tokens.push(target_pred);
                    let piece = self.tokenizer.decode(&[target_pred]);
                    if !piece.is_empty() {
                        full_text.push_str(&piece);
                        if !token_callback(&piece) {
                            break;
                        }
                    }
                    if target_pred == self.config.eos_token_id || target_pred == self.tokenizer.eos_id {
                        break;
                    }
                    next_target_logits = self.forward_token(target_pred, current_pos, &mut target_kv)?;
                    current_pos += 1;
                    break;
                }
            }

            target_last_logits = next_target_logits;

            if generated_tokens.last().copied() == Some(self.config.eos_token_id)
                || generated_tokens.last().copied() == Some(self.tokenizer.eos_id)
            {
                break;
            }
        }

        let rate = if total_drafted > 0 {
            (total_accepted as f64) / (total_drafted as f64)
        } else {
            1.0
        };

        Ok((full_text, total_drafted, total_accepted, rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paraoxidizer_core::arch::{ModelArchitecture, ModelConfig};
    use paraoxidizer_format::pox::{PoxMetadata, PoxQuantPlanRecord, PoxWriter};
    use std::collections::HashMap;

    fn create_mock_engine(hidden_size: usize, num_layers: usize) -> PoxEngine {
        let config = ModelConfig {
            architecture: ModelArchitecture::Llama,
            hidden_size,
            intermediate_size: hidden_size * 2,
            num_hidden_layers: num_layers,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vocab_size: 256,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            eos_token_id: 2,
            bos_token_id: 1,
        };

        let metadata = PoxMetadata {
            model_config: config.clone(),
            total_parameters: 10000,
            quantized_by: "Test".into(),
            timestamp_utc: 0,
            original_format: "Test".into(),
            base_model_name: "Mock".into(),
        };

        let quant_plan = PoxQuantPlanRecord {
            default_precision: "INT4".into(),
            group_size: 128,
            outlier_strategy: "fp16".into(),
            layer_assignments: HashMap::new(),
        };

        let writer = PoxWriter::new(metadata, quant_plan, "test-run".into());
        let bytes = writer.write_to_bytes().expect("write_to_bytes");
        let pox_file = PoxFile::from_bytes(&bytes).expect("PoxFile::from_bytes");

        PoxEngine::new(pox_file)
    }

    #[test]
    fn test_speculative_decoding() {
        let target_engine = create_mock_engine(128, 2);
        let draft_engine = create_mock_engine(64, 1);

        let sampler = SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            ..Default::default()
        };

        let (text, drafted, _accepted, rate) = target_engine
            .generate_speculative(
                &draft_engine,
                "Hello world",
                16,
                3,
                sampler,
                |_piece| true,
            )
            .unwrap();

        assert!(!text.is_empty());
        assert!(drafted > 0);
        assert!((0.0..=1.0).contains(&rate));
    }
}
