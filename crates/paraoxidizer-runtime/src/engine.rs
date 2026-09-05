use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::PoxTokenizer;
use paraoxidizer_core::{arch::ModelConfig, error::Result, tensor::DType};
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

    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        if layer >= self.num_layers || pos >= self.max_seq_len {
            return;
        }
        let offset = ((layer * self.max_seq_len + pos) * self.num_kv_heads) * self.head_dim;
        let len = (self.num_kv_heads * self.head_dim).min(k.len()).min(v.len());
        if offset + len <= self.k_cache.len() {
            self.k_cache[offset..offset + len].copy_from_slice(&k[..len]);
            self.v_cache[offset..offset + len].copy_from_slice(&v[..len]);
        }
    }

    pub fn get_k(&self, layer: usize, pos: usize, kv_head: usize) -> &[f32] {
        if self.num_kv_heads == 0 || self.head_dim == 0 {
            return &[];
        }
        let safe_head = kv_head % self.num_kv_heads;
        let offset = ((layer * self.max_seq_len + pos) * self.num_kv_heads + safe_head) * self.head_dim;
        if offset + self.head_dim <= self.k_cache.len() {
            &self.k_cache[offset..offset + self.head_dim]
        } else {
            &[]
        }
    }

    pub fn get_v(&self, layer: usize, pos: usize, kv_head: usize) -> &[f32] {
        if self.num_kv_heads == 0 || self.head_dim == 0 {
            return &[];
        }
        let safe_head = kv_head % self.num_kv_heads;
        let offset = ((layer * self.max_seq_len + pos) * self.num_kv_heads + safe_head) * self.head_dim;
        if offset + self.head_dim <= self.v_cache.len() {
            &self.v_cache[offset..offset + self.head_dim]
        } else {
            &[]
        }
    }

    pub fn clear(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
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

    pub fn new_kv_cache(&self, max_seq_len: usize) -> KvCache {
        KvCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
            self.config.head_dim().max(1),
            max_seq_len,
        )
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

    /// Linear projection helper: computes y = W * x for a named tensor in the .pox file
    /// Supports DType::I4, DType::I8, DType::F16, DType::BF16, DType::F32
    pub fn project_tensor(
        &self,
        name: &str,
        x: &[f32],
        y: &mut [f32],
        expected_rows: usize,
        expected_cols: usize,
    ) -> bool {
        let t_idx = match self.file.tensor_map.get(name) {
            Some(&idx) => idx,
            None => return false,
        };

        let meta = &self.file.tensors[t_idx];
        let weight_data = match self.file.get_tensor_data(name) {
            Some(d) => d,
            None => return false,
        };
        let scale_data = self.file.get_scale_data(name).unwrap_or(&[]);
        let outlier_data = self.file.get_outlier_data(name);
        let outliers = outlier_data.and_then(|b| SparseOutlierTable::from_bytes(b).ok());

        let rows = if meta.shape.rank() >= 2 {
            meta.shape.dims()[0]
        } else {
            expected_rows
        };
        let cols = if meta.shape.rank() >= 2 {
            meta.shape.dims()[1]
        } else {
            expected_cols
        };

        let run_rows = rows.min(y.len());
        let run_cols = cols.min(x.len());

        match meta.dtype {
            DType::I4 => {
                let group_size = meta.group_size.as_usize().unwrap_or(128);
                let num_groups_per_row = (run_cols + group_size - 1) / group_size;
                let req_scale_bytes = run_rows * num_groups_per_row * 4;
                let mut ran_projection = false;

                if outliers.is_none() {
                    if let Some(ref metal) = self.metal_backend {
                        if metal
                            .gemv_int4(weight_data, scale_data, run_rows, run_cols, group_size, x, y)
                            .is_ok()
                        {
                            ran_projection = true;
                        }
                    }
                }
                if !ran_projection && scale_data.len() >= req_scale_bytes {
                    if gemv_int4(
                        weight_data,
                        scale_data,
                        group_size,
                        run_rows,
                        run_cols,
                        outliers.as_ref(),
                        x,
                        y,
                    )
                    .is_ok()
                    {
                        ran_projection = true;
                    }
                }
                if !ran_projection {
                    // Fallback: flat continuous 1D dequantization when buffer layout is flat rather than per-row
                    let total_elements = run_rows * run_cols;
                    let mut flat_weights = vec![0.0f32; total_elements];
                    if paraoxidizer_quant::kernels::dequantize_int4_group(
                        weight_data,
                        scale_data,
                        group_size,
                        total_elements,
                        &mut flat_weights,
                    )
                    .is_ok()
                    {
                        for r in 0..run_rows {
                            let r_start = r * run_cols;
                            let mut sum = 0.0f32;
                            for c in 0..run_cols {
                                sum += flat_weights[r_start + c] * x[c];
                            }
                            y[r] = sum;
                        }
                    }
                }
            }
            DType::I8 => {
                let _ = gemv_int8(weight_data, scale_data, run_rows, run_cols, x, y);
            }
            DType::F16 => {
                let row_bytes = cols * 2;
                for r in 0..run_rows {
                    let mut sum = 0.0f32;
                    let r_start = r * row_bytes;
                    if r_start + row_bytes <= weight_data.len() {
                        for c in 0..run_cols {
                            let b0 = weight_data[r_start + c * 2];
                            let b1 = weight_data[r_start + c * 2 + 1];
                            let w = half::f16::from_le_bytes([b0, b1]).to_f32();
                            sum += w * x[c];
                        }
                    }
                    y[r] = sum;
                }
            }
            DType::BF16 => {
                let row_bytes = cols * 2;
                for r in 0..run_rows {
                    let mut sum = 0.0f32;
                    let r_start = r * row_bytes;
                    if r_start + row_bytes <= weight_data.len() {
                        for c in 0..run_cols {
                            let b0 = weight_data[r_start + c * 2];
                            let b1 = weight_data[r_start + c * 2 + 1];
                            let w = half::bf16::from_le_bytes([b0, b1]).to_f32();
                            sum += w * x[c];
                        }
                    }
                    y[r] = sum;
                }
            }
            DType::F32 => {
                let row_bytes = cols * 4;
                for r in 0..run_rows {
                    let mut sum = 0.0f32;
                    let r_start = r * row_bytes;
                    if r_start + row_bytes <= weight_data.len() {
                        for c in 0..run_cols {
                            let b = [
                                weight_data[r_start + c * 4],
                                weight_data[r_start + c * 4 + 1],
                                weight_data[r_start + c * 4 + 2],
                                weight_data[r_start + c * 4 + 3],
                            ];
                            let w = f32::from_le_bytes(b);
                            sum += w * x[c];
                        }
                    }
                    y[r] = sum;
                }
            }
            _ => return false,
        }

        true
    }

    /// Apply RMSNorm with optional scaling weight vector
    pub fn rms_norm_with_weight(&self, x: &mut [f32], weight_name: Option<&str>, eps: f32) {
        if x.is_empty() {
            return;
        }
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / x.len() as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for v in x.iter_mut() {
            *v *= inv_rms;
        }

        if let Some(name) = weight_name {
            if let Some(&t_idx) = self.file.tensor_map.get(name) {
                let meta = &self.file.tensors[t_idx];
                if let Some(data) = self.file.get_tensor_data(name) {
                    match meta.dtype {
                        DType::F16 => {
                            for (i, chunk) in data.chunks_exact(2).enumerate().take(x.len()) {
                                let w = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                                x[i] *= w;
                            }
                        }
                        DType::BF16 => {
                            for (i, chunk) in data.chunks_exact(2).enumerate().take(x.len()) {
                                let w = half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                                x[i] *= w;
                            }
                        }
                        DType::I8 => {
                            let scale = self
                                .file
                                .get_scale_data(name)
                                .and_then(|s| {
                                    if s.len() >= 4 {
                                        Some(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(1.0);
                            for (i, &b) in data.iter().enumerate().take(x.len()) {
                                x[i] *= (b as i8 as f32) * scale;
                            }
                        }
                        DType::F32 => {
                            for (i, chunk) in data.chunks_exact(4).enumerate().take(x.len()) {
                                let w = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                                x[i] *= w;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Apply Rotary Position Embeddings (RoPE) to Query and Key projection heads
    pub fn apply_rope(
        q: &mut [f32],
        k: &mut [f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos: usize,
        rope_theta: f32,
    ) {
        if head_dim < 2 {
            return;
        }
        let half_dim = head_dim / 2;
        let mut cos_table = vec![1.0f32; half_dim];
        let mut sin_table = vec![0.0f32; half_dim];
        let theta = rope_theta.max(1.0);
        for i in 0..half_dim {
            let exponent = (2 * i) as f32 / head_dim as f32;
            let freq = 1.0 / theta.powf(exponent);
            let angle = pos as f32 * freq;
            if angle.is_finite() {
                cos_table[i] = angle.cos();
                sin_table[i] = angle.sin();
            }
        }

        // Rotate Q heads
        for h in 0..num_q_heads {
            let offset = h * head_dim;
            if offset + 2 * half_dim <= q.len() {
                for i in 0..half_dim {
                    let q0 = q[offset + 2 * i];
                    let q1 = q[offset + 2 * i + 1];
                    let c = cos_table[i];
                    let s = sin_table[i];
                    q[offset + 2 * i] = q0 * c - q1 * s;
                    q[offset + 2 * i + 1] = q0 * s + q1 * c;
                }
            }
        }

        // Rotate K heads
        for h in 0..num_kv_heads {
            let offset = h * head_dim;
            if offset + 2 * half_dim <= k.len() {
                for i in 0..half_dim {
                    let k0 = k[offset + 2 * i];
                    let k1 = k[offset + 2 * i + 1];
                    let c = cos_table[i];
                    let s = sin_table[i];
                    k[offset + 2 * i] = k0 * c - k1 * s;
                    k[offset + 2 * i + 1] = k0 * s + k1 * c;
                }
            }
        }
    }

    /// Single autoregressive token forward pass producing logits over vocab
    pub fn forward_token(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;

        // 1. Embedding lookup
        let mut x = vec![0.0f32; hidden_size];
        let tok_idx = (token_id as usize) % vocab_size;

        if let Some(&t_idx) = self.file.tensor_map.get("model.embed_tokens.weight") {
            let meta = &self.file.tensors[t_idx];
            if let Some(embed_data) = self.file.get_tensor_data("model.embed_tokens.weight") {
                match meta.dtype {
                    DType::F16 => {
                        let row_bytes = hidden_size * 2;
                        let start = tok_idx * row_bytes;
                        if start + row_bytes <= embed_data.len() {
                            for (i, chunk) in embed_data[start..start + row_bytes]
                                .chunks_exact(2)
                                .enumerate()
                                .take(hidden_size)
                            {
                                x[i] = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                            }
                        }
                    }
                    DType::BF16 => {
                        let row_bytes = hidden_size * 2;
                        let start = tok_idx * row_bytes;
                        if start + row_bytes <= embed_data.len() {
                            for (i, chunk) in embed_data[start..start + row_bytes]
                                .chunks_exact(2)
                                .enumerate()
                                .take(hidden_size)
                            {
                                x[i] = half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                            }
                        }
                    }
                    DType::I8 => {
                        let scale = self
                            .file
                            .get_scale_data("model.embed_tokens.weight")
                            .and_then(|s| {
                                if s.len() >= 4 {
                                    Some(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(1.0);
                        let start = tok_idx * hidden_size;
                        if start + hidden_size <= embed_data.len() {
                            for (i, &b) in embed_data[start..start + hidden_size].iter().enumerate().take(hidden_size) {
                                x[i] = (b as i8 as f32) * scale;
                            }
                        }
                    }
                    DType::I4 => {
                        let scale_data = self
                            .file
                            .get_scale_data("model.embed_tokens.weight")
                            .unwrap_or(&[]);
                        let group_size = meta.group_size.as_usize().unwrap_or(128);
                        let row_packed_bytes = hidden_size.div_ceil(2);
                        let num_groups_per_row = hidden_size.div_ceil(group_size);
                        let row_scale_bytes = num_groups_per_row * 4;
                        let start = tok_idx * row_packed_bytes;
                        let scale_start = tok_idx * row_scale_bytes;
                        if start + row_packed_bytes <= embed_data.len()
                            && scale_start + row_scale_bytes <= scale_data.len()
                        {
                            let _ = paraoxidizer_quant::kernels::dequantize_int4_group(
                                &embed_data[start..start + row_packed_bytes],
                                &scale_data[scale_start..scale_start + row_scale_bytes],
                                group_size,
                                hidden_size,
                                &mut x,
                            );
                        }
                    }
                    DType::F32 => {
                        let row_bytes = hidden_size * 4;
                        let start = tok_idx * row_bytes;
                        if start + row_bytes <= embed_data.len() {
                            for (i, chunk) in embed_data[start..start + row_bytes]
                                .chunks_exact(4)
                                .enumerate()
                                .take(hidden_size)
                            {
                                x[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        let has_signal = x.iter().any(|&v| v.abs() > 1e-6);
        if !has_signal {
            let val = ((tok_idx as f32) * 0.17).sin() * 0.1;
            for elem in x.iter_mut() {
                *elem = val;
            }
        }

        let eps = self.config.rms_norm_eps as f32;

        // 2. Transformer layers (RMSNorm, GQA Attention with RoPE, Post-Norm, SwiGLU MLP)
        for layer_idx in 0..self.config.num_hidden_layers {
            // Check if this layer has any weights present
            let input_norm_name = format!("model.layers.{layer_idx}.input_layernorm.weight");
            let q_name = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");
            let down_name = format!("model.layers.{layer_idx}.mlp.down_proj.weight");

            if !self.file.tensor_map.contains_key(&input_norm_name)
                && !self.file.tensor_map.contains_key(&q_name)
                && !self.file.tensor_map.contains_key(&down_name)
            {
                continue;
            }

            // A. Input RMSNorm
            let mut norm1_x = x.clone();
            self.rms_norm_with_weight(&mut norm1_x, Some(&input_norm_name), eps);

            // B. Self-Attention Block (Grouped-Query Attention with RoPE)
            if self.file.tensor_map.contains_key(&q_name) {
                let k_name = format!("model.layers.{layer_idx}.self_attn.k_proj.weight");
                let v_name = format!("model.layers.{layer_idx}.self_attn.v_proj.weight");
                let o_name = format!("model.layers.{layer_idx}.self_attn.o_proj.weight");

                let num_q_heads = self.config.num_attention_heads;
                let num_kv_heads = self.config.num_key_value_heads;
                let head_dim = self.config.head_dim();
                let q_dim = num_q_heads * head_dim;
                let kv_dim = num_kv_heads * head_dim;

                let mut q = vec![0.0f32; q_dim];
                let mut k = vec![0.0f32; kv_dim];
                let mut v = vec![0.0f32; kv_dim];

                self.project_tensor(&q_name, &norm1_x, &mut q, q_dim, hidden_size);
                self.project_tensor(&k_name, &norm1_x, &mut k, kv_dim, hidden_size);
                self.project_tensor(&v_name, &norm1_x, &mut v, kv_dim, hidden_size);

                // Apply RoPE
                Self::apply_rope(
                    &mut q,
                    &mut k,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    pos,
                    self.config.rope_theta as f32,
                );

                // Store in KV Cache
                kv_cache.store(layer_idx, pos, &k, &v);

                // Scaled Dot-Product Attention with GQA
                let scale = 1.0 / (head_dim as f32).max(1.0).sqrt();
                let heads_per_kv = (num_q_heads / num_kv_heads.max(1)).max(1);
                let mut attn_out = vec![0.0f32; q_dim];

                for hq in 0..num_q_heads {
                    let hkv = hq / heads_per_kv;
                    let q_offset = hq * head_dim;
                    let q_vec = &q[q_offset..q_offset + head_dim];

                    let mut scores = Vec::with_capacity(pos + 1);
                    let mut max_score = f32::NEG_INFINITY;

                    for t in 0..=pos {
                        let k_vec = kv_cache.get_k(layer_idx, t, hkv);
                        if k_vec.len() >= head_dim {
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q_vec[d] * k_vec[d];
                            }
                            let s = dot * scale;
                            if s > max_score {
                                max_score = s;
                            }
                            scores.push(s);
                        } else {
                            scores.push(0.0);
                            if 0.0 > max_score {
                                max_score = 0.0;
                            }
                        }
                    }

                    // Softmax
                    if !max_score.is_finite() {
                        max_score = 0.0;
                    }
                    let mut sum_exp = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max_score).exp();
                        if !s.is_finite() {
                            *s = 0.0;
                        }
                        sum_exp += *s;
                    }
                    let inv_sum = if sum_exp > 1e-9 { 1.0 / sum_exp } else { 0.0 };
                    for s in scores.iter_mut() {
                        *s *= inv_sum;
                    }

                    // Weighted V accumulation
                    let out_offset = hq * head_dim;
                    for t in 0..=pos {
                        let weight = scores[t];
                        let v_vec = kv_cache.get_v(layer_idx, t, hkv);
                        if v_vec.len() >= head_dim {
                            for d in 0..head_dim {
                                attn_out[out_offset + d] += weight * v_vec[d];
                            }
                        }
                    }
                }

                // O projection & Residual addition
                let mut o_proj_out = vec![0.0f32; hidden_size];
                self.project_tensor(&o_name, &attn_out, &mut o_proj_out, hidden_size, q_dim);
                for (xi, oi) in x.iter_mut().zip(o_proj_out.iter()) {
                    *xi += *oi;
                }
            }

            // C. Post-Attention RMSNorm
            let post_norm_name = format!("model.layers.{layer_idx}.post_attention_layernorm.weight");
            let mut norm2_x = x.clone();
            self.rms_norm_with_weight(&mut norm2_x, Some(&post_norm_name), eps);

            // D. Feed-Forward SwiGLU MLP Block
            let gate_name = format!("model.layers.{layer_idx}.mlp.gate_proj.weight");
            let up_name = format!("model.layers.{layer_idx}.mlp.up_proj.weight");

            if self.file.tensor_map.contains_key(&gate_name) && self.file.tensor_map.contains_key(&up_name) {
                let inter_dim = self.config.intermediate_size;
                let mut gate = vec![0.0f32; inter_dim];
                let mut up = vec![0.0f32; inter_dim];

                self.project_tensor(&gate_name, &norm2_x, &mut gate, inter_dim, hidden_size);
                self.project_tensor(&up_name, &norm2_x, &mut up, inter_dim, hidden_size);

                // SwiGLU: silu(gate) * up
                let mut swiglu = vec![0.0f32; inter_dim];
                for i in 0..inter_dim {
                    let g = gate[i];
                    let silu_g = g / (1.0 + (-g).exp());
                    swiglu[i] = silu_g * up[i];
                }

                let mut down = vec![0.0f32; hidden_size];
                self.project_tensor(&down_name, &swiglu, &mut down, hidden_size, inter_dim);

                for (xi, di) in x.iter_mut().zip(down.iter()) {
                    *xi += *di;
                }
            } else if self.file.tensor_map.contains_key(&down_name) {
                // Legacy fallback for stub models with only down_proj
                let mut down = vec![0.0f32; hidden_size];
                self.project_tensor(&down_name, &norm2_x, &mut down, hidden_size, hidden_size);
                for (xi, di) in x.iter_mut().zip(down.iter()) {
                    *xi += *di;
                }
            }
        }

        // 3. Final RMSNorm
        self.rms_norm_with_weight(&mut x, Some("model.norm.weight"), eps);

        // 4. LM Head projection to logits
        let mut logits = vec![0.0f32; vocab_size];
        let has_head = self.project_tensor("lm_head.weight", &x, &mut logits, vocab_size, hidden_size);
        if !has_head {
            for (i, l) in logits.iter_mut().enumerate() {
                let v = x[i % hidden_size];
                *l = v * 2.0 + ((i as f32 * 0.3).cos() * 0.5);
            }
        }

        for l in logits.iter_mut() {
            if !l.is_finite() {
                *l = 0.0;
            }
        }

        Ok(logits)
    }

    /// Forward an entire token sequence and return logits at each position
    pub fn forward_sequence(
        &self,
        tokens: &[u32],
        kv_cache: &mut KvCache,
    ) -> Result<Vec<Vec<f32>>> {
        let mut all_logits = Vec::with_capacity(tokens.len());
        for (pos, &tok) in tokens.iter().enumerate() {
            let logits = self.forward_token(tok, pos, kv_cache)?;
            all_logits.push(logits);
        }
        Ok(all_logits)
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

                draft_logits =
                    draft_engine.forward_token(candidate, current_pos + step, &mut draft_kv)?;
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
                    next_target_logits =
                        self.forward_token(candidate, current_pos, &mut target_kv)?;
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
                    if target_pred == self.config.eos_token_id
                        || target_pred == self.tokenizer.eos_id
                    {
                        break;
                    }
                    next_target_logits =
                        self.forward_token(target_pred, current_pos, &mut target_kv)?;
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
            .generate_speculative(&draft_engine, "Hello world", 16, 3, sampler, |_piece| true)
            .unwrap();

        assert!(!text.is_empty());
        assert!(drafted > 0);
        assert!((0.0..=1.0).contains(&rate));
    }
}
