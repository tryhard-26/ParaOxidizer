use serde::{Deserialize, Serialize};

/// Supported decoder-only transformer model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Llama,
    Qwen,
    Mistral,
    Gemma,
    Phi,
    Custom,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArchitecture::Llama => write!(f, "Llama"),
            ModelArchitecture::Qwen => write!(f, "Qwen"),
            ModelArchitecture::Mistral => write!(f, "Mistral"),
            ModelArchitecture::Gemma => write!(f, "Gemma"),
            ModelArchitecture::Phi => write!(f, "Phi"),
            ModelArchitecture::Custom => write!(f, "Custom"),
        }
    }
}

impl ModelArchitecture {
    pub fn from_str_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("llama") {
            ModelArchitecture::Llama
        } else if lower.contains("qwen") {
            ModelArchitecture::Qwen
        } else if lower.contains("mistral") || lower.contains("mixtral") {
            ModelArchitecture::Mistral
        } else if lower.contains("gemma") {
            ModelArchitecture::Gemma
        } else if lower.contains("phi") {
            ModelArchitecture::Phi
        } else {
            ModelArchitecture::Custom
        }
    }
}

/// Structural component of a Transformer model layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    Embedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionOutput,
    MlpGate,
    MlpUp,
    MlpDown,
    Norm,
    LmHead,
    Other,
}

impl ComponentType {
    pub fn classify_tensor_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("embed_tokens")
            || lower.contains("wte")
            || lower.contains("tok_embeddings")
        {
            ComponentType::Embedding
        } else if lower.contains("q_proj") || lower.contains("wq") {
            ComponentType::AttentionQ
        } else if lower.contains("k_proj") || lower.contains("wk") {
            ComponentType::AttentionK
        } else if lower.contains("v_proj") || lower.contains("wv") {
            ComponentType::AttentionV
        } else if lower.contains("o_proj") || lower.contains("wo") || lower.contains("out_proj") {
            ComponentType::AttentionOutput
        } else if lower.contains("gate_proj") || lower.contains("w1") {
            ComponentType::MlpGate
        } else if lower.contains("up_proj") || lower.contains("w3") {
            ComponentType::MlpUp
        } else if lower.contains("down_proj") || lower.contains("w2") {
            ComponentType::MlpDown
        } else if lower.contains("norm") || lower.contains("ln") {
            ComponentType::Norm
        } else if lower.contains("lm_head") {
            ComponentType::LmHead
        } else {
            ComponentType::Other
        }
    }
}

/// Full configuration hyper-parameters of an LLM
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: u32,
    #[serde(default)]
    pub bos_token_id: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Llama,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            eos_token_id: 2,
            bos_token_id: 1,
        }
    }
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        (self.hidden_size / self.num_attention_heads.max(1)).max(1)
    }

    pub fn total_parameters_approx(&self) -> u64 {
        // Embeddings: vocab_size * hidden_size
        let embed = (self.vocab_size * self.hidden_size) as u64;
        // Per layer:
        // Attn: (hidden_size * (num_heads + 2 * kv_heads) * head_dim) + (hidden_size * hidden_size)
        let q_dim = self.num_attention_heads * self.head_dim();
        let k_dim = self.num_key_value_heads * self.head_dim();
        let v_dim = self.num_key_value_heads * self.head_dim();
        let attn_weights = (self.hidden_size * (q_dim + k_dim + v_dim + self.hidden_size)) as u64;
        // FFN: Gate + Up + Down
        let ffn_weights = (3 * self.hidden_size * self.intermediate_size) as u64;
        // Norms (small)
        let layer_weights = (attn_weights + ffn_weights) * (self.num_hidden_layers as u64);
        let lm_head = if self.tie_word_embeddings {
            0
        } else {
            (self.vocab_size * self.hidden_size) as u64
        };
        embed + layer_weights + lm_head
    }
}
