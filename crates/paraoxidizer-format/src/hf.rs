use crate::safetensors::SafeTensorsModel;
use paraoxidizer_core::{
    arch::{ModelArchitecture, ModelConfig},
    error::{PoxError, Result},
    tensor::{DType, Shape},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

/// Hugging Face config.json schema representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfConfigJson {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub bos_token_id: Option<serde_json::Value>,
}

impl HfConfigJson {
    pub fn to_model_config(&self) -> ModelConfig {
        let arch_str = self
            .architectures
            .first()
            .cloned()
            .or_else(|| self.model_type.clone())
            .unwrap_or_else(|| "Llama".to_string());

        let architecture = ModelArchitecture::from_str_name(&arch_str);
        let hidden_size = self.hidden_size.unwrap_or(2048);
        let num_attention_heads = self.num_attention_heads.unwrap_or(32);
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(num_attention_heads);
        let intermediate_size = self
            .intermediate_size
            .unwrap_or_else(|| (hidden_size * 8) / 3);
        let num_hidden_layers = self.num_hidden_layers.unwrap_or(22);
        let vocab_size = self.vocab_size.unwrap_or(32000);
        let max_position_embeddings = self.max_position_embeddings.unwrap_or(4096);
        let rms_norm_eps = self.rms_norm_eps.unwrap_or(1e-5);
        let rope_theta = self.rope_theta.unwrap_or(10000.0);
        let tie_word_embeddings = self.tie_word_embeddings.unwrap_or(false);

        let eos_token_id = match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(2) as u32,
            Some(serde_json::Value::Array(arr)) => {
                arr.first().and_then(|v| v.as_u64()).unwrap_or(2) as u32
            }
            _ => 2,
        };

        let bos_token_id = match &self.bos_token_id {
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(1) as u32,
            _ => 1,
        };

        ModelConfig {
            architecture,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings,
            eos_token_id,
            bos_token_id,
        }
    }
}

/// Sharded SafeTensors index file (model.safetensors.index.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfShardedIndex {
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub weight_map: HashMap<String, String>,
}

/// Loader for Hugging Face model directories (single or sharded safetensors)
pub struct HfModel {
    pub model_config: ModelConfig,
    pub tensors: HashMap<String, (Shape, DType, Vec<f32>)>,
    pub base_path: PathBuf,
}

impl HfModel {
    /// Download a model repository from Hugging Face Hub if not already cached
    pub fn download_from_hub(repo_id: &str) -> Result<PathBuf> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let safe_repo_name = repo_id.replace('/', "_");
        let cache_dir = PathBuf::from(home)
            .join(".cache")
            .join("paraoxidizer")
            .join("hub")
            .join(safe_repo_name);

        std::fs::create_dir_all(&cache_dir)?;

        let config_file = cache_dir.join("config.json");
        let safetensors_file = cache_dir.join("model.safetensors");

        if config_file.exists() && safetensors_file.exists() {
            println!(
                "Using cached Hugging Face repository at {}",
                cache_dir.display()
            );
            return Ok(cache_dir);
        }

        println!(
            "Fetching Hugging Face repository '{}' from https://huggingface.co/{}...",
            repo_id, repo_id
        );

        // 1. Download config.json
        let config_url = format!("https://huggingface.co/{}/raw/main/config.json", repo_id);
        let resp = ureq::get(&config_url).call().map_err(|e| {
            PoxError::Format(format!(
                "Failed to fetch config.json from Hugging Face Hub: {e}"
            ))
        })?;

        let mut config_out = File::create(&config_file)?;
        let mut reader = resp.into_reader();
        std::io::copy(&mut reader, &mut config_out)?;

        // 2. Try downloading model.safetensors
        let st_url = format!(
            "https://huggingface.co/{}/resolve/main/model.safetensors",
            repo_id
        );
        let st_resp = ureq::get(&st_url).call();

        match st_resp {
            Ok(resp) => {
                println!("Downloading model.safetensors...");
                let mut st_out = File::create(&safetensors_file)?;
                let mut reader = resp.into_reader();
                std::io::copy(&mut reader, &mut st_out)?;
            }
            Err(_) => {
                // Try downloading model.safetensors.index.json
                let index_url = format!(
                    "https://huggingface.co/{}/raw/main/model.safetensors.index.json",
                    repo_id
                );
                let idx_resp = ureq::get(&index_url).call().map_err(|e| {
                    PoxError::Format(format!(
                        "Could not find model.safetensors or index on Hugging Face: {e}"
                    ))
                })?;

                let index_file = cache_dir.join("model.safetensors.index.json");
                let mut idx_out = File::create(&index_file)?;
                let mut reader = idx_resp.into_reader();
                std::io::copy(&mut reader, &mut idx_out)?;

                // Parse shards
                let mut contents = String::new();
                File::open(&index_file)?.read_to_string(&mut contents)?;
                let idx: HfShardedIndex = serde_json::from_str(&contents)?;
                let mut unique_shards = std::collections::HashSet::new();
                for shard in idx.weight_map.values() {
                    unique_shards.insert(shard.clone());
                }

                for shard in unique_shards {
                    let shard_file = cache_dir.join(&shard);
                    if !shard_file.exists() {
                        println!("Downloading shard {}...", shard);
                        let shard_url =
                            format!("https://huggingface.co/{}/resolve/main/{}", repo_id, shard);
                        let s_resp = ureq::get(&shard_url).call().map_err(|e| {
                            PoxError::Format(format!("Failed to download shard {}: {}", shard, e))
                        })?;
                        let mut s_out = File::create(&shard_file)?;
                        let mut r = s_resp.into_reader();
                        std::io::copy(&mut r, &mut s_out)?;
                    }
                }
            }
        }

        println!(
            "Successfully downloaded and cached repository to {}",
            cache_dir.display()
        );
        Ok(cache_dir)
    }

    /// Load from a local Hugging Face directory, local safetensors file, or Hugging Face Hub repo ID
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let path_str = path_ref.to_string_lossy();

        let effective_path = if path_ref.exists() {
            path_ref.to_path_buf()
        } else if path_str.starts_with("hf://")
            || (path_str.contains('/') && !path_str.starts_with('.') && !path_str.starts_with('/'))
        {
            let repo_id = path_str.trim_start_matches("hf://");
            Self::download_from_hub(repo_id)?
        } else {
            return Err(PoxError::Format(format!(
                "Path does not exist or is not an accessible Hugging Face repo: {}",
                path_ref.display()
            )));
        };

        let path = effective_path.as_path();

        if path.is_file() {
            // Single safetensors file
            let st = SafeTensorsModel::open(path)?;
            let model_config = ModelConfig::default();
            return Ok(Self {
                model_config,
                tensors: st.tensors().clone(),
                base_path: path.to_path_buf(),
            });
        }

        if !path.is_dir() {
            return Err(PoxError::Format(format!(
                "Path does not exist or is not accessible: {}",
                path.display()
            )));
        }

        // 1. Read config.json
        let config_path = path.join("config.json");
        let model_config = if config_path.exists() {
            let mut file = File::open(&config_path)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            let hf_cfg: HfConfigJson = serde_json::from_str(&contents)?;
            hf_cfg.to_model_config()
        } else {
            ModelConfig::default()
        };

        let mut all_tensors = HashMap::new();

        // 2. Check for sharded index: model.safetensors.index.json
        let index_path = path.join("model.safetensors.index.json");
        if index_path.exists() {
            let mut file = File::open(&index_path)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            let index: HfShardedIndex = serde_json::from_str(&contents)?;

            // Group weights by shard file
            let mut shards = HashMap::new();
            for (tensor_name, shard_file) in index.weight_map {
                shards
                    .entry(shard_file)
                    .or_insert_with(Vec::new)
                    .push(tensor_name);
            }

            for (shard_filename, _expected_tensors) in shards {
                let shard_path = path.join(&shard_filename);
                if shard_path.exists() {
                    let st = SafeTensorsModel::open(&shard_path)?;
                    for (t_name, data) in st.tensors() {
                        all_tensors.insert(t_name.clone(), data.clone());
                    }
                }
            }
        } else {
            // 3. Scan for any *.safetensors files in the directory
            let entries = std::fs::read_dir(path)?;
            let mut found_any = false;
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                    found_any = true;
                    let st = SafeTensorsModel::open(&p)?;
                    for (t_name, data) in st.tensors() {
                        all_tensors.insert(t_name.clone(), data.clone());
                    }
                }
            }

            if !found_any {
                return Err(PoxError::Format(format!(
                    "No .safetensors files found in directory {}",
                    path.display()
                )));
            }
        }

        Ok(Self {
            model_config,
            tensors: all_tensors,
            base_path: path.to_path_buf(),
        })
    }
}
