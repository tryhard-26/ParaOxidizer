use paraoxidizer_core::error::{PoxError, Result};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::Read, path::Path};

/// Fast BPE / Byte-level tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoxTokenizer {
    pub vocab: HashMap<String, u32>,
    pub inv_vocab: HashMap<u32, String>,
    pub bos_id: u32,
    pub eos_id: u32,
}

impl Default for PoxTokenizer {
    fn default() -> Self {
        Self::new_byte_fallback()
    }
}

impl PoxTokenizer {
    /// Built-in fallback byte-level tokenizer: guarantees any text is reversible without external files
    pub fn new_byte_fallback() -> Self {
        let mut vocab = HashMap::new();
        let mut inv_vocab = HashMap::new();

        // Special tokens
        vocab.insert("<|pad|>".to_string(), 0);
        inv_vocab.insert(0, "<|pad|>".to_string());
        vocab.insert("<|bos|>".to_string(), 1);
        inv_vocab.insert(1, "<|bos|>".to_string());
        vocab.insert("<|eos|>".to_string(), 2);
        inv_vocab.insert(2, "<|eos|>".to_string());

        // Common subwords / ASCII bytes
        for b in 0..=255u8 {
            let s = (b as char).to_string();
            let id = (b as u32) + 3;
            vocab.insert(s.clone(), id);
            inv_vocab.insert(id, s);
        }

        Self {
            vocab,
            inv_vocab,
            bos_id: 1,
            eos_id: 2,
        }
    }

    /// Load from Hugging Face tokenizer.json if present
    pub fn from_hf_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        #[derive(Deserialize)]
        struct HfTokenizerFormat {
            model: Option<HfModelSection>,
        }
        #[derive(Deserialize)]
        struct HfModelSection {
            vocab: Option<HashMap<String, u32>>,
        }

        let parsed: HfTokenizerFormat = serde_json::from_str(&contents)
            .map_err(|e| PoxError::Format(format!("Failed to parse tokenizer.json: {e}")))?;

        if let Some(m) = parsed.model {
            if let Some(vocab) = m.vocab {
                let mut inv_vocab = HashMap::new();
                for (k, v) in &vocab {
                    inv_vocab.insert(*v, k.clone());
                }
                return Ok(Self {
                    vocab,
                    inv_vocab,
                    bos_id: 1,
                    eos_id: 2,
                });
            }
        }

        Ok(Self::new_byte_fallback())
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        for c in text.chars() {
            let s = c.to_string();
            if let Some(&id) = self.vocab.get(&s) {
                tokens.push(id);
            } else {
                // Byte fallback
                let mut buf = [0u8; 4];
                let enc = c.encode_utf8(&mut buf);
                for &b in enc.as_bytes() {
                    let byte_str = (b as char).to_string();
                    let id = self.vocab.get(&byte_str).copied().unwrap_or(b as u32 + 3);
                    tokens.push(id);
                }
            }
        }

        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut out = String::new();
        for &tok in tokens {
            if tok == self.eos_id || tok == self.bos_id || tok == 0 {
                continue;
            }
            if let Some(s) = self.inv_vocab.get(&tok) {
                out.push_str(s);
            } else if (3..=258).contains(&tok) {
                let b = (tok - 3) as u8;
                out.push(b as char);
            }
        }
        out
    }
}
