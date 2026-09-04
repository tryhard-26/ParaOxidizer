use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub stop_sequences: Vec<String>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            stop_sequences: Vec::new(),
        }
    }
}

pub struct Sampler {
    config: SamplerConfig,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self { config }
    }

    /// Sample the next token from raw logits given previously generated tokens
    pub fn sample(&self, logits: &mut [f32], generated_tokens: &[u32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        // 1. Repetition penalty
        if (self.config.repetition_penalty - 1.0).abs() > 1e-5 {
            for &tok in generated_tokens {
                let idx = tok as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= self.config.repetition_penalty;
                    } else {
                        logits[idx] *= self.config.repetition_penalty;
                    }
                }
            }
        }

        // Greedy decoding if temperature is near zero
        if self.config.temperature < 1e-4 {
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() {
                if l > best_val {
                    best_val = l;
                    best_idx = i;
                }
            }
            return best_idx as u32;
        }

        // 2. Temperature scaling
        for l in logits.iter_mut() {
            *l /= self.config.temperature;
        }

        // 3. Softmax with numerical stability
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, |a, b| {
            if b.is_finite() {
                a.max(b)
            } else {
                a
            }
        });
        let max_l = if max_l.is_finite() { max_l } else { 0.0 };
        let mut sum_exp = 0.0f32;
        for l in logits.iter_mut() {
            if !l.is_finite() {
                *l = 0.0;
            } else {
                *l = (*l - max_l).exp();
            }
            sum_exp += *l;
        }
        if sum_exp > 1e-12 {
            for l in logits.iter_mut() {
                *l /= sum_exp;
            }
        } else {
            let uniform = 1.0 / logits.len() as f32;
            for l in logits.iter_mut() {
                *l = uniform;
            }
        }

        // 4. Pair logits with token indices
        let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 5. Top-K cutoff
        if self.config.top_k > 0 && self.config.top_k < indexed.len() {
            indexed.truncate(self.config.top_k);
        }

        // 6. Top-P (nucleus) cutoff
        if self.config.top_p < 1.0 {
            let mut cum_prob = 0.0f32;
            let mut cutoff = indexed.len();
            for (i, &(_, p)) in indexed.iter().enumerate() {
                cum_prob += p;
                if cum_prob >= self.config.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
        }

        // 7. Renormalize and sample
        let total_prob: f32 = indexed.iter().map(|(_, p)| p).sum();
        let mut rng = rand::thread_rng();
        let mut r = rng.gen::<f32>() * total_prob;

        for (idx, p) in indexed {
            r -= p;
            if r <= 0.0 {
                return idx as u32;
            }
        }

        0
    }
}
