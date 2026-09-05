//! Model evaluation metrics: Cross-Entropy Loss, Perplexity, KL-Divergence, and Top-K Agreement.

/// Compute numerically stable log-sum-exp of a logit vector
pub fn log_sum_exp(logits: &[f32]) -> f64 {
    if logits.is_empty() {
        return 0.0;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_exp: f64 = logits.iter().map(|&l| ((l as f64) - max_logit).exp()).sum();
    max_logit + sum_exp.ln()
}

/// Compute cross-entropy negative log likelihood (NLL) for a target token
pub fn compute_nll(logits: &[f32], target_token: u32) -> f64 {
    let idx = target_token as usize;
    if idx >= logits.len() {
        return 15.0; // High penalty for out of vocab
    }
    let lse = log_sum_exp(logits);
    let target_logit = logits[idx] as f64;
    (lse - target_logit).max(0.0)
}

/// Compute perplexity from a sequence of negative log likelihoods: PPL = exp(mean(NLL))
pub fn compute_perplexity(nll_losses: &[f64]) -> f64 {
    if nll_losses.is_empty() {
        return 1.0;
    }
    let mean_nll = nll_losses.iter().sum::<f64>() / (nll_losses.len() as f64);
    mean_nll.exp()
}

/// Compute Kullback-Leibler (KL) divergence D_KL(P || Q) between baseline and quantized logits
pub fn compute_kl_divergence(base_logits: &[f32], test_logits: &[f32]) -> f64 {
    let len = base_logits.len().min(test_logits.len());
    if len == 0 {
        return 0.0;
    }

    let lse_base = log_sum_exp(&base_logits[..len]);
    let lse_test = log_sum_exp(&test_logits[..len]);

    let mut kl = 0.0f64;
    for i in 0..len {
        let log_p = (base_logits[i] as f64) - lse_base;
        let log_q = (test_logits[i] as f64) - lse_test;
        let p = log_p.exp();
        if p > 1e-12 {
            kl += p * (log_p - log_q);
        }
    }
    kl.max(0.0)
}

/// Check if top-1 predicted token matches between baseline and quantized logits
pub fn compute_top1_agreement(base_logits: &[f32], test_logits: &[f32]) -> bool {
    let len = base_logits.len().min(test_logits.len());
    if len == 0 {
        return true;
    }

    let mut best_base_idx = 0;
    let mut best_base_val = f32::NEG_INFINITY;
    for (i, &v) in base_logits[..len].iter().enumerate() {
        if v > best_base_val {
            best_base_val = v;
            best_base_idx = i;
        }
    }

    let mut best_test_idx = 0;
    let mut best_test_val = f32::NEG_INFINITY;
    for (i, &v) in test_logits[..len].iter().enumerate() {
        if v > best_test_val {
            best_test_val = v;
            best_test_idx = i;
        }
    }

    best_base_idx == best_test_idx
}

/// Check if baseline top-1 predicted token is in the top-k predictions of the quantized logits
pub fn compute_topk_agreement(base_logits: &[f32], test_logits: &[f32], k: usize) -> bool {
    let len = base_logits.len().min(test_logits.len());
    if len == 0 {
        return true;
    }

    let mut best_base_idx = 0;
    let mut best_base_val = f32::NEG_INFINITY;
    for (i, &v) in base_logits[..len].iter().enumerate() {
        if v > best_base_val {
            best_base_val = v;
            best_base_idx = i;
        }
    }

    let mut indices: Vec<usize> = (0..len).collect();
    indices.sort_by(|&a, &b| {
        test_logits[b]
            .partial_cmp(&test_logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    indices.iter().take(k).any(|&idx| idx == best_base_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perplexity_and_kl() {
        let logits1 = vec![2.0, 1.0, 0.5, -1.0];
        let logits2 = vec![2.05, 0.95, 0.48, -0.98];

        let nll = compute_nll(&logits1, 0);
        assert!(nll > 0.0);
        let ppl = compute_perplexity(&[nll]);
        assert!(ppl >= 1.0);

        let kl = compute_kl_divergence(&logits1, &logits2);
        assert!(kl < 0.05, "KL divergence should be very small: {}", kl);

        assert!(compute_top1_agreement(&logits1, &logits2));
        assert!(compute_topk_agreement(&logits1, &logits2, 3));
    }
}
