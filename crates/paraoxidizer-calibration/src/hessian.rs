use serde::{Deserialize, Serialize};

/// Represents an empirical Hessian matrix H = 2/N * X * X^T and its inverse H^-1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HessianMatrix {
    pub dim: usize,
    pub data: Vec<f32>, // Row-major dim x dim
    pub inv_data: Vec<f32>,
    pub activation_scales: Vec<f32>,
}

impl HessianMatrix {
    /// Create a new empty Hessian accumulator for dimension `dim`
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: vec![0.0f32; dim * dim],
            inv_data: Vec::new(),
            activation_scales: vec![0.0f32; dim],
        }
    }

    /// Accumulate activation batch X where X has shape [dim, num_samples]
    pub fn accumulate_activations(&mut self, x: &[f32], num_samples: usize) {
        if num_samples == 0 || x.len() < self.dim * num_samples {
            return;
        }

        // 1. Accumulate per-channel activation scales: s_j = sum(|x_j|)
        for j in 0..self.dim {
            let row_start = j * num_samples;
            let mut sum_abs = 0.0f32;
            for s in 0..num_samples {
                sum_abs += x[row_start + s].abs();
            }
            self.activation_scales[j] += sum_abs / (num_samples as f32);
        }

        // 2. Accumulate outer product: H_jk += 2/N * sum_s (x_{j,s} * x_{k,s})
        let norm_factor = 2.0 / (num_samples as f32);
        for j in 0..self.dim {
            let j_start = j * num_samples;
            let row_offset = j * self.dim;
            for k in j..self.dim {
                let k_start = k * num_samples;
                let mut dot = 0.0f32;
                for s in 0..num_samples {
                    dot += x[j_start + s] * x[k_start + s];
                }
                let val = dot * norm_factor;
                self.data[row_offset + k] += val;
                if j != k {
                    self.data[k * self.dim + j] += val; // Symmetric
                }
            }
        }
    }

    /// Compute damped inverse Hessian H^-1
    pub fn compute_inverse(&mut self, damping: f32) {
        let n = self.dim;
        let mut h = self.data.clone();

        // Compute mean diagonal for ridge damping: lambda = damping * mean(diag(H))
        let mut trace = 0.0f32;
        for i in 0..n {
            trace += h[i * n + i];
        }
        let lambda = (damping * (trace / n as f32)).max(1e-4);

        // Add damping lambda to diagonal
        for i in 0..n {
            h[i * n + i] += lambda;
        }

        // Invert using Gauss-Jordan elimination with partial pivoting
        let mut inv = vec![0.0f32; n * n];
        for i in 0..n {
            inv[i * n + i] = 1.0;
        }

        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = h[col * n + col].abs();
            for r in (col + 1)..n {
                let val = h[r * n + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = r;
                }
            }

            if max_row != col {
                for c in 0..n {
                    h.swap(col * n + c, max_row * n + c);
                    inv.swap(col * n + c, max_row * n + c);
                }
            }

            let pivot = h[col * n + col];
            let pivot_inv = if pivot.abs() > 1e-12 { 1.0 / pivot } else { 1.0 };

            for c in 0..n {
                h[col * n + c] *= pivot_inv;
                inv[col * n + c] *= pivot_inv;
            }

            for r in 0..n {
                if r != col {
                    let factor = h[r * n + col];
                    for c in 0..n {
                        h[r * n + c] -= factor * h[col * n + c];
                        inv[r * n + c] -= factor * inv[col * n + c];
                    }
                }
            }
        }

        self.inv_data = inv;
    }

    pub fn get_inv(&self, r: usize, c: usize) -> f32 {
        if self.inv_data.is_empty() {
            if r == c { 1.0 } else { 0.0 }
        } else {
            self.inv_data[r * self.dim + c]
        }
    }
}
