use std::sync::atomic::{AtomicU64, Ordering};

pub struct ServerMetrics {
    pub requests_total: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    pub total_latency_ms: AtomicU64,
    pub model_load_time_ms: AtomicU64,
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            tokens_generated_total: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            model_load_time_ms: AtomicU64::new(0),
        }
    }
}

impl ServerMetrics {
    pub fn record_request(&self, tokens: u64, latency_ms: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.tokens_generated_total
            .fetch_add(tokens, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(latency_ms, Ordering::Relaxed);
    }

    pub fn to_prometheus_text(&self) -> String {
        let reqs = self.requests_total.load(Ordering::Relaxed);
        let tokens = self.tokens_generated_total.load(Ordering::Relaxed);
        let latency = self.total_latency_ms.load(Ordering::Relaxed);
        let load_time = self.model_load_time_ms.load(Ordering::Relaxed);

        let tok_per_sec = if latency > 0 {
            (tokens as f64) / (latency as f64 / 1000.0)
        } else {
            0.0
        };

        let avg_lat = if reqs > 0 {
            (latency as f64) / (reqs as f64)
        } else {
            0.0
        };

        format!(
            "# HELP requests_total Total number of inference requests processed\n\
             # TYPE requests_total counter\n\
             requests_total {}\n\n\
             # HELP tokens_generated_total Total number of tokens generated\n\
             # TYPE tokens_generated_total counter\n\
             tokens_generated_total {}\n\n\
             # HELP tokens_per_second Mean generation throughput\n\
             # TYPE tokens_per_second gauge\n\
             tokens_per_second {:.2}\n\n\
             # HELP inference_latency_seconds Average request inference latency in seconds\n\
             # TYPE inference_latency_seconds gauge\n\
             inference_latency_seconds {:.4}\n\n\
             # HELP model_load_time_seconds Time taken to load and verify .pox artifact\n\
             # TYPE model_load_time_seconds gauge\n\
             model_load_time_seconds {:.4}\n",
            reqs,
            tokens,
            tok_per_sec,
            avg_lat / 1000.0,
            (load_time as f64) / 1000.0
        )
    }
}
