//! OpenAI-compatible HTTP server with streaming SSE and Prometheus metrics.

pub mod metrics;
pub mod server;

pub use metrics::ServerMetrics;
pub use server::{create_router, AppState, ChatCompletionRequest, ChatCompletionResponse};

use paraoxidizer_core::error::{PoxError, Result};
use paraoxidizer_runtime::engine::PoxEngine;
use std::{net::SocketAddr, sync::Arc};

pub async fn run_server(engine: PoxEngine, host: &str, port: u16, model_id: String) -> Result<()> {
    let metrics = Arc::new(ServerMetrics::default());
    let state = AppState {
        engine: Arc::new(engine),
        metrics,
        model_id,
    };

    let router = create_router(state);
    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .map_err(|e| PoxError::Runtime(format!("Invalid bind address: {e}")))?;

    println!("ParaOxidizer inference server listening on http://{}", addr);
    println!("  - OpenAI API: http://{}/v1/chat/completions", addr);
    println!("  - Metrics:    http://{}/metrics", addr);
    println!("  - Health:     http://{}/health", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
