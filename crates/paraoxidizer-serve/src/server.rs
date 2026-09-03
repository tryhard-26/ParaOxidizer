use crate::metrics::ServerMetrics;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use paraoxidizer_runtime::{
    engine::PoxEngine,
    sampler::SamplerConfig,
};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant};
use tower_http::cors::CorsLayer;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<PoxEngine>,
    pub metrics: Arc<ServerMetrics>,
    pub model_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_max_tokens() -> usize {
    128
}
fn default_top_p() -> f32 {
    0.9
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessageOutput,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageOutput {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/completions", post(completions_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_handler() -> &'static str {
    "OK"
}

async fn metrics_handler(State(state): State<AppState>) -> String {
    state.metrics.to_prometheus_text()
}

async fn models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "paraoxidizer",
        }]
    }))
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let start_time = Instant::now();

    // Construct unified prompt from conversation
    let mut prompt = String::new();
    for msg in &req.messages {
        prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
    }
    prompt.push_str("<|assistant|>\n");

    let sampler_config = SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: 40,
        repetition_penalty: 1.1,
        stop_sequences: vec!["<|user|>".into(), "<|assistant|>".into()],
    };

    if req.stream {
        // SSE streaming response
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);
        let engine = state.engine.clone();
        let max_toks = req.max_tokens;
        let model_id = state.model_id.clone();
        let metrics = state.metrics.clone();

        tokio::task::spawn_blocking(move || {
            let mut tok_count = 0u64;
            let _ = engine.generate_stream(&prompt, max_toks, sampler_config, |piece| {
                tok_count += 1;
                let chunk_json = serde_json::json!({
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": piece },
                        "finish_reason": null
                    }]
                });
                let _ = tx.blocking_send(chunk_json.to_string());
                true
            });

            // Final done event
            let _ = tx.blocking_send("[DONE]".to_string());
            metrics.record_request(tok_count, start_time.elapsed().as_millis() as u64);
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let sse_stream = stream.map(|msg| {
            if msg == "[DONE]" {
                Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
            } else {
                Ok::<Event, std::convert::Infallible>(Event::default().data(msg))
            }
        });

        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        // Non-streaming response
        let mut full_output = String::new();
        let mut tok_count = 0u64;
        let _ = state
            .engine
            .generate_stream(&prompt, req.max_tokens, sampler_config, |piece| {
                tok_count += 1;
                full_output.push_str(piece);
                true
            });

        state
            .metrics
            .record_request(tok_count, start_time.elapsed().as_millis() as u64);

        let resp = ChatCompletionResponse {
            id: "chatcmpl-pox".into(),
            object: "chat.completion".into(),
            created: 1700000000,
            model: state.model_id.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessageOutput {
                    role: "assistant".into(),
                    content: full_output,
                },
                finish_reason: "stop".into(),
            }],
        };

        (StatusCode::OK, Json(resp)).into_response()
    }
}

async fn completions_handler(
    State(state): State<AppState>,
    Json(req): Json<serde_json::Value>,
) -> Response {
    let prompt = req["prompt"].as_str().unwrap_or("").to_string();
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(64) as usize;

    let sampler_config = SamplerConfig::default();
    let mut full_output = String::new();
    let _ = state
        .engine
        .generate_stream(&prompt, max_tokens, sampler_config, |piece| {
            full_output.push_str(piece);
            true
        });

    Json(serde_json::json!({
        "id": "cmpl-pox",
        "object": "text_completion",
        "model": state.model_id,
        "choices": [{
            "text": full_output,
            "index": 0,
            "finish_reason": "length"
        }]
    }))
    .into_response()
}

use tokio_stream::StreamExt;
