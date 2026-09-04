mod common;

use common::create_hf_llama_model;
use paraoxidizer::cli::commands;
use paraoxidizer::format::PoxFile;
use paraoxidizer::runtime::engine::PoxEngine;
use paraoxidizer_serve::create_router;
use paraoxidizer_serve::server::AppState;
use paraoxidizer_serve::ServerMetrics;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[tokio::test]
async fn test_openai_server_endpoints() {
    let tmp = tempdir().unwrap();
    let model_dir = tmp.path().join("llama_serve");
    create_hf_llama_model(&model_dir);

    let pox_path = tmp.path().join("serve_model.pox");
    commands::run_quantize(
        model_dir.to_str().unwrap(),
        4,
        128,
        "automatic",
        "min-max",
        pox_path.to_str().unwrap(),
    )
    .unwrap();

    let pox = PoxFile::open(&pox_path).unwrap();
    let engine = PoxEngine::new(pox);
    let metrics = Arc::new(ServerMetrics::default());

    let state = AppState {
        engine: Arc::new(engine),
        metrics,
        model_id: "llama-3-8b-pox".to_string(),
    };

    let router = create_router(state);

    // Bind on random local port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // 1. Test GET /health
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();

    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    assert!(response.contains("200 OK"));
    assert!(response.contains("OK"));

    // 2. Test GET /v1/models
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();

    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    assert!(response.contains("200 OK"));
    assert!(response.contains("llama-3-8b-pox"));

    // 3. Test POST /v1/chat/completions (Non-streaming)
    let body = serde_json::json!({
        "model": "llama-3-8b-pox",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 16,
        "temperature": 0.5,
        "stream": false
    })
    .to_string();

    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(req.as_bytes()).await.unwrap();

    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    assert!(response.contains("200 OK"));
    assert!(response.contains("chatcmpl-pox"));
    assert!(response.contains("assistant"));

    // 4. Test GET /metrics (Prometheus)
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();

    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    assert!(response.contains("200 OK"));
    assert!(response.contains("requests_total 1"));
    assert!(response.contains("tokens_generated_total"));
}
