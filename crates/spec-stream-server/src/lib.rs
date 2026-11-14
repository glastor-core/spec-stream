//! # spec-stream-server
//!
//! OpenAI-compatible HTTP inference server for `spec-stream`.

pub mod config;
pub mod handlers;
pub mod types;

pub use config::ServerConfig;

use axum::{
    routing::{get, post},
    Router,
};
use handlers::{chat_completions_handler, health_handler, index_handler, metrics_handler, AppState};
use spec_stream::prelude::*;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// Creates an HTTP router using default server configuration.
pub fn create_router() -> Router {
    create_router_with_config(&ServerConfig::default())
}

/// Creates an HTTP router using the specified server configuration.
pub fn create_router_with_config(config: &ServerConfig) -> Router {
    let cache_manager = Arc::new(PagedKvCacheManager::new(
        config.cache_total_blocks,
        config.cache_block_size,
    ));
    let state = AppState {
        cache_manager,
        next_seq_id: Arc::new(AtomicU64::new(1)),
        total_requests: Arc::new(AtomicU64::new(0)),
        total_tokens_generated: Arc::new(AtomicU64::new(0)),
    };

    Router::new()
        .route("/", get(index_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // for `oneshot`

    #[tokio::test]
    async fn test_index_endpoint() {
        let app = create_router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(body_str.contains("spec-stream"));
        assert!(body_str.contains("Speculative Decoding"));
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let app = create_router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(body_str.contains("spec_stream_free_kv_blocks"));
        assert!(body_str.contains("spec_stream_requests_total"));
    }
}
