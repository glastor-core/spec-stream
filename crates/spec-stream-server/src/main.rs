use spec_stream_server::{create_router_with_config, ServerConfig};
use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = ServerConfig::load_or_default("spec-stream.toml");
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let app = create_router_with_config(&config);

    tracing::info!("Starting spec-stream HTTP Server on http://{}", addr);
    tracing::info!("Interactive Visual Showcase: http://{}/", addr);
    tracing::info!("Prometheus Metrics: http://{}/metrics", addr);
    tracing::info!("OpenAI Compatible Endpoint: POST http://{}/v1/chat/completions", addr);
    tracing::info!("Health Check: GET http://{}/health", addr);
    tracing::info!(
        "Cache Pool: {} blocks (block size: {} tokens) | Speculative K: {} (adaptive: {})",
        config.cache_total_blocks,
        config.cache_block_size,
        config.k_draft_tokens,
        config.adaptive_k
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

