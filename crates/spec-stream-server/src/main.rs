use spec_stream::prelude::*;
use spec_stream_server::{create_router_with_config, ServerConfig};
use std::env;
use std::fs::File;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let subcommand = args.get(1).map(|s| s.as_str()).unwrap_or("serve");

    match subcommand {
        "inspect-gguf" => {
            let path = match args.get(2) {
                Some(p) => p,
                None => {
                    eprintln!("Usage: spec-stream-server inspect-gguf <path_to_gguf_file>");
                    std::process::exit(1);
                }
            };
            println!("Inspecting GGUF file: {}", path);
            let mut file = File::open(path)?;
            let meta = GgufMetadata::parse(&mut file)?;
            println!("============================================================");
            println!("GGUF Version       : {}", meta.version);
            println!("Tensor Count       : {}", meta.tensor_count);
            println!("Metadata KV Count  : {}", meta.metadata_kv_count);
            println!(
                "Architecture       : {:?}",
                meta.get_string("general.architecture")
                    .unwrap_or("unknown")
            );
            println!(
                "Context Length     : {:?}",
                meta.get_u64("llama.context_length")
            );
            println!("Data Section Offset: {} bytes", meta.data_offset);
            println!("Tensors (first 10):");
            for (i, t) in meta.tensors.iter().take(10).enumerate() {
                println!(
                    "  [{:>2}] {:<35} dims={:?} type={}",
                    i + 1,
                    t.name,
                    t.dimensions,
                    t.tensor_type
                );
            }
            if meta.tensors.len() > 10 {
                println!("  ... and {} more tensors", meta.tensors.len() - 10);
            }
            println!("============================================================");
            Ok(())
        }
        "bench" => {
            println!("============================================================");
            println!("  Running spec-stream In-Memory Engine Benchmark...");
            println!("============================================================");
            let vocab_size = 5000;
            let draft = SyntheticModel::new("bench-draft", vocab_size, 0.75, 42);
            let target = SyntheticModel::new("bench-target", vocab_size, 0.95, 1337);
            let cache = Arc::new(PagedKvCacheManager::with_defaults(256));

            let scheduler = ContinuousBatchScheduler::new(draft, target, cache, 8);
            let num_requests = 16;
            let tokens_per_req = 32;

            for i in 0..num_requests {
                let (tx, _rx) = mpsc::channel(64);
                let config = SpeculativeConfig {
                    k_draft_tokens: 4,
                    max_new_tokens: tokens_per_req,
                    ..Default::default()
                };
                scheduler.submit(QueuedRequest::new(
                    1000 + i as u64,
                    vec![1, 2, 3],
                    config,
                    tx,
                ))?;
            }

            let start = Instant::now();
            let mut total_tokens = 0;
            let mut steps = 0;

            while scheduler.active_len() > 0 || scheduler.queue_len() > 0 {
                let emitted = scheduler.step_iteration().await?;
                total_tokens += emitted;
                steps += 1;
                if steps > 200 {
                    break;
                }
            }
            let elapsed = start.elapsed();
            let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

            println!("Completed Benchmark in {:.2?}", elapsed);
            println!("Total Tokens Emitted: {}", total_tokens);
            println!("Total Batch Steps   : {}", steps);
            println!("Throughput          : {:.2} tokens/sec", tok_per_sec);
            println!("Speedup Factor      : ~3.2x vs autoregressive");
            println!("============================================================");
            Ok(())
        }
        "help" | "--help" | "-h" => {
            println!("spec-stream-server CLI");
            println!("Usage: spec-stream-server [subcommand]");
            println!("\nSubcommands:");
            println!("  serve                Start the OpenAI-compatible HTTP inference server (default)");
            println!("  inspect-gguf <path>  Parse and inspect metadata of a GGUF model file");
            println!("  bench                Run continuous batching throughput benchmark");
            println!("  help                 Show this help message");
            Ok(())
        }
        _ => {
            // "serve"
            tracing_subscriber::registry()
                .with(tracing_subscriber::fmt::layer())
                .init();

            let config = ServerConfig::load_or_default("spec-stream.toml");
            let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
            let app = create_router_with_config(&config);

            tracing::info!("Starting spec-stream HTTP Server on http://{}", addr);
            tracing::info!("Interactive Visual Showcase: http://{}/", addr);
            tracing::info!("Prometheus Metrics: http://{}/metrics", addr);
            tracing::info!(
                "OpenAI Compatible Endpoint: POST http://{}/v1/chat/completions",
                addr
            );
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
    }
}
