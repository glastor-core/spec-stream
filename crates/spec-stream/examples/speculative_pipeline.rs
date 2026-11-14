use spec_stream::prelude::*;
use std::sync::Arc;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("       SPEC-STREAM: High-Performance Speculative Decoding   ");
    println!("============================================================\n");

    let vocab_size = 1000;
    println!("[1/4] Initializing simulated inference models...");
    // Draft model: Fast, small model (e.g. Qwen-2.5-0.5B / LLaMA-3.2-1B)
    let draft_model = SyntheticModel::new("draft-small-0.5B", vocab_size, 0.75, 42);

    // Target model: Large, authoritative model (e.g. Qwen-2.5-7B / LLaMA-3-8B)
    let target_model = SyntheticModel::new("target-large-7B", vocab_size, 0.98, 1337);

    println!("[2/4] Setting up Paged KV-Cache Manager (128 physical blocks)...");
    let cache_manager = Arc::new(PagedKvCacheManager::with_defaults(128));

    println!("[3/4] Configuring Speculative Pipeline (K = 4 draft tokens / burst)...");
    let config = SpeculativeConfig {
        k_draft_tokens: 4,
        max_new_tokens: 24,
        eos_token_id: Some(999),
        strategy: VerificationStrategy::Greedy,
        ..Default::default()
    };

    let pipeline = SpeculativePipeline::new(draft_model, target_model, cache_manager, config);

    let prompt = vec![10, 20, 30]; // Simulated prompt token sequence
    let sequence_id = 1;

    println!("[4/4] Starting Speculative Generation Stream...\n");
    let mut stream = pipeline.generate_stream(prompt, sequence_id)?;

    let mut total_tokens = 0;

    while let Some(event) = stream.next().await {
        match event {
            GenerationEvent::Token {
                token_id,
                is_draft,
            } => {
                total_tokens += 1;
                let tag = if is_draft {
                    "[DRAFT ACCEPTED]"
                } else {
                    "[TARGET COMMIT] "
                };
                println!("  Token #{total_tokens:02}: id={token_id:<5} {tag}");
            }
            GenerationEvent::Burst {
                accepted,
                total_produced,
            } => {
                println!(
                    "  --> Speculative Burst: {accepted} draft tokens accepted, {total_produced} total committed in 1 target pass\n"
                );
            }
            GenerationEvent::AdaptiveKChanged { old_k, new_k, acceptance_rate } => {
                println!(
                    "  [ADAPTIVE-K] Adjusted draft window from K={old_k} to K={new_k} (EMA acceptance: {:.1}%)\n",
                    acceptance_rate * 100.0
                );
            }
            GenerationEvent::Metrics(metrics) => {
                println!("  [Telemetry] Current Speedup: {:.2}x | Acceptance Rate: {:.1}%",
                    metrics.effective_speedup(),
                    metrics.acceptance_rate() * 100.0
                );
            }
            GenerationEvent::Finished {
                total_tokens,
                reason,
            } => {
                println!("\n============================================================");
                println!("Generation Complete!");
                println!("  Total tokens emitted : {total_tokens}");
                println!("  Finish reason        : {reason}");
                println!("============================================================");
            }
        }
    }

    Ok(())
}

