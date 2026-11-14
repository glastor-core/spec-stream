use crate::types::*;
use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use spec_stream::prelude::*;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;

#[derive(Clone)]
pub struct AppState {
    pub cache_manager: Arc<PagedKvCacheManager>,
    pub next_seq_id: Arc<AtomicU64>,
    pub total_requests: Arc<AtomicU64>,
    pub total_tokens_generated: Arc<AtomicU64>,
}

pub async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy",
        version: "0.1.0",
        free_kv_blocks: state.cache_manager.free_blocks(),
        allocated_kv_blocks: state.cache_manager.allocated_blocks(),
    })
}

pub async fn index_handler() -> impl IntoResponse {
    axum::response::Html(include_str!("../../../index.html"))
}

pub async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let free_blocks = state.cache_manager.free_blocks();
    let alloc_blocks = state.cache_manager.allocated_blocks();
    let prefix_entries = state.cache_manager.prefix_entries();
    let requests = state.total_requests.load(Ordering::Relaxed);
    let tokens = state.total_tokens_generated.load(Ordering::Relaxed);

    let body = format!(
        "# HELP spec_stream_free_kv_blocks Number of free physical KV-cache blocks.\n\
         # TYPE spec_stream_free_kv_blocks gauge\n\
         spec_stream_free_kv_blocks {free_blocks}\n\n\
         # HELP spec_stream_allocated_kv_blocks Number of allocated physical KV-cache blocks.\n\
         # TYPE spec_stream_allocated_kv_blocks gauge\n\
         spec_stream_allocated_kv_blocks {alloc_blocks}\n\n\
         # HELP spec_stream_prefix_cache_entries Number of active prefix cache entries.\n\
         # TYPE spec_stream_prefix_cache_entries gauge\n\
         spec_stream_prefix_cache_entries {prefix_entries}\n\n\
         # HELP spec_stream_requests_total Total number of chat completion requests processed.\n\
         # TYPE spec_stream_requests_total counter\n\
         spec_stream_requests_total {requests}\n\n\
         # HELP spec_stream_tokens_generated_total Total number of tokens generated.\n\
         # TYPE spec_stream_tokens_generated_total counter\n\
         spec_stream_tokens_generated_total {tokens}\n"
    );

    (
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

pub async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    state.total_requests.fetch_add(1, Ordering::Relaxed);
    let seq_id = state.next_seq_id.fetch_add(1, Ordering::Relaxed);
    let is_streaming = payload.stream.unwrap_or(false);
    let max_tokens = payload.max_tokens.unwrap_or(32);
    let temperature = payload.temperature.unwrap_or(0.0);

    // Mock models for demonstration (can be switched to Candle models)
    let vocab_size = 5000;
    let draft_model = SyntheticModel::new("draft-small", vocab_size, 0.75, 42);
    let target_model = SyntheticModel::new("target-large", vocab_size, 0.98, 1337);

    let strategy = if temperature > 1e-4 {
        VerificationStrategy::Stochastic { temperature }
    } else {
        VerificationStrategy::Greedy
    };

    let config = SpeculativeConfig {
        k_draft_tokens: 4,
        max_new_tokens: max_tokens,
        eos_token_id: Some(9999),
        strategy,
        ..Default::default()
    };

    let pipeline = SpeculativePipeline::new(
        draft_model,
        target_model,
        state.cache_manager.clone(),
        config,
    );

    let tokenizer = ByteLevelTokenizer::new();

    // Encode messages into real prompt tokens
    let combined_prompt = payload
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let mut prompt_tokens = tokenizer.encode(&combined_prompt);
    if prompt_tokens.is_empty() {
        prompt_tokens.push(tokenizer.bos_token_id);
    }

    let prompt_len = prompt_tokens.len().max(1);
    let stream_res = pipeline.generate_stream(prompt_tokens, seq_id);

    let mut stream = match stream_res {
        Ok(s) => s,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Pipeline initialization failed: {e}"),
            )
                .into_response();
        }
    };

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let request_id = format!("chatcmpl-{seq_id}");
    let model_name = payload.model.clone();
    let state_tokens = state.total_tokens_generated.clone();

    if is_streaming {
        let tokenizer_stream = tokenizer.clone();
        let sse_stream = async_stream::stream! {
            while let Some(event) = stream.next().await {
                if let GenerationEvent::Token { token_id, .. } = event {
                    state_tokens.fetch_add(1, Ordering::Relaxed);
                    let piece = tokenizer_stream.decode(&[token_id]);
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_name.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: ChoiceDelta {
                                content: Some(piece),
                                role: None,
                            },
                            finish_reason: None,
                        }],
                    };

                    if let Ok(json_str) = serde_json::to_string(&chunk) {
                        yield Ok::<_, Infallible>(Event::default().data(json_str));
                    }
                }
            }

            // Final [DONE] chunk
            yield Ok::<_, Infallible>(Event::default().data("[DONE]"));
        };

        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let mut tokens_out = Vec::new();
        while let Some(event) = stream.next().await {
            if let GenerationEvent::Token { token_id, .. } = event {
                tokens_out.push(token_id);
            }
        }

        let total_gen = tokens_out.len();
        state.total_tokens_generated.fetch_add(total_gen as u64, Ordering::Relaxed);
        let content = tokenizer.decode(&tokens_out);

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion",
            created,
            model: payload.model,
            choices: vec![ResponseChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content,
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: total_gen,
                total_tokens: prompt_len + total_gen,
            },
        };

        Json(response).into_response()
    }
}

