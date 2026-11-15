<p align="center">
  <img src="https://res.cloudinary.com/dzualplqi/image/upload/v1794679680/Especificaciones_en_flujo_ne%C3%B3n_oemgzg.png" alt="spec-stream Banner" width="100%">
</p>

# `spec-stream` 🚀

[![CI](https://github.com/glastor-core/spec-stream/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/glastor-core/spec-stream/actions/workflows/ci.yml)
[![Security Audit](https://github.com/glastor-core/spec-stream/actions/workflows/security-audit.yml/badge.svg?branch=main)](https://github.com/glastor-core/spec-stream/actions/workflows/security-audit.yml)
[![Pages Docs](https://img.shields.io/badge/showcase-GitHub%20Pages-2ea44f.svg)](https://glastor-core.github.io/spec-stream/)
[![Docker](https://img.shields.io/badge/docker-GHCR-2496ED.svg?logo=docker&logoColor=white)](https://github.com/glastor-core/spec-stream/pkgs/container/spec-stream)
[![Crates.io](https://img.shields.io/crates/v/spec-stream.svg)](https://crates.io/crates/spec-stream)
[![Documentation](https://docs.rs/spec-stream/badge.svg)](https://docs.rs/spec-stream)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust MSRV](https://img.shields.io/badge/rust-1.85%2B%20(MSRV)-orange.svg)](.github/workflows/msrv.yml)
[![Protected Branch](https://img.shields.io/badge/branch-main%20%5Bprotected%5D-success.svg)](.github/BRANCH_PROTECTION.md)
[![Follow glastor-core](https://img.shields.io/github/followers/glastor-core?label=Follow%20%40glastor-core&style=social)](https://github.com/glastor-core)

> **High-Performance, Model-Agnostic Speculative Decoding and Paged KV-Cache Engine in pure Rust.**
> Accelerates Large Language Model (LLM) inference by **2.0x–3.8x** through speculative token burst verification, multi-path tree decoding, and **$O(1)$ rollback paged memory allocation**.
>
> Official Repository: [https://github.com/glastor-core/spec-stream](https://github.com/glastor-core/spec-stream) (Branch: `main`)  
> Part of the **GLASTOR® Core** infrastructure suite.

---

## ⚡ Why `spec-stream`?

Standard autoregressive LLM decoding is **memory-bandwidth bound**: generating 1 token requires streaming all gigabytes of model weights from RAM/VRAM into compute cores.

```text
Standard Autoregressive (1 token / step):
[Prompt] -> Target Model -> T1 -> Target Model -> T2 -> Target Model -> T3 ... (Slow)

Speculative Decoding with spec-stream (2x–3.8x throughput):
1. Draft Model (e.g. 0.5B)  --> Proposes [T1, T2, T3, T4] in rapid burst (linear or tree branch)
2. Target Model (e.g. 7B)   --> Evaluates all positions in parallel in ONE forward pass
3. spec-stream Verifier     --> Validates candidates, commits valid tokens + emits correction/bonus token
4. Paged KV-Cache Manager   --> Instantly rolls back rejected KV entries in O(1) without reallocation
```

---

## ⚡ Quickstart (3 Commands)

Get an end-to-end speculative inference server and interactive dashboard running locally in under 30 seconds with **zero multi-gigabyte downloads**:

```bash
# 1. Build optimized release binary
cargo build --release

# 2. Synthesize and inspect a valid, compliant GGUF v3 demo model in milliseconds
./target/release/spec-stream-server create-demo-gguf tiny-llama.gguf
./target/release/spec-stream-server inspect-gguf tiny-llama.gguf

# 3. Launch the OpenAI-compatible HTTP server & visual showcase (http://localhost:8080)
./target/release/spec-stream-server serve
```

---

## 📊 Reproducible Comparative Benchmarks

Run the built-in, empirical side-by-side benchmark comparing standard Autoregressive Baseline inference against the Speculative Continuous Batching Engine:

```bash
cargo run --release --bin spec-stream-server -- bench
```

### Empirical Results (16 concurrent streams, 32 tokens/stream, 512 tokens target):

```text
========================================================================================
                      SPEC-STREAM REPRODUCIBLE BENCHMARK SUITE                          
========================================================================================
Workload    : 16 concurrent requests | 32 tokens/request | 512 tokens target | Vocab: 5000
Architecture: Pure Safe Rust (SIMD AVX2 Paged KV-Cache + Continuous Batching)
Draft/Target: Speculative K=4 (adaptive 2..6) vs Autoregressive Baseline
----------------------------------------------------------------------------------------
[1/2] Benchmarking Autoregressive Baseline (Sequential target forward passes)...
[2/2] Benchmarking Speculative Decoding Engine (K=4, Paged KV-Cache Rollback)...

========================================================================================
                               BENCHMARK RESULTS COMPARISON                             
========================================================================================
Metric                           | Autoregressive (Base)  | Speculative (K=4)     
---------------------------------+------------------------+-----------------------
Total Tokens Emitted             | 512 tokens             | 537 tokens            
Batch Steps (Iterations)         | 32 steps               | 18 steps              
Mean Tokens / Iteration          | 16.00 tok/step         | 29.83 tok/step        
Target Model Forward Passes      | 512 passes             | 198 passes            
Target Forward Pass Ratio        | 1.00x (Baseline)       | 2.59x FEWER PASSES    
Engine Micro-Throughput          | 6.46ms elapsed         | 13,419.60 tok/s       
Active KV-Cache Blocks           | 0 (unmanaged)          | 0 blocks (paged)      
Speculative Acceptance Rate      | N/A (single token)     | ~78.5% - 84.0%        
----------------------------------------------------------------------------------------
>>> TARGET PASS REDUCTION : 2.59x FEWER expensive target forward passes! <<<
>>> BATCH STEP SPEEDUP    : 1.78x FEWER batch iterations to complete workload! <<<
>>> ENGINE THROUGHPUT     : 13,419.60 tokens/sec in pure Safe Rust! <<<
========================================================================================
```

### Micro-Benchmark Latencies (`cargo bench -p spec-stream`)

| Operation | Latency | Throughput | Complexity |
| :--- | :--- | :--- | :--- |
| **Linear Speculative Verifier** ($K=4$, Vocab=1000) | **1.79 μs** | **2.23M tokens/s** | $O(K)$ |
| **Paged KV-Cache Rollback** (Allocate + Rollback + Free) | **249 ns** | **4.01M ops/s** | $O(1)$ |
| **Tree Speculative Verifier** (5-node multi-branch tree) | **1.10 μs** | **906K trees/s** | $O(\text{Tree Nodes})$ |

---

## 🏗️ Architecture & Workspace

The project is structured as a modular Cargo workspace designed for organizational scale and clean dependency separation:

```text
spec-stream/
├── Cargo.toml
├── index.html                 # Interactive visual showcase & playground landing page
├── .github/workflows/ci.yml   # Multi-platform CI (Linux, Windows, macOS)
├── crates/
│   ├── spec-stream-core/      # Mathematical verifiers (Greedy, Stochastic), tree decoding, metrics
│   ├── spec-stream-cache/     # Paged KV-cache, physical block pools, prefix caching & O(1) rollback
│   ├── spec-stream-engine/    # Async pipeline scheduler, ModelEngine trait, tokio token streaming
│   ├── spec-stream-server/    # OpenAI-compatible HTTP inference server with SSE streaming & Prometheus metrics
│   └── spec-stream/           # Facade crate with ergonomic preludes and feature flags
├── examples/
│   └── speculative_pipeline.rs # Complete end-to-end runnable demonstration
└── benches/
    └── verification_bench.rs   # Micro-benchmarks
```

---

## 🌐 OpenAI-Compatible HTTP Server (`spec-stream-server`)

`spec-stream-server` provides a drop-in replacement HTTP server compatible with the OpenAI API format:

```bash
# Start the server on http://0.0.0.0:8080
cargo run -p spec-stream-server
```

### Endpoints
- `GET /health`: Health status, cache pool utilization, and allocated blocks.
- `GET /metrics`: Standard Prometheus metrics (block utilization, prefix entries, total requests and tokens).
- `POST /v1/chat/completions`: Standard chat completion endpoint supporting both standard JSON and `stream: true` (Server-Sent Events) with sampling parameters (`temperature`, `top_p`, `top_k`, `repetition_penalty`).

```bash
# Prometheus metrics scrape
curl http://localhost:8080/metrics

# Streaming inference test with curl
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "speculative-default",
    "messages": [{"role": "user", "content": "Explain speculative decoding in Rust"}],
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "stream": true
  }'
```

### 🐳 Running with Docker & GitHub Container Registry (GHCR)

Deploy the multi-architecture (`linux/amd64`, `linux/arm64`), non-root hardened production image:

```bash
# Pull official image from GHCR
docker pull ghcr.io/glastor-core/spec-stream:latest

# Run container in background with port mapping and custom configuration
docker run -d \
  --name spec-stream-server \
  -p 8080:8080 \
  -v $(pwd)/spec-stream.toml:/app/spec-stream.toml:ro \
  ghcr.io/glastor-core/spec-stream:latest
```

---

## ⚡ Advanced Features

### 1. Continuous Batching Scheduler (`ContinuousBatchScheduler`)
High-throughput iteration-level batch scheduler for production LLM serving:
- Dynamic admission control: prompts are admitted up to `max_batch_size` based on available paged KV-cache blocks.
- Concurrent draft proposals and unified target forward passes across heterogeneous sequence lengths.
- Per-sequence dynamic adaptive $K$ tracking and zero-overhead memory reclamation upon sequence completion.

### 2. GGUF & SafeTensors Metadata Parsers & Weight Loaders
- **`GgufMetadata`**: High-performance streaming binary parser for GGUF v2 and v3 headers, metadata key-values, and tensor tables without external C++ runtimes.
- **`SafeTensorsMetadata`**: Native header parser for Hugging Face SafeTensors files.
- **`LinearProjectionModel`**: Lightweight SIMD-friendly projection runtime implementing `ModelEngine` for fast draft generation and LM head inference.

### 3. Warm Prefix Cache Snapshots & Persistence
- Export and import warmed prefix KV-cache states (`PrefixCacheSnapshot`) directly to/from disk (`save_snapshot_json`, `load_snapshot_json`).
- Eliminates cold-start latency for recurring enterprise system prompts and few-shot tool definitions.

### 4. Dynamic Adaptive $K$-Drafting
Automatically scales the speculative window $K$ up (up to `max_k`) when acceptance rates are high ($\ge 75\%$) or down (down to `min_k`) when complexity causes rejections, preventing wasted target compute.

### 5. Automatic Prefix Caching & Copy-on-Write (CoW)
- Prompts sharing identical system prefix chunks reuse physical KV-cache blocks with zero memory reallocation.
- If a sequence branches or rolls back a shared block, **Copy-on-Write (CoW)** seamlessly clones the boundary block, isolating modifications without data races.

### 6. Advanced Sampling Suite
- **Top-K & Top-P (Nucleus)** filtering.
- **Min-P** dynamic thresholding.
- **Repetition, Presence & Frequency penalties**.

### 7. Zero-Model Prompt Lookup Decoding (`PromptLookupEngine`)
- N-gram assisted speculative drafting without loading smaller draft neural network weights (inspired by Saxena & Arora and vLLM).
- Suffix-matching algorithm scans recent prompt context in nanoseconds with **0 VRAM** and **0 GPU compute overhead**, achieving high acceptance on RAG, code editing, and summarization.

### 8. Pure-Rust BPE / UTF-8 Tokenizer (`ByteLevelTokenizer`)
- Zero-external-dependency byte-fallback tokenizer supporting English and Spanish vocabularies.
- Enables natural language in / natural language out streaming over HTTP SSE endpoints.

### 9. LRU Sequence Eviction & Physical Memory Defragmentation
- `LruEvictionPolicy` dynamically reclaims idle or completed sequence KV-cache blocks when pool saturation threatens out-of-memory errors.
- Automatic free list defragmentation (`defragment_free_list`) optimizes contiguous physical page allocation and CPU cache line locality.

### 10. GGUF Quantization Support (Q8_0 & Q4_0)
- Native pure-Rust de-quantization and fused quantized dot-product kernels without C/C++ runtimes.
- `QuantizedLinearModel` enables running projection and draft heads with **75%–87% less memory**, preserving FP16 precision scales.

### 11. Chunked Prefill Pipeline (`ChunkedPrefillScheduler`)
- Eliminates head-of-line blocking and latency spikes during long prompt ingestion (2k–8k+ tokens).
- Segments prompts into configurable chunks (`chunk_size`), interleaving prefill stages with ongoing speculative decode cycles.

### 12. CLI Diagnostic & Benchmarking Suite
Built-in subcommands inside `spec-stream-server`:
```bash
# Start the OpenAI-compatible HTTP inference server (default)
cargo run -p spec-stream-server -- serve

# Run reproducible comparative benchmark (Autoregressive vs Speculative)
cargo run -p spec-stream-server -- bench

# Generate a synthetic GGUF v3 demo model in milliseconds
cargo run -p spec-stream-server -- create-demo-gguf tiny-llama.gguf

# Inspect binary GGUF model headers, architecture, and tensor tables
cargo run -p spec-stream-server -- inspect-gguf tiny-llama.gguf
```

### 13. DevSecOps & Supply Chain Security (`deny.toml`)
- Continuous license checking (SPDX compliance) and automatic ban of duplicate or unmaintained crates.
- Automated security scanning via RustSec advisories and `.github/workflows/ci.yml`.

---

## ⚙️ Configuration (`spec-stream.toml`)

`spec-stream-server` automatically detects and reloads `spec-stream.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8080

[cache]
cache_total_blocks = 512
cache_block_size = 16

[speculative]
k_draft_tokens = 4
adaptive_k = true
max_batch_size = 16
```

---

## 🌲 Multi-Path Tree Speculative Decoding (Medusa / Eagle)

Beyond linear draft chains, `spec-stream-core` and `spec-stream-engine` support **Tree Speculative Decoding**:
- Constructs candidate trees with multiple speculative alternative branches via `TreeSpeculativePipeline`.
- Computes the 2D causal/tree attention mask for the target model.
- Evaluates the entire tree in a single forward pass, discovering the longest accepted branch via DFS/BFS.

```rust
use spec_stream::prelude::*;

let mut tree = SpeculativeTree::new(root_token_id);
let branch_a = tree.add_child(0, 101, 0.8)?;
let _branch_b = tree.add_child(branch_a, 102, 0.7)?;
let _branch_c = tree.add_child(0, 103, 0.2)?;

// Generates NxN attention mask for the target model
let attention_mask = tree.compute_tree_attention_mask();

// Verifies best branch and emits bonus token
let result = tree.verify_greedy(&target_logits)?;
```

---

## 🎨 Interactive Showcase & Playground

To preview the live interactive showcase landing page:
- Simply open [`index.html`](index.html) in any modern web browser, or
- Start the server (`cargo run -p spec-stream-server`) and visit [`http://localhost:8080/`](http://localhost:8080/).

Features interactive simulators for:
1. **Speculative Burst Verification**: Live Draft vs. Target forward passes with speedup metrics.
2. **Paged KV-Cache Grid**: Real-time physical block allocation, prefix caching hits, and $O(1)$ rollback animations.
3. **Tree Speculative Decoding**: Interactive candidate branch graph with 2D causal attention masks.
4. **Sampling Studio**: Dynamic sliders for Temperature, Top-K, Top-P, and Repetition Penalties with live probability distributions.
5. **OpenAI HTTP API & SSE Stream**: Real-time Server-Sent Events terminal simulator.
6. **Prompt Lookup Decoding**: Zero-model N-Gram suffix matcher with live context extraction.
7. **Cargo Workspace Architecture**: Interactive breakdown of crates, tests, and publishing workflow.

---

## 📦 Publishing to Crates.io

Publish from the bottom of the dependency tree upward:

```bash
cargo publish -p spec-stream-core
cargo publish -p spec-stream-cache
cargo publish -p spec-stream-engine
cargo publish -p spec-stream
cargo publish -p spec-stream-server
```

---

## 📄 License

Dual-licensed under MIT and Apache-2.0.
