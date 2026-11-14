# syntax=docker/dockerfile:1.4
# ==============================================================================
# spec-stream: High-Performance Speculative Decoding & Paged KV-Cache Engine
# Multi-stage Hardened Production Dockerfile
# ==============================================================================

# Stage 1: Build Workspace
FROM rust:1.85-bookworm AS builder

WORKDIR /usr/src/spec-stream

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifest and lockfile
COPY Cargo.toml Cargo.lock ./

# Copy all workspace crates
COPY crates ./crates

# Build release binary for spec-stream-server
RUN cargo build --release --bin spec-stream-server

# ==============================================================================
# Stage 2: Minimal Hardened Runtime
# ==============================================================================
FROM debian:bookworm-slim AS runtime

# Install CA certificates and curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin specstream

# Setup application directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder --chown=specstream:specstream /usr/src/spec-stream/target/release/spec-stream-server /usr/local/bin/spec-stream-server

# Copy default configuration file
COPY --chown=specstream:specstream spec-stream.toml /app/spec-stream.toml

# Switch to non-root unprivileged user
USER specstream:specstream

# Default HTTP port
EXPOSE 8080

# Healthcheck configuration
HEALTHCHECK --interval=20s --timeout=4s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Launch the OpenAI SSE & JSON speculative server
ENTRYPOINT ["/usr/local/bin/spec-stream-server"]
CMD ["serve", "--config", "/app/spec-stream.toml"]
