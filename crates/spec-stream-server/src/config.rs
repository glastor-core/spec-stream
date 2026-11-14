//! # Server Configuration
//!
//! Provides configuration loading from `spec-stream.toml` or JSON configuration files
//! with sensible production-ready defaults.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Configuration parameters for the `spec-stream` inference server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host address to bind the HTTP server to (e.g. `0.0.0.0` or `127.0.0.1`).
    pub host: String,
    /// TCP port for HTTP listening.
    pub port: u16,
    /// Total number of physical blocks allocated to the paged KV-cache pool.
    pub cache_total_blocks: usize,
    /// Number of tokens stored per physical KV-cache block.
    pub cache_block_size: usize,
    /// Initial or static speculative drafting burst depth ($K$).
    pub k_draft_tokens: usize,
    /// Whether dynamic EMA adaptive $K$-drafting is enabled.
    pub adaptive_k: bool,
    /// Maximum number of concurrent sequences in continuous batching.
    pub max_batch_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            cache_total_blocks: 512,
            cache_block_size: 16,
            k_draft_tokens: 4,
            adaptive_k: true,
            max_batch_size: 16,
        }
    }
}

impl ServerConfig {
    /// Loads server configuration from a `.toml` or `.json` file, falling back to defaults if not found.
    pub fn load_or_default(path: impl AsRef<Path>) -> Self {
        let p = path.as_ref();
        if !p.exists() {
            return Self::default();
        }

        match fs::read_to_string(p) {
            Ok(content) => Self::parse_str(&content).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Parses configuration from a TOML or JSON formatted string.
    pub fn parse_str(content: &str) -> Result<Self, String> {
        let trimmed = content.trim();
        if trimmed.starts_with('{') {
            return serde_json::from_str(trimmed)
                .map_err(|e| format!("Invalid JSON config: {}", e));
        }

        // Lightweight, zero-dependency TOML key-value parser for spec-stream.toml
        let mut config = Self::default();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                continue;
            }

            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim();
                let val = val.split('#').next().unwrap_or(val).trim().trim_matches('"');

                match key {
                    "host" => config.host = val.to_string(),
                    "port" => {
                        if let Ok(v) = val.parse::<u16>() {
                            config.port = v;
                        }
                    }
                    "cache_total_blocks" => {
                        if let Ok(v) = val.parse::<usize>() {
                            config.cache_total_blocks = v;
                        }
                    }
                    "cache_block_size" => {
                        if let Ok(v) = val.parse::<usize>() {
                            config.cache_block_size = v;
                        }
                    }
                    "k_draft_tokens" => {
                        if let Ok(v) = val.parse::<usize>() {
                            config.k_draft_tokens = v;
                        }
                    }
                    "adaptive_k" => {
                        if let Ok(v) = val.parse::<bool>() {
                            config.adaptive_k = v;
                        }
                    }
                    "max_batch_size" => {
                        if let Ok(v) = val.parse::<usize>() {
                            config.max_batch_size = v;
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(config)
    }

    /// Serializes configuration into a clean TOML string representation.
    pub fn to_toml_string(&self) -> String {
        format!(
            r#"# ==============================================================================
# spec-stream Configuration File
# Pure Rust Speculative Decoding & Paged KV-Cache Engine
# https://github.com/glastor-core/spec-stream
# Contact: ventas@glastor.es | Maintainer: glastor.info@gmail.com
# ==============================================================================

[server]
host = "{}"
port = {}

[cache]
cache_total_blocks = {}
cache_block_size = {}

[speculative]
k_draft_tokens = {}
adaptive_k = {}
max_batch_size = {}
"#,
            self.host,
            self.port,
            self.cache_total_blocks,
            self.cache_block_size,
            self.k_draft_tokens,
            self.adaptive_k,
            self.max_batch_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.port, 8080);
        assert_eq!(cfg.cache_total_blocks, 512);
        assert_eq!(cfg.k_draft_tokens, 4);
        assert!(cfg.adaptive_k);
    }

    #[test]
    fn test_parse_toml_string() {
        let toml_src = r#"
            # Spec Stream Server Settings
            [server]
            host = "127.0.0.1"
            port = 9090

            [cache]
            cache_total_blocks = 1024
            cache_block_size = 32

            [speculative]
            k_draft_tokens = 6
            adaptive_k = false
            max_batch_size = 32
        "#;

        let cfg = ServerConfig::parse_str(toml_src).unwrap();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 9090);
        assert_eq!(cfg.cache_total_blocks, 1024);
        assert_eq!(cfg.cache_block_size, 32);
        assert_eq!(cfg.k_draft_tokens, 6);
        assert!(!cfg.adaptive_k);
        assert_eq!(cfg.max_batch_size, 32);
    }

    #[test]
    fn test_to_toml_string_roundtrip() {
        let orig = ServerConfig::default();
        let serialized = orig.to_toml_string();
        let parsed = ServerConfig::parse_str(&serialized).unwrap();
        assert_eq!(orig, parsed);
    }
}

