//! # Tokenizer
//!
//! Provides lightweight, pure-Rust tokenization and detokenization routines
//! with byte-fallback support and subword vocabulary mapping.

use crate::logits::TokenId;
use std::collections::HashMap;

/// Standard abstract interface for tokenizers in `spec-stream`.
pub trait Tokenizer: Send + Sync {
    /// Encodes a UTF-8 text string into a sequence of token IDs.
    fn encode(&self, text: &str) -> Vec<TokenId>;

    /// Decodes a sequence of token IDs back into a UTF-8 string.
    fn decode(&self, tokens: &[TokenId]) -> String;

    /// Total vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Maps a token ID to its string piece representation, if present.
    fn token_to_piece(&self, token: TokenId) -> Option<&str>;

    /// Maps a string piece to its token ID, if present.
    fn piece_to_token(&self, piece: &str) -> Option<TokenId>;
}

/// A robust byte-level tokenizer with subword vocabulary lookup and lossless byte-fallback.
#[derive(Debug, Clone)]
pub struct ByteLevelTokenizer {
    token_to_piece_map: HashMap<TokenId, String>,
    piece_to_token_map: HashMap<String, TokenId>,
    vocab_size: usize,
    pub eos_token_id: TokenId,
    pub bos_token_id: TokenId,
}

impl ByteLevelTokenizer {
    /// Creates a new tokenizer with common English and Spanish subwords plus a 256-byte fallback table.
    pub fn new() -> Self {
        let mut token_to_piece_map = HashMap::new();
        let mut piece_to_token_map = HashMap::new();

        // 0..256: Single byte fallback tokens
        for b in 0..=255u8 {
            let id = b as TokenId;
            let s = (b as char).to_string();
            token_to_piece_map.insert(id, s.clone());
            piece_to_token_map.insert(s, id);
        }

        let mut next_id = 256 as TokenId;

        // Special tokens
        let bos_id = next_id;
        token_to_piece_map.insert(bos_id, "<s>".to_string());
        piece_to_token_map.insert("<s>".to_string(), bos_id);
        next_id += 1;

        let eos_id = next_id;
        token_to_piece_map.insert(eos_id, "</s>".to_string());
        piece_to_token_map.insert("</s>".to_string(), eos_id);
        next_id += 1;

        // Frequent subwords and words (English + Spanish + Code)
        let subwords = [
            " ", "  ", "   ", "    ", "\n", "\n\n",
            "the", "The", "be", "to", "of", "and", "a", "in", "that", "have",
            "I", "it", "for", "not", "on", "with", "he", "as", "you", "do",
            "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
            "most", "us", "spec", "stream", "speculative", "decoding", "cache", "token", "model",
            "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "en", "que", "es",
            "por", "para", "con", "no", "si", "su", "al", "lo", "como", "mas", "más",
            "pero", "sus", "le", "ya", "o", "fue", "este", "ha", "me", "si", "sin",
            "sobre", "este", "ya", "entre", "cuando", "todo", "esta", "ser", "son", "dos",
            "también", "tambien", "era", "muy", "hasta", "desde", "mi", "porque", "cada",
            "rust", "fn", "let", "mut", "pub", "struct", "impl", "enum", "match", "async",
            "await", "return", "true", "false", "None", "Some", "Ok", "Err", "Self", "vec!",
        ];

        for &word in &subwords {
            if !piece_to_token_map.contains_key(word) {
                token_to_piece_map.insert(next_id, word.to_string());
                piece_to_token_map.insert(word.to_string(), next_id);
                next_id += 1;
            }
        }

        let vocab_size = next_id as usize;

        Self {
            token_to_piece_map,
            piece_to_token_map,
            vocab_size,
            eos_token_id: eos_id,
            bos_token_id: bos_id,
        }
    }

    /// Registers a custom token string piece in the vocabulary.
    pub fn add_token(&mut self, piece: impl Into<String>) -> TokenId {
        let piece_str = piece.into();
        if let Some(&id) = self.piece_to_token_map.get(&piece_str) {
            return id;
        }

        let id = self.vocab_size as TokenId;
        self.token_to_piece_map.insert(id, piece_str.clone());
        self.piece_to_token_map.insert(piece_str, id);
        self.vocab_size += 1;
        id
    }
}

impl Default for ByteLevelTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for ByteLevelTokenizer {
    fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut tokens = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Greedy longest matching piece search
            let mut matched_len = 0;
            let mut matched_id = None;

            // Try matching prefixes up to 24 chars
            let max_check = remaining.len().min(24);
            for len in (1..=max_check).rev() {
                if let Some(sub) = remaining.get(..len) {
                    if let Some(&id) = self.piece_to_token_map.get(sub) {
                        matched_len = len;
                        matched_id = Some(id);
                        break;
                    }
                }
            }

            if let (Some(id), len) = (matched_id, matched_len) {
                tokens.push(id);
                remaining = &remaining[len..];
            } else {
                // Byte fallback
                let b = remaining.as_bytes()[0];
                tokens.push(b as TokenId);
                remaining = &remaining[1..];
            }
        }

        tokens
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        let mut out = String::new();
        let mut raw_bytes = Vec::new();

        for &tok in tokens {
            if let Some(piece) = self.token_to_piece_map.get(&tok) {
                // If we accumulated raw bytes, flush them to UTF-8
                if !raw_bytes.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&raw_bytes));
                    raw_bytes.clear();
                }

                // Skip special tags like <s> or </s>
                if piece == "<s>" || piece == "</s>" {
                    continue;
                }
                out.push_str(piece);
            } else if tok < 256 {
                raw_bytes.push(tok as u8);
            } else {
                // Unknown high token fallback
                if !raw_bytes.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&raw_bytes));
                    raw_bytes.clear();
                }
                out.push_str(&format!("<|token_{tok}|>"));
            }
        }

        if !raw_bytes.is_empty() {
            out.push_str(&String::from_utf8_lossy(&raw_bytes));
        }

        out
    }

    #[inline]
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[inline]
    fn token_to_piece(&self, token: TokenId) -> Option<&str> {
        self.token_to_piece_map.get(&token).map(|s| s.as_str())
    }

    #[inline]
    fn piece_to_token(&self, piece: &str) -> Option<TokenId> {
        self.piece_to_token_map.get(piece).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_level_tokenizer_roundtrip() {
        let tokenizer = ByteLevelTokenizer::new();
        let input = "Speculative decoding with spec-stream in Rust!";
        let encoded = tokenizer.encode(input);
        assert!(!encoded.is_empty());

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_tokenizer_spanish_roundtrip() {
        let tokenizer = ByteLevelTokenizer::new();
        let input = "Inferencia acelerada con caché de prefijos y decodificación especulativa.";
        let encoded = tokenizer.encode(input);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = ByteLevelTokenizer::new();
        let tokens = vec![tokenizer.bos_token_id, 100, tokenizer.eos_token_id];
        let decoded = tokenizer.decode(&tokens);
        // Special tokens <s> and </s> are stripped from clean output
        assert!(!decoded.contains("<s>"));
        assert!(!decoded.contains("</s>"));
    }
}
