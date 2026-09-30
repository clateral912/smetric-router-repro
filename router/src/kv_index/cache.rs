//! Common event-backed prefix source for the benchmark routing policies.
use super::{KVBlockIndex, PrefixScorer};
use crate::tokenizer::traits::{Encoder, Encoding};
use std::collections::HashMap;
use std::sync::Arc;

pub struct EventCache {
    pub index: Arc<KVBlockIndex>,
    pub tokenizer: Arc<dyn Encoder>,
    pub block_size: usize,
}

impl std::fmt::Debug for EventCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventCache").field("block_size", &self.block_size).finish()
    }
}

pub struct CachePrefixScores {
    pub length: usize,
    pub hits: HashMap<String, usize>,
}

impl EventCache {
    pub fn score_tokens(&self, tokens: &[u32]) -> CachePrefixScores {
        let mut hits = PrefixScorer::new(&self.index).score_tokens(tokens, self.block_size).scores;
        for hit in hits.values_mut() {
            *hit = hit.saturating_mul(self.block_size).min(tokens.len());
        }
        CachePrefixScores { length: tokens.len(), hits }
    }

    pub fn score_text(&self, text: &str) -> Option<CachePrefixScores> {
        // Native token-ID completion inputs use the existing reversible routing key.
        if let Some(encoded) = text.strip_prefix("\u{10ffff}\u{10fffe}") {
            let mut ids = Vec::with_capacity(encoded.chars().count());
            for scalar in encoded.chars().map(u32::from) {
                if scalar == 0x10fffd { return None; }
                ids.push(if scalar >= 0xe000 { scalar - 0x800 } else { scalar });
            }
            return Some(self.score_tokens(&ids));
        }
        let encoding = self.tokenizer.encode(text).ok()?;
        let Encoding::Hf(inner) = &encoding else { return None; };
        let offsets = inner.get_offsets();
        let tokens = encoding.token_ids();
        let mut scores = self.score_tokens(tokens);
        scores.length = text.chars().count();
        for hit in scores.hits.values_mut() {
            let count = *hit;
            let mut end = offsets[..count].iter().map(|&(_, end)| end).max().unwrap_or(0).min(text.len());
            // A block may end between byte-level tokens representing one character.
            if let Some(&(next_start, _)) = offsets.get(count) {
                end = end.min(next_start);
            }
            while !text.is_char_boundary(end) { end -= 1; }
            *hit = text[..end].chars().count();
        }
        Some(scores)
    }
}
