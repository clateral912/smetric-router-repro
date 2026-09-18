//! Prefix scoring over token transitions learned from vLLM KV events.

use super::index::KVBlockIndex;
use std::collections::HashMap;

#[derive(Debug)]
pub struct PrefixScorer<'a> {
    index: &'a KVBlockIndex,
}

#[derive(Debug, Clone)]
pub struct PrefixScoreResult {
    pub scores: HashMap<String, usize>,
    pub total_blocks: usize,
}

impl PrefixScoreResult {
    pub fn best_worker(&self) -> Option<(&str, usize)> {
        self.scores
            .iter()
            .max_by_key(|(_, score)| *score)
            .map(|(url, score)| (url.as_str(), *score))
    }
    pub fn match_ratio(&self, worker_url: &str) -> f32 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.scores.get(worker_url).copied().unwrap_or(0) as f32 / self.total_blocks as f32
    }
    pub fn uncached_tokens(&self, worker_url: &str, block_size: usize) -> usize {
        self.total_blocks
            .saturating_sub(self.scores.get(worker_url).copied().unwrap_or(0))
            .saturating_mul(block_size)
    }
}

impl<'a> PrefixScorer<'a> {
    pub fn new(index: &'a KVBlockIndex) -> Self {
        Self { index }
    }
    pub fn score_tokens(&self, token_ids: &[u32], block_size: usize) -> PrefixScoreResult {
        let (scores, total_blocks) = self.index.score_tokens(token_ids, block_size);
        PrefixScoreResult {
            scores,
            total_blocks,
        }
    }
}
