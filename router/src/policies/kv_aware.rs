//! KV-aware routing policy backed by real-time KV events from vLLM.
//!
//! Unlike the approximate [`CacheAwarePolicy`](super::CacheAwarePolicy) that
//! infers cache state from request routing history, this policy queries the
//! [`KVBlockIndex`] which is populated from actual `BlockStored` / `BlockRemoved`
//! events published by vLLM workers.  This gives an accurate, near-real-time
//! view of each worker's KV cache contents.
//!
//! # Scoring algorithm
//!
//! 1. Tokenize the request text using the configured HuggingFace tokenizer.
//! 2. Query [`KVBlockIndex`] through [`PrefixScorer`] to find the longest
//!    contiguous prefix match per worker.
//! 3. Select the healthy worker with the highest score.
//!
//! Falls back to least-load selection when no worker has any cached blocks.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::kv_index::{KVBlockIndex, PrefixScorer};
use crate::metrics::RouterMetrics;
use crate::tokenizer::traits::Encoder;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

/// Configuration for the KV-aware policy.
#[derive(Debug, Clone)]
pub struct KvAwareConfig {
    /// Block size in tokens (must match vLLM `--block-size`).
    pub block_size: usize,
    /// Hash seed (must match vLLM `PYTHONHASHSEED`).
    pub hash_seed: u64,
    /// Retained for configuration compatibility. Confirmed event state is
    /// never polluted with speculative entries.
    pub enable_speculative: bool,
    /// TTL for speculative index entries.
    pub speculative_ttl: Duration,
}

impl Default for KvAwareConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            hash_seed: 0,
            enable_speculative: true,
            speculative_ttl: Duration::from_secs(2),
        }
    }
}

/// Routing policy that uses real KV cache state from vLLM event streams.
pub struct KvAwarePolicy {
    config: KvAwareConfig,
    block_index: Arc<KVBlockIndex>,
    tokenizer: Arc<dyn Encoder>,
}

impl std::fmt::Debug for KvAwarePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvAwarePolicy")
            .field("config", &self.config)
            .field("block_index", &self.block_index)
            .finish()
    }
}

impl KvAwarePolicy {
    /// Create a new policy.
    ///
    /// # Arguments
    /// * `config` – Block size, hash seed, speculative settings.
    /// * `block_index` – Shared reference to the global KV block index.
    /// * `tokenizer` – Tokenizer for converting request text to token IDs.
    pub fn new(
        config: KvAwareConfig,
        block_index: Arc<KVBlockIndex>,
        tokenizer: Arc<dyn Encoder>,
    ) -> Self {
        Self {
            config,
            block_index,
            tokenizer,
        }
    }

    /// Tokenize request text and generate block keys.
    fn text_to_tokens(&self, text: &str) -> Vec<u32> {
        match self.tokenizer.encode(text) {
            Ok(encoding) => encoding.token_ids().to_vec(),
            Err(e) => {
                warn!("KvAwarePolicy: tokenization failed: {}", e);
                Vec::new()
            }
        }
    }

    /// Score workers and select the best one, with speculative indexing.
    fn select_with_scoring(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        token_ids: &[u32],
    ) -> Option<usize> {
        let scorer = PrefixScorer::new(&self.block_index);
        let result = scorer.score_tokens(token_ids, self.config.block_size);

        // Find the healthy worker with the highest prefix score.
        let best_idx = healthy_indices
            .iter()
            .max_by_key(|&&idx| result.scores.get(workers[idx].url()).copied().unwrap_or(0))
            .copied()?;

        let best_score = result
            .scores
            .get(workers[best_idx].url())
            .copied()
            .unwrap_or(0);

        debug!(
            "KvAware: best worker {} with prefix score {}/{} ({:.1}%)",
            workers[best_idx].url(),
            best_score,
            result.total_blocks,
            if result.total_blocks == 0 {
                0.0
            } else {
                best_score as f64 / result.total_blocks as f64 * 100.0
            }
        );

        // If no worker has any cached blocks, fall back to least-load.
        let selected = if best_score == 0 {
            debug!("KvAware: no cache hits, falling back to least-load");
            healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied()?
        } else {
            best_idx
        };

        Some(selected)
    }
}

impl LoadBalancingPolicy for KvAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        // Single healthy worker: skip scoring.
        if healthy_indices.len() == 1 {
            return Some(healthy_indices[0]);
        }

        let text = request_text.unwrap_or("");

        // Generate block keys from request text.
        let token_ids = self.text_to_tokens(text);

        if token_ids.len() < self.config.block_size {
            // Prompt too short for even one full block; use least-load.
            return healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied();
        }

        let selected = self.select_with_scoring(workers, &healthy_indices, &token_ids)?;

        workers[selected].increment_processed();
        RouterMetrics::record_processed_request(workers[selected].url());
        RouterMetrics::record_policy_decision(self.name(), workers[selected].url());

        Some(selected)
    }

    fn name(&self) -> &'static str {
        "kv_aware"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
