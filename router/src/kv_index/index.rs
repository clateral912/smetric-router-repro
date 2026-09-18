//! Event-backed KV block index.
//!
//! vLLM owns the block hash algorithm. The router treats hashes as opaque and
//! learns `(parent hash, token block) -> child hash` edges from events.

use crate::kv_events::decoder::BlockHash;
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TransitionKey {
    parent: Option<BlockHash>,
    tokens: Vec<u32>,
}

pub struct KVBlockIndex {
    locations: DashMap<BlockHash, HashSet<String>>,
    transitions: DashMap<TransitionKey, HashSet<BlockHash>>,
    max_entries: usize,
}

impl std::fmt::Debug for KVBlockIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KVBlockIndex")
            .field("entries", &self.locations.len())
            .field("transitions", &self.transitions.len())
            .field("max_entries", &self.max_entries)
            .finish()
    }
}

impl KVBlockIndex {
    pub fn new(max_entries: usize) -> Self {
        Self {
            locations: DashMap::with_capacity(max_entries.min(1_000_000)),
            transitions: DashMap::new(),
            max_entries,
        }
    }

    pub fn on_blocks_stored(
        &self,
        block_hashes: &[BlockHash],
        parent_block_hash: Option<&BlockHash>,
        token_ids: &[u32],
        block_size: usize,
        worker_url: &str,
    ) -> bool {
        if block_size == 0
            || block_hashes.is_empty()
            || token_ids.len() < block_hashes.len() * block_size
        {
            return false;
        }
        let mut parent = parent_block_hash.cloned();
        for (hash, tokens) in block_hashes.iter().zip(token_ids.chunks_exact(block_size)) {
            self.transitions
                .entry(TransitionKey {
                    parent: parent.clone(),
                    tokens: tokens.to_vec(),
                })
                .or_default()
                .insert(hash.clone());
            self.locations
                .entry(hash.clone())
                .or_default()
                .insert(worker_url.to_string());
            parent = Some(hash.clone());
        }
        true
    }

    pub fn on_block_removed(&self, block_hash: &BlockHash, worker_url: &str) {
        if let Some(mut workers) = self.locations.get_mut(block_hash) {
            workers.remove(worker_url);
            if workers.is_empty() {
                drop(workers);
                self.locations.remove(block_hash);
            }
        }
    }

    pub fn on_all_blocks_cleared(&self, worker_url: &str) {
        let keys: Vec<BlockHash> = self.locations.iter().map(|e| e.key().clone()).collect();
        for key in keys {
            if let Some(mut workers) = self.locations.get_mut(&key) {
                workers.remove(worker_url);
            }
        }
        self.locations.retain(|_, workers| !workers.is_empty());
    }

    pub fn get_workers_for_block(&self, block_hash: &BlockHash) -> Vec<String> {
        self.locations
            .get(block_hash)
            .map(|workers| workers.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn score_tokens(
        &self,
        token_ids: &[u32],
        block_size: usize,
    ) -> (HashMap<String, usize>, usize) {
        if block_size == 0 {
            return (HashMap::new(), 0);
        }
        let total_blocks = token_ids.len() / block_size;
        let mut parents: HashSet<Option<BlockHash>> = HashSet::from([None]);
        let mut active: HashSet<String> = HashSet::new();
        let mut scores = HashMap::new();

        for (position, tokens) in token_ids.chunks_exact(block_size).enumerate() {
            let mut children = HashSet::new();
            for parent in &parents {
                let key = TransitionKey {
                    parent: parent.clone(),
                    tokens: tokens.to_vec(),
                };
                if let Some(found) = self.transitions.get(&key) {
                    children.extend(found.iter().cloned());
                }
            }
            if children.is_empty() {
                break;
            }
            let holding: HashSet<String> = children
                .iter()
                .flat_map(|hash| self.get_workers_for_block(hash))
                .collect();
            if position == 0 {
                active = holding;
            } else {
                active.retain(|w| holding.contains(w));
            }
            if active.is_empty() {
                break;
            }
            for worker in &active {
                scores.insert(worker.clone(), position + 1);
            }
            parents = children.into_iter().map(Some).collect();
        }
        (scores, total_blocks)
    }

    pub fn len(&self) -> usize {
        self.locations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hash(byte: u8) -> BlockHash {
        BlockHash::Bytes(vec![byte; 32])
    }

    #[test]
    fn indexes_sha256_events_and_scores_tokens() {
        let index = KVBlockIndex::new(100);
        assert!(index.on_blocks_stored(
            &[hash(1), hash(2)],
            None,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            4,
            "w1"
        ));
        let (scores, total) = index.score_tokens(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_eq!(total, 2);
        assert_eq!(scores.get("w1"), Some(&2));
    }

    #[test]
    fn removal_and_clear_remove_positive_evidence() {
        let index = KVBlockIndex::new(100);
        index.on_blocks_stored(&[hash(1)], None, &[1, 2, 3, 4], 4, "w1");
        index.on_blocks_stored(&[hash(1)], None, &[1, 2, 3, 4], 4, "w2");
        index.on_block_removed(&hash(1), "w1");
        assert_eq!(index.get_workers_for_block(&hash(1)), vec!["w2"]);
        index.on_all_blocks_cleared("w2");
        assert!(index.get_workers_for_block(&hash(1)).is_empty());
    }

    #[test]
    fn rejects_incomplete_event() {
        let index = KVBlockIndex::new(100);
        assert!(!index.on_blocks_stored(&[hash(1), hash(2)], None, &[1, 2, 3, 4], 4, "w"));
        assert!(index.is_empty());
    }
}
