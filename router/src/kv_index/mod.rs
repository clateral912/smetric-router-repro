//! Precise KV block index and scoring for cache-aware routing.
//!
//! This module maintains a global, near-real-time index that maps KV block
//! opaque engine block hashes to workers and learns token-block transitions
//! from the events themselves. This avoids reproducing vLLM's configurable
//! hash algorithm in the router.
//!
//! # Components
//!
//! * [`KVBlockIndex`] – Thread-safe map from block hash → worker locations,
//!   updated via [`KVEvent`]s from the event pool.
//! * [`PrefixScorer`] – Given request tokens, scores each worker
//!   by the length of the longest contiguous prefix match.
//! * [`run_kv_index_updater`] – Background task that consumes events from
//!   the [`KVEventPool`] channel and applies them to the index.

pub mod block_hash;
pub mod index;
pub mod scorer;
pub mod updater;

pub use block_hash::BlockKeyGenerator;
pub use index::KVBlockIndex;
pub use scorer::PrefixScorer;
pub use updater::run_kv_index_updater;
