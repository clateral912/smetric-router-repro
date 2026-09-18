//! Native SMetric routing policy.
//!
//! This implementation deliberately uses only state observable by the router:
//! request text, the router's approximate prefix tree, and request lifecycle
//! timing. It does not query workers or any external state store.

use super::{get_healthy_worker_indices, normalize_model_key, LoadBalancingPolicy, RequestHeaders};
use crate::config::{SMetricGate, SMetricPolicyConfig};
use crate::core::Worker;
use crate::kv_index::{KVBlockIndex, PrefixScorer};
use crate::metrics::RouterMetrics;
use crate::tokenizer::traits::Encoder;
use crate::tree::Tree;
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::debug;

const MAX_SESSION_LEASES: usize = 100_000;

#[derive(Debug, Clone)]
struct SessionLease {
    worker_url: String,
    last_stuck: bool,
    last_seen: Instant,
}

pub type SMetricConfig = SMetricPolicyConfig;

#[derive(Clone)]
struct ExactCacheState {
    index: Arc<KVBlockIndex>,
    tokenizer: Arc<dyn Encoder>,
    block_size: usize,
}

impl std::fmt::Debug for ExactCacheState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExactCacheState")
            .field("index", &self.index)
            .field("block_size", &self.block_size)
            .finish()
    }
}

#[derive(Debug)]
struct Reservation {
    started_at: Instant,
    input_units: f64,
    prefill_units: f64,
    work_units: f64,
    rate_recorded: bool,
    response_started: bool,
}

#[derive(Debug, Default)]
struct WorkerState {
    pending_prefill_units: f64,
    decode_context_units: f64,
    pending_work: f64,
    active_requests: usize,
    reservations: HashMap<u64, Reservation>,
    /// Completed prefill observations: (completion time, start, effective work).
    /// The estimator merges overlapping [start, completion] intervals, so
    /// concurrent requests do not double-count worker busy wall time and idle
    /// gaps do not inflate the denominator.
    drain_samples: VecDeque<(Instant, Instant, f64)>,
}

#[derive(Debug)]
pub struct SMetricPolicy {
    config: SMetricConfig,
    trees: DashMap<String, Arc<Tree>>,
    worker_models: DashMap<String, String>,
    state: Mutex<HashMap<String, WorkerState>>,
    rr: AtomicUsize,
    next_reservation: AtomicU64,
    last_eviction: Mutex<Instant>,
    session_leases: Mutex<HashMap<String, SessionLease>>,
    exact_cache: Option<ExactCacheState>,
}

impl SMetricPolicy {
    pub fn new() -> Self {
        Self::with_config(SMetricConfig::default())
    }

    pub fn with_config(config: SMetricConfig) -> Self {
        Self {
            config,
            trees: DashMap::new(),
            worker_models: DashMap::new(),
            state: Mutex::new(HashMap::new()),
            rr: AtomicUsize::new(0),
            next_reservation: AtomicU64::new(1),
            last_eviction: Mutex::new(Instant::now()),
            session_leases: Mutex::new(HashMap::new()),
            exact_cache: None,
        }
    }

    /// Attach the same event-fed exact KV index used by cache_aware. When the
    /// index has no positive evidence, preserve the historical Tree fallback.
    pub fn with_exact_cache(
        config: SMetricConfig,
        index: Arc<KVBlockIndex>,
        tokenizer: Arc<dyn Encoder>,
        block_size: usize,
    ) -> Self {
        let mut policy = Self::with_config(config);
        policy.exact_cache = Some(ExactCacheState {
            index,
            tokenizer,
            block_size,
        });
        policy
    }

    fn token_ids_for_exact_cache(text: &str) -> Option<Vec<u32>> {
        let encoded = text.strip_prefix("\u{10ffff}\u{10fffe}")?;
        let mut ids = Vec::with_capacity(encoded.chars().count());
        for scalar in encoded.chars().map(u32::from) {
            if scalar == 0x10fffd {
                return None;
            }
            ids.push(if scalar >= 0xe000 {
                scalar - 0x800
            } else {
                scalar
            });
        }
        Some(ids)
    }

    /// Return request token count and exact cached-token counts by worker.
    /// A present result with all-zero scores means the event index has no
    /// positive evidence and callers should use the approximate Tree fallback.
    fn exact_cache_scores(&self, text: &str) -> Option<(usize, HashMap<String, usize>)> {
        let exact = self.exact_cache.as_ref()?;
        let token_ids = Self::token_ids_for_exact_cache(text).or_else(|| {
            exact
                .tokenizer
                .encode(text)
                .ok()
                .map(|encoding| encoding.token_ids().to_vec())
        })?;
        let scored = PrefixScorer::new(&exact.index).score_tokens(&token_ids, exact.block_size);
        let cached = scored
            .scores
            .into_iter()
            .map(|(worker, blocks)| {
                (
                    worker,
                    blocks.saturating_mul(exact.block_size).min(token_ids.len()),
                )
            })
            .collect();
        Some((token_ids.len(), cached))
    }

    #[inline]
    fn work_units(&self, uncached: usize, input: usize) -> f64 {
        let n = uncached as f64;
        let l = input as f64;
        n + n * (l - n / 2.0).max(0.0) / self.config.attention_l_eq
    }

    fn choose_tie(&self, candidates: &[usize]) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        let n = self.rr.fetch_add(1, Ordering::Relaxed);
        Some(candidates[n % candidates.len()])
    }

    fn maybe_evict(&self) {
        if self.config.eviction_interval_secs == 0 {
            return;
        }
        let mut last = self.last_eviction.lock().unwrap();
        if last.elapsed() < Duration::from_secs(self.config.eviction_interval_secs) {
            return;
        }
        for tree in self.trees.iter() {
            tree.value().evict_tenant_by_size(self.config.max_tree_size);
        }
        *last = Instant::now();
    }

    fn estimated_rate(&self, worker: &mut WorkerState, now: Instant) -> f64 {
        let window = Duration::from_secs(self.config.drain_window_secs);
        let window_start = now.checked_sub(window).unwrap_or(now);
        worker
            .drain_samples
            .retain(|(completion, _, _)| *completion >= window_start && *completion <= now);
        if worker.drain_samples.len() < self.config.drain_min_samples {
            return self.config.drain_tps_fallback;
        }
        let work: f64 = worker.drain_samples.iter().map(|(_, _, work)| *work).sum();
        let mut intervals: Vec<(Instant, Instant)> = worker
            .drain_samples
            .iter()
            .map(|(completion, start, _)| {
                (
                    std::cmp::max(*start, window_start),
                    std::cmp::min(*completion, now),
                )
            })
            .collect();
        intervals.sort_by_key(|(start, _)| *start);
        let mut busy_secs = 0.0;
        let mut merged: Option<(Instant, Instant)> = None;
        for (start, end) in intervals {
            if end <= start {
                continue;
            }
            if let Some((merged_start, merged_end)) = merged {
                if start <= merged_end {
                    merged = Some((merged_start, merged_end.max(end)));
                } else {
                    busy_secs += merged_end.duration_since(merged_start).as_secs_f64();
                    merged = Some((start, end));
                }
            } else {
                merged = Some((start, end));
            }
        }
        if let Some((start, end)) = merged {
            busy_secs += end.duration_since(start).as_secs_f64();
        }
        if !work.is_finite() || !busy_secs.is_finite() || busy_secs <= 0.0 {
            return self.config.drain_tps_fallback;
        }
        work / busy_secs
    }

    /// Return the aggregate rate and a relative uncertainty estimate derived
    /// from the same completed observations. This uncertainty is used only
    /// for session-lease hysteresis around the budget boundary.
    fn estimated_rate_with_uncertainty(
        &self,
        worker: &mut WorkerState,
        now: Instant,
    ) -> (f64, f64) {
        let rate = self.estimated_rate(worker, now);
        if worker.drain_samples.len() < self.config.drain_min_samples || rate <= 0.0 {
            return (rate, 0.0);
        }
        let mut rates: Vec<f64> = worker
            .drain_samples
            .iter()
            .filter_map(|(completion, start, work)| {
                let secs = completion.duration_since(*start).as_secs_f64();
                (secs > 0.0 && work.is_finite()).then_some(*work / secs)
            })
            .collect();
        if rates.len() < self.config.drain_min_samples {
            return (rate, 0.0);
        }
        rates.sort_by(f64::total_cmp);
        let median = rates[rates.len() / 2];
        let mut deviations: Vec<f64> = rates.iter().map(|value| (value - median).abs()).collect();
        deviations.sort_by(f64::total_cmp);
        let mad = deviations[deviations.len() / 2];
        if !median.is_finite() || median <= 0.0 || !mad.is_finite() {
            return (rate, 0.0);
        }
        // 1.96 is the conventional two-sided 95% normal quantile; 1.4826
        // makes MAD consistent with standard deviation for normal noise.
        let relative = 1.96 * 1.4826 * mad / median / (rates.len() as f64).sqrt();
        (rate, relative.max(0.0))
    }

    fn session_id(headers: Option<&RequestHeaders>) -> Option<&str> {
        headers.and_then(|values| values.get("x-session-id").map(String::as_str))
    }

    fn leased_worker(
        &self,
        session_id: &str,
        workers: &[Arc<dyn Worker>],
        now: Instant,
    ) -> Option<(usize, bool)> {
        let mut leases = self.session_leases.lock().unwrap();
        let ttl = Duration::from_secs(self.config.drain_window_secs);
        leases.retain(|_, lease| now.duration_since(lease.last_seen) <= ttl);
        let lease = leases.get_mut(session_id)?;
        lease.last_seen = now;
        workers
            .iter()
            .position(|worker| worker.url() == lease.worker_url)
            .map(|idx| (idx, lease.last_stuck))
    }

    fn remember_lease(&self, session_id: &str, worker_url: &str, last_stuck: bool, now: Instant) {
        let mut leases = self.session_leases.lock().unwrap();
        let ttl = Duration::from_secs(self.config.drain_window_secs);
        leases.retain(|_, lease| now.duration_since(lease.last_seen) <= ttl);
        if !leases.contains_key(session_id) && leases.len() >= MAX_SESSION_LEASES {
            if let Some(oldest) = leases
                .iter()
                .min_by_key(|(_, lease)| lease.last_seen)
                .map(|(key, _)| key.clone())
            {
                leases.remove(&oldest);
            }
        }
        leases.insert(
            session_id.to_string(),
            SessionLease {
                worker_url: worker_url.to_string(),
                last_stuck,
                last_seen: now,
            },
        );
    }

    #[inline]
    fn budget_accepts(predicted: f64, budget: f64, uncertainty: f64, leased_stuck: bool) -> bool {
        predicted <= budget || (leased_stuck && predicted <= budget * (1.0 + uncertainty))
    }

    #[cfg(test)]
    fn snapshot(&self, worker_url: &str) -> (f64, f64, usize, usize) {
        let state = self.state.lock().unwrap();
        let worker = state.get(worker_url).unwrap();
        (
            worker.pending_prefill_units + worker.decode_context_units,
            worker.pending_work,
            worker.active_requests,
            worker.drain_samples.len(),
        )
    }

    #[cfg(test)]
    fn measured_rate(&self, worker_url: &str) -> f64 {
        let mut state = self.state.lock().unwrap();
        self.estimated_rate(state.get_mut(worker_url).unwrap(), Instant::now())
    }
}

impl Default for SMetricPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingPolicy for SMetricPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        let text = request_text.unwrap_or("");
        let exact = self.exact_cache_scores(text);
        let exact_has_match = exact
            .as_ref()
            .is_some_and(|(_, scores)| scores.values().any(|score| *score > 0));
        let input = exact
            .as_ref()
            .map(|(tokens, _)| *tokens)
            .unwrap_or_else(|| text.chars().count());
        let model = normalize_model_key(workers[healthy[0]].model_id());
        let tree = self.trees.get(model).map(|v| Arc::clone(v.value()));
        let matched = tree.as_ref().map(|t| t.prefix_match_with_counts(text));
        let (home, hit) = if exact_has_match {
            healthy
                .iter()
                .copied()
                .map(|idx| {
                    let score = exact
                        .as_ref()
                        .and_then(|(_, scores)| scores.get(workers[idx].url()))
                        .copied()
                        .unwrap_or(0);
                    (idx, score)
                })
                .max_by_key(|(_, score)| *score)
                .map_or((None, 0), |(idx, score)| (Some(idx), score))
        } else {
            let home = matched.as_ref().and_then(|m| {
                healthy
                    .iter()
                    .copied()
                    .find(|&idx| workers[idx].url() == m.tenant.as_ref())
            });
            let hit = matched
                .as_ref()
                .map_or(0, |m| m.matched_char_count.min(input));
            (home, hit)
        };
        let session_id = Self::session_id(headers);
        let lease = session_id.and_then(|id| self.leased_worker(id, workers, Instant::now()));
        let leased_home = lease.map(|(idx, _)| idx);
        let leased_stuck = lease.is_some_and(|(_, stuck)| stuck);

        let mut states = self.state.lock().unwrap();
        for &idx in &healthy {
            states.entry(workers[idx].url().to_string()).or_default();
        }

        let load = |state: &WorkerState| {
            self.config.prefill_load_scale
                * (state.pending_prefill_units + state.decode_context_units)
                / self.config.block_size as f64
                + self.config.active_request_weight * state.active_requests as f64
        };
        let mean_load = healthy
            .iter()
            .map(|&idx| load(states.get(workers[idx].url()).unwrap()))
            .sum::<f64>()
            / healthy.len() as f64;

        let explicitly_first = headers
            .and_then(|h| h.get("x-session-turn"))
            .and_then(|v| v.parse::<u64>().ok())
            .is_some_and(|turn| turn <= 1);
        let hit_ok = input > 0 && (hit as f64) > self.config.hit_ratio * input as f64;

        // Keep the budget-gate inputs in the decision log so benchmark runs
        // can explain a stick/fallback choice without consulting worker
        // internals. These are router-observed values only.
        let mut budget_observation: Option<(f64, f64, f64, f64, f64, f64)> = None;
        let stick = home.filter(|&home_idx| {
            if explicitly_first || !hit_ok {
                return false;
            }
            let home_state = states.get_mut(workers[home_idx].url()).unwrap();
            match self.config.gate {
                SMetricGate::Overload => {
                    let home_load = load(home_state);
                    home_load <= self.config.overload_factor * mean_load
                        || (home_load == 0.0 && mean_load == 0.0)
                }
                SMetricGate::BudgetAttention => {
                    let own = self.work_units(input.saturating_sub(hit), input);
                    let queued = home_state.pending_work + own;
                    let (rate, uncertainty) =
                        self.estimated_rate_with_uncertainty(home_state, Instant::now());
                    let budget = self.config.budget_gamma
                        * (self.config.budget_base_s
                            + input as f64 / self.config.budget_input_tokens_per_s);
                    let predicted = queued / rate;
                    budget_observation = Some((queued, own, rate, predicted, budget, uncertainty));
                    Self::budget_accepts(
                        predicted,
                        budget,
                        uncertainty,
                        leased_stuck && leased_home == Some(home_idx),
                    )
                }
            }
        });

        let selected = if let Some(idx) = stick {
            idx
        } else {
            // Native Dynamo-style fallback. The approximate tree provides a
            // device-prefix credit for the best matching worker. Host credit
            // is zero because Router has no native host-cache observation.
            let min_pending = healthy
                .iter()
                .map(|&idx| {
                    let state = states.get(workers[idx].url()).unwrap();
                    state.pending_prefill_units + state.decode_context_units
                })
                .fold(f64::INFINITY, f64::min);
            let request_blocks = (input as f64 / self.config.block_size as f64).max(1.0);
            let mut scores = Vec::with_capacity(healthy.len());
            for &idx in &healthy {
                let worker = states.get(workers[idx].url()).unwrap();
                let current_load = worker.pending_prefill_units + worker.decode_context_units;
                let excess_blocks =
                    (current_load - min_pending).max(0.0) / self.config.block_size as f64;
                let decay = if self.config.overlap_score_credit_decay > 0.0 {
                    1.0 / (1.0
                        + self.config.overlap_score_credit_decay * excess_blocks / request_blocks)
                } else {
                    1.0
                };
                let device_hit = if exact_has_match {
                    exact
                        .as_ref()
                        .and_then(|(_, scores)| scores.get(workers[idx].url()))
                        .copied()
                        .unwrap_or(0)
                } else if Some(idx) == home {
                    hit
                } else {
                    0
                };
                let raw_blocks = (current_load + input as f64) / self.config.block_size as f64;
                let credit = self.config.overlap_score_credit * decay * device_hit as f64
                    / self.config.block_size as f64;
                let score = self.config.prefill_load_scale * (raw_blocks - credit).max(0.0)
                    + self.config.active_request_weight * worker.active_requests as f64;
                scores.push((idx, score));
            }
            let best = scores.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
            let tied: Vec<usize> = scores
                .into_iter()
                .filter_map(|(idx, score)| (score == best).then_some(idx))
                .collect();
            self.choose_tie(&tied)?
        };
        drop(states);

        if let Some(session_id) = session_id {
            self.remember_lease(
                session_id,
                workers[selected].url(),
                stick.is_some(),
                Instant::now(),
            );
        }
        workers[selected].increment_processed();
        RouterMetrics::record_processed_request(workers[selected].url());
        RouterMetrics::record_policy_decision(self.name(), workers[selected].url());
        debug!(
            worker = workers[selected].url(),
            matched_units = hit,
            input_units = input,
            stuck = stick.is_some(),
            exact_kv = exact_has_match,
            home = home.map(|idx| workers[idx].url()),
            queued_work = budget_observation.map(|v| v.0),
            own_work = budget_observation.map(|v| v.1),
            estimated_rate = budget_observation.map(|v| v.2),
            predicted_s = budget_observation.map(|v| v.3),
            budget_s = budget_observation.map(|v| v.4),
            rate_uncertainty = budget_observation.map(|v| v.5),
            stick = stick.is_some(),
            "SMetric decision"
        );
        Some(selected)
    }

    fn on_request_start(
        &self,
        worker_url: &str,
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<u64> {
        let text = request_text.unwrap_or("");
        let exact = self.exact_cache_scores(text);
        let exact_has_match = exact
            .as_ref()
            .is_some_and(|(_, scores)| scores.values().any(|score| *score > 0));
        let input = exact
            .as_ref()
            .map(|(tokens, _)| *tokens)
            .unwrap_or_else(|| text.chars().count());
        let model = self.worker_models.get(worker_url)?.clone();
        let tree = self.trees.get(model.as_str())?.clone();
        let matched = tree.prefix_match_with_counts(text);
        let hit = if exact_has_match {
            exact
                .as_ref()
                .and_then(|(_, scores)| scores.get(worker_url))
                .copied()
                .unwrap_or(0)
        } else if matched.tenant.as_ref() == worker_url {
            matched.matched_char_count.min(input)
        } else {
            0
        };
        let work = self.work_units(input.saturating_sub(hit), input);
        let prefill = input.saturating_sub(hit) as f64;
        let reservation_id = self.next_reservation.fetch_add(1, Ordering::Relaxed);

        let mut state = self.state.lock().unwrap();
        let worker = state.entry(worker_url.to_string()).or_default();
        worker.pending_prefill_units += prefill;
        worker.pending_work += work;
        worker.active_requests += 1;
        worker.reservations.insert(
            reservation_id,
            Reservation {
                started_at: Instant::now(),
                input_units: input as f64,
                prefill_units: prefill,
                work_units: work,
                rate_recorded: false,
                response_started: false,
            },
        );
        drop(state);

        tree.insert(text, worker_url);
        self.maybe_evict();
        Some(reservation_id)
    }

    fn on_request_first_response(
        &self,
        worker_url: &str,
        reservation_id: Option<u64>,
        elapsed: Duration,
    ) {
        let Some(id) = reservation_id else { return };
        let mut state = self.state.lock().unwrap();
        let Some(worker) = state.get_mut(worker_url) else {
            return;
        };
        let sample = worker.reservations.get_mut(&id).and_then(|reservation| {
            if reservation.response_started {
                return None;
            }
            reservation.response_started = true;
            worker.pending_prefill_units =
                (worker.pending_prefill_units - reservation.prefill_units).max(0.0);
            worker.pending_work = (worker.pending_work - reservation.work_units).max(0.0);
            worker.decode_context_units += reservation.input_units;
            if reservation.rate_recorded || reservation.work_units <= 0.0 || elapsed.is_zero() {
                return None;
            }
            reservation.rate_recorded = true;
            Some((reservation.started_at, reservation.work_units))
        });
        if let Some((started_at, work)) = sample.filter(|(_, work)| work.is_finite() && *work > 0.0)
        {
            worker
                .drain_samples
                .push_back((Instant::now(), started_at, work));
        }
    }

    fn on_request_complete(
        &self,
        worker_url: &str,
        reservation_id: Option<u64>,
        success: bool,
        elapsed: Duration,
    ) {
        let Some(id) = reservation_id else { return };
        if success {
            self.on_request_first_response(worker_url, Some(id), elapsed);
        }
        let mut state = self.state.lock().unwrap();
        let Some(worker) = state.get_mut(worker_url) else {
            return;
        };
        if let Some(reservation) = worker.reservations.remove(&id) {
            if reservation.response_started {
                worker.decode_context_units =
                    (worker.decode_context_units - reservation.input_units).max(0.0);
            } else {
                worker.pending_prefill_units =
                    (worker.pending_prefill_units - reservation.prefill_units).max(0.0);
                worker.pending_work = (worker.pending_work - reservation.work_units).max(0.0);
            }
            worker.active_requests = worker.active_requests.saturating_sub(1);
        }
    }

    fn name(&self) -> &'static str {
        "smetric"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn tracks_worker_load(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn requires_initialization(&self) -> bool {
        true
    }

    fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        let mut state = self.state.lock().unwrap();
        for worker in workers {
            let model = normalize_model_key(worker.model_id()).to_string();
            let tree = self
                .trees
                .entry(model.clone())
                .or_insert_with(|| Arc::new(Tree::new()))
                .clone();
            tree.insert("", worker.url());
            self.worker_models.insert(worker.url().to_string(), model);
            state.entry(worker.url().to_string()).or_default();
        }
    }

    fn remove_worker_by_url(&self, url: &str) {
        for tree in self.trees.iter() {
            tree.value().remove_tenant(url);
        }
        self.worker_models.remove(url);
        self.state.lock().unwrap().remove(url);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use crate::kv_events::decoder::BlockHash;
    use crate::tokenizer::mock::MockTokenizer;

    fn workers() -> Vec<Arc<dyn Worker>> {
        vec![
            Arc::new(BasicWorker::new("http://w0".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://w1".into(), WorkerType::Regular)),
        ]
    }

    #[test]
    fn exact_event_match_overrides_approximate_home() {
        let index = Arc::new(KVBlockIndex::new(100));
        let hashes = [BlockHash::Bytes(vec![1; 32]), BlockHash::Bytes(vec![2; 32])];
        index.on_blocks_stored(&hashes, None, &[1, 2, 3, 4], 2, "http://w1");
        let policy = SMetricPolicy::with_exact_cache(
            SMetricConfig {
                eviction_interval_secs: 0,
                ..Default::default()
            },
            index,
            Arc::new(MockTokenizer::new()),
            2,
        );
        let workers = workers();
        policy.init_workers(&workers);

        // Plant contradictory approximate state. Exact engine evidence wins.
        policy
            .trees
            .get("default")
            .unwrap()
            .insert("Hello world test token", "http://w0");
        assert_eq!(
            policy.select_worker(&workers, Some("Hello world test token")),
            Some(1)
        );
    }

    #[test]
    fn reservation_is_accounted_and_released_exactly_once() {
        let policy = SMetricPolicy::new();
        let workers = workers();
        policy.init_workers(&workers);
        let id = policy
            .on_request_start("http://w0", Some("abcdef"), None)
            .unwrap();
        assert_eq!(policy.snapshot("http://w0").2, 1);
        assert_eq!(policy.snapshot("http://w0").0, 6.0);
        policy.on_request_complete("http://w0", Some(id), true, Duration::from_secs(1));
        assert_eq!(policy.snapshot("http://w0").2, 0);
        assert_eq!(policy.snapshot("http://w0").0, 0.0);
        policy.on_request_complete("http://w0", Some(id), true, Duration::from_secs(1));
        assert_eq!(policy.snapshot("http://w0").2, 0);
    }

    #[test]
    fn continuation_sticks_until_home_is_overloaded() {
        let config = SMetricConfig {
            eviction_interval_secs: 0,
            overload_factor: 1.1,
            ..Default::default()
        };
        let policy = SMetricPolicy::with_config(config);
        let workers = workers();
        policy.init_workers(&workers);
        let first = policy.select_worker(&workers, Some("abcdefghij")).unwrap();
        let home = workers[first].url().to_string();
        let id = policy
            .on_request_start(&home, Some("abcdefghij"), None)
            .unwrap();
        policy.on_request_complete(&home, Some(id), true, Duration::from_secs(1));
        let continuation = "abcdefghijk";
        assert_eq!(
            policy.select_worker(&workers, Some(continuation)),
            Some(first)
        );

        // Add enough router-observed work to the home to fail the ratio gate.
        let id2 = policy
            .on_request_start(&home, Some("xxxxxxxxxxxxxxxxxxxx"), None)
            .unwrap();
        assert_ne!(
            policy.select_worker(&workers, Some(continuation)),
            Some(first)
        );
        policy.on_request_complete(&home, Some(id2), true, Duration::from_secs(1));
    }

    #[test]
    fn budget_gate_samples_busy_queue_and_aggregates_effective_work() {
        let config = SMetricConfig {
            gate: SMetricGate::BudgetAttention,
            drain_min_samples: 1,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = SMetricPolicy::with_config(config);
        let workers = workers();
        policy.init_workers(&workers);
        let id1 = policy
            .on_request_start("http://w0", Some("abcdefghij"), None)
            .unwrap();
        let id2 = policy
            .on_request_start("http://w0", Some("abcdefghijk"), None)
            .unwrap();
        policy.on_request_first_response("http://w0", Some(id1), Duration::from_secs(1));
        policy.on_request_first_response("http://w0", Some(id2), Duration::from_secs(1));
        assert_eq!(policy.snapshot("http://w0").3, 2);
        // Both requests are sampled, including the second request that was
        // queued behind the first. Their intervals overlap, so the estimator
        // reports a finite effective-work rate rather than filtering either
        // sample by queue state.
        assert!(policy.measured_rate("http://w0") > 0.0);
    }

    #[test]
    fn drain_rate_falls_back_until_minimum_samples() {
        let config = SMetricConfig {
            drain_min_samples: 2,
            drain_tps_fallback: 123.0,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = SMetricPolicy::with_config(config);
        policy
            .state
            .lock()
            .unwrap()
            .insert("http://w0".into(), WorkerState::default());
        assert_eq!(policy.measured_rate("http://w0"), 123.0);
        policy
            .state
            .lock()
            .unwrap()
            .get_mut("http://w0")
            .unwrap()
            .drain_samples
            .push_back((Instant::now(), Instant::now(), 10.0));
        assert_eq!(policy.measured_rate("http://w0"), 123.0);
    }

    #[test]
    fn drain_rate_expires_old_samples_and_uses_window_sum() {
        let config = SMetricConfig {
            drain_min_samples: 2,
            drain_window_secs: 10,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let fallback = config.drain_tps_fallback;
        let policy = SMetricPolicy::with_config(config);
        let now = Instant::now();
        let mut worker = WorkerState::default();
        worker.drain_samples.push_back((
            now - Duration::from_secs(11),
            now - Duration::from_secs(11),
            100.0,
        ));
        worker.drain_samples.push_back((
            now - Duration::from_secs(1),
            now - Duration::from_secs(2),
            30.0,
        ));
        policy
            .state
            .lock()
            .unwrap()
            .insert("http://w0".into(), worker);
        // The expired sample is removed; the remaining sample leaves us below
        // the minimum and therefore uses the configured fallback.
        assert_eq!(policy.measured_rate("http://w0"), fallback);
        policy
            .state
            .lock()
            .unwrap()
            .get_mut("http://w0")
            .unwrap()
            .drain_samples
            .push_back((now, now - Duration::from_secs(1), 20.0));
        // The two live intervals overlap for one second, so their 50 work
        // units are divided by two seconds of union busy time.
        assert_eq!(policy.measured_rate("http://w0"), 25.0);
    }

    #[test]
    fn drain_rate_handles_concurrent_completions_without_queue_filter() {
        let config = SMetricConfig {
            drain_min_samples: 2,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = SMetricPolicy::with_config(config);
        let workers = workers();
        policy.init_workers(&workers);
        let first = policy
            .on_request_start("http://w0", Some("abcdefghij"), None)
            .unwrap();
        let second = policy
            .on_request_start("http://w0", Some("abcdefghijklmnop"), None)
            .unwrap();
        policy.on_request_first_response("http://w0", Some(second), Duration::from_secs(2));
        policy.on_request_first_response("http://w0", Some(first), Duration::from_secs(1));
        assert_eq!(policy.snapshot("http://w0").3, 2);
        assert!(policy.measured_rate("http://w0") > 0.0);
    }

    #[test]
    fn drain_rate_excludes_idle_gaps_from_busy_denominator() {
        let config = SMetricConfig {
            drain_min_samples: 2,
            drain_window_secs: 30,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = SMetricPolicy::with_config(config);
        let now = Instant::now();
        let mut worker = WorkerState::default();
        worker.drain_samples.push_back((
            now - Duration::from_secs(8),
            now - Duration::from_secs(10),
            20.0,
        ));
        worker.drain_samples.push_back((
            now - Duration::from_secs(2),
            now - Duration::from_secs(4),
            40.0,
        ));
        policy
            .state
            .lock()
            .unwrap()
            .insert("http://w0".into(), worker);
        // Two 2-second busy intervals separated by 4 seconds of idle time.
        assert_eq!(policy.measured_rate("http://w0"), 15.0);
    }

    #[test]
    fn drain_rate_orders_completions_and_clips_cross_window_intervals() {
        let config = SMetricConfig {
            drain_min_samples: 2,
            drain_window_secs: 10,
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = SMetricPolicy::with_config(config);
        let now = Instant::now();
        let mut worker = WorkerState::default();
        // Insert completion events out of order. The first interval started
        // before the window and must be clipped at window_start (10 seconds
        // before `now`) before the union duration is computed.
        worker.drain_samples.push_back((
            now - Duration::from_secs(5),
            now - Duration::from_secs(7),
            30.0,
        ));
        worker.drain_samples.push_back((
            now - Duration::from_secs(1),
            now - Duration::from_secs(20),
            100.0,
        ));
        policy
            .state
            .lock()
            .unwrap()
            .insert("http://w0".into(), worker);
        // The clipped union is [now-10, now-1] (9 seconds), so 130 work
        // units yield 130/9 effective work units per second.
        let rate = policy.measured_rate("http://w0");
        // `measured_rate` samples a fresh `Instant`, so allow the few
        // microseconds between constructing the fixtures and that call.
        assert!((rate - (130.0 / 9.0)).abs() < 1e-3, "rate={rate}");
    }

    #[test]
    fn budget_hysteresis_sticks_inside_band_and_releases_beyond_it() {
        assert!(SMetricPolicy::budget_accepts(1.04, 1.0, 0.05, true));
        assert!(!SMetricPolicy::budget_accepts(1.06, 1.0, 0.05, true));
        assert!(!SMetricPolicy::budget_accepts(1.04, 1.0, 0.05, false));
        assert!(SMetricPolicy::budget_accepts(0.99, 1.0, 0.0, false));
    }

    #[test]
    fn session_lease_has_ttl_and_bounded_capacity() {
        let policy = SMetricPolicy::with_config(SMetricConfig {
            drain_window_secs: 10,
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers = workers();
        let now = Instant::now();
        policy.remember_lease("s", "http://w0", true, now);
        assert_eq!(policy.leased_worker("s", &workers, now), Some((0, true)));
        assert_eq!(
            policy.leased_worker("s", &workers, now + Duration::from_secs(11)),
            None
        );

        let mut leases = policy.session_leases.lock().unwrap();
        for i in 0..MAX_SESSION_LEASES {
            leases.insert(
                format!("old-{i}"),
                SessionLease {
                    worker_url: "http://w0".into(),
                    last_stuck: true,
                    last_seen: now,
                },
            );
        }
        drop(leases);
        policy.remember_lease("new", "http://w1", false, now);
        assert_eq!(
            policy.session_leases.lock().unwrap().len(),
            MAX_SESSION_LEASES
        );
    }

    #[test]
    fn no_session_header_keeps_lease_state_empty() {
        let policy = SMetricPolicy::with_config(SMetricConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers = workers();
        assert!(policy.select_worker(&workers, Some("no session")).is_some());
        assert!(policy.session_leases.lock().unwrap().is_empty());
    }

    #[test]
    fn first_response_moves_prefill_work_to_decode_context() {
        let policy = SMetricPolicy::new();
        let workers = workers();
        policy.init_workers(&workers);
        let id = policy
            .on_request_start("http://w0", Some("abcdefghij"), None)
            .unwrap();
        assert!(policy.snapshot("http://w0").1 > 0.0);
        policy.on_request_first_response("http://w0", Some(id), Duration::from_millis(50));
        let during_decode = policy.snapshot("http://w0");
        assert_eq!(during_decode.0, 10.0);
        assert_eq!(during_decode.1, 0.0);
        assert_eq!(during_decode.2, 1);
        policy.on_request_complete("http://w0", Some(id), true, Duration::from_secs(1));
        assert_eq!(policy.snapshot("http://w0").0, 0.0);
    }
}
