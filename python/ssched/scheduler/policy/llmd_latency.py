"""llm-d predicted-latency (SLO-aware) routing baseline.

This is the fourth external port and the only TTFT-based one: llm-d's
`latencyPredictor` profile, shipped in the same Helm chart and at the same
frozen revision as the two llm-d arms already in `external.py`
(`08dce7a47a83ffffe83dd03fc522b8bc000647db`).  It is a profile upstream
ships behind `router.latencyPredictor.enabled`, not a scorer combination
assembled here; `config/charts/routerlib/templates/_config.yaml` wires it as

    predicted-latency-producer
      -> prefix-cache-affinity-filter  (strict, 0.99, ttftSource=latencyPredictor)
      -> slo-headroom-tier-filter      (99% positive / 1% negative)
      -> prefix-cache-affinity-filter  (loose, 0.80, same TTFT gate)
      -> latency-scorer                (headroom = SLO - predicted)
      -> weighted-random-picker        (A-Res sampling on the scores)

and the sidecar it talks to is open source as well
(github.com/llm-d/llm-d-latency-predictor, Apache-2.0): an online-trained
XGBoost pair (TTFT and TPOT) whose features are five model-server /
router counters plus two derived terms.  Both halves are ported here; the
sidecar's HTTP boundary is not, because a policy port has no reason to
serialize its own state over localhost.

As shipped, the profile is TTFT-only.  `StreamingMode` defaults to false
in the producer and the chart passes no parameters, so TPOT is neutralized
(`TPOTValid=true, headroom=0`) for every endpoint: the tier filter's TPOT
clause can never reject, and the scorer's range-based weight
re-normalization drives beta to zero because the TPOT range is zero.  The
0.8/0.2 weights in the config are therefore inert at the shipped setting.
`streaming_mode=True` restores them and is offered as a parameter.

Deviations, all deliberate and all listed in
docs/external-routing-baselines.md:

* Three feature columns are dropped because upstream feeds them constants
  on a text-only monolithic fleet: `pod_type_cat` (no P/D role labels),
  `encoder_matched_size`/`encoder_input_size` (multimodal only), and
  `decode_tokens_in_flight` (the Go producer never sends it, so the
  training server zero-fills it).  Constant columns are no-ops for a tree
  model; dropping them is an identity, not a re-specification.
* `prefill_score_bucket` is an ordered 4-level categorical upstream and an
  ordinal float here.  For a tree model over an ORDERED categorical the
  split set is the same.
* One TPOT training sample per request (the stream's mean inter-token
  time) instead of upstream's sampled per-token observations.  Only
  reachable with `streaming_mode=True`.
* The 10% holdout upstream carves out with `hash(str(sorted(sample)))` is
  kept, but keyed on a deterministic digest: Python's str hash is salted
  per process, so the upstream expression is not reproducible across runs
  and a replay harness needs it to be.

Load-signal note.  The blanket `eff_num_requests` substitution the other
six ports use is NOT applied wholesale here, and that is the faithful
choice rather than an exception to it.  Upstream reads two different
sources on purpose: the model server's own `/metrics` for the predictor's
`kv_cache_percentage` / `num_request_waiting` / `num_request_running`
features, and the EPP's in-flight ledger for `prefill_tokens_in_flight`
and `DispatchedRequestCount`.  Collapsing the first three into one blended
counter would destroy the queue-gated model, whose entire structure is the
`num_request_waiting == 0` split.  So the three feature columns read the
engine feed (`InstanceView.real_state`, ssched's equivalent of the
scraped model-server metrics) and the two ledger quantities read
`eff_pending_prefill` / `eff_num_requests` -- the same split upstream
makes, mapped onto the two state sources ssched has.
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from collections import deque

import numpy as np

from ..core import ClusterSnapshot, InstanceView, RequestContext
from ...trace.schema import HASH_BLOCK_TOKENS
from .base import Decision, register

# Upstream training-server constants.  The first block is the Helm chart's
# `latencyPredictor.trainingServer.config` (which overrides the server's own
# defaults); the second is the server's frozen XGBoost parameter set.
_RETRAIN_INTERVAL_S = 10.0        # LATENCY_RETRAINING_INTERVAL_SEC
_MIN_SAMPLES_FOR_RETRAIN = 100    # LATENCY_MIN_SAMPLES_FOR_RETRAIN
_MIN_SAMPLES_FRESH = 10           # LATENCY_MIN_SAMPLES_FOR_RETRAIN_FRESH
_BUCKET_SIZE = 500                # LATENCY_MAX_TRAINING_DATA_SIZE_PER_BUCKET
_TEST_TRAIN_RATIO = 0.1           # LATENCY_TEST_TRAIN_RATIO
_CACHE_BUCKETS = 20
_PREFIX_BUCKETS = 4
_DEFAULT_PREDICTION_MS = 10.0     # the single-sample prior model's target

# The training server's frozen sklearn-API kwargs, kept verbatim so the
# port can be diffed against upstream, and translated below to the native
# booster names.  The native API is used because it needs no scikit-learn
# in the router's environment; the fitted trees are the same.
_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.2,
    "reg_alpha": 0.01,
    "reg_lambda": 0.1,
    "tree_method": "hist",
    # Upstream runs n_jobs=-1 in its own pod; here the trainer shares the
    # box with the router's event loop, so it gets one thread.
    "n_jobs": 1,
    "random_state": 42,
    # LATENCY_OBJECTIVE_TYPE=mean in the chart, i.e. not the quantile head.
    "objective": "reg:squarederror",
}
_XGB_NATIVE_PARAMS = {
    "max_depth": _XGB_PARAMS["max_depth"],
    "eta": _XGB_PARAMS["learning_rate"],
    "subsample": _XGB_PARAMS["subsample"],
    "colsample_bytree": _XGB_PARAMS["colsample_bytree"],
    "min_child_weight": _XGB_PARAMS["min_child_weight"],
    "gamma": _XGB_PARAMS["gamma"],
    "alpha": _XGB_PARAMS["reg_alpha"],
    "lambda": _XGB_PARAMS["reg_lambda"],
    "tree_method": _XGB_PARAMS["tree_method"],
    "nthread": _XGB_PARAMS["n_jobs"],
    "seed": _XGB_PARAMS["random_state"],
    "objective": _XGB_PARAMS["objective"],
}
_XGB_ROUNDS = _XGB_PARAMS["n_estimators"]

# Feature order follows the training server's `feature_cols`, minus the
# constant columns named in the module docstring.
_TTFT_COLUMNS = (
    "is_queued",
    "kv_cache_percentage",
    "input_token_length",
    "num_request_waiting",
    "num_request_running",
    "prefill_tokens_in_flight",
    "prefix_cache_score",
    "effective_input_tokens",
    "prefill_score_bucket",
)
_TPOT_COLUMNS = (
    "is_queued",
    "kv_cache_percentage",
    "input_token_length",
    "num_request_waiting",
    "num_request_running",
    "prefill_tokens_in_flight",
    "num_tokens_generated",
)
# The queue-gated "noqueue" sub-model drops the two columns that are
# constant within its own regime.
_QUEUE_COLUMNS = ("is_queued", "num_request_waiting")


def _queue_bucket(num_waiting: float) -> int:
    if num_waiting <= 0:
        return 0
    if num_waiting <= 2:
        return 1
    if num_waiting <= 5:
        return 2
    if num_waiting <= 10:
        return 3
    return 4


def _cache_bucket(pct: float) -> int:
    pct = min(1.0, max(0.0, pct))
    return min(int(pct * _CACHE_BUCKETS), _CACHE_BUCKETS - 1)


def _prefix_bucket(score: float) -> int:
    score = min(1.0, max(0.0, score))
    return min(int(score * _PREFIX_BUCKETS), _PREFIX_BUCKETS - 1)


def _is_holdout(sample: dict) -> bool:
    """Upstream's 10% test split, made reproducible across processes."""
    key = repr(sorted((k, round(v, 6)) for k, v in sample.items()))
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100 < int(_TEST_TRAIN_RATIO * 100)


class _ConstantModel:
    """The prior model upstream trains on one synthetic row before any
    traffic has been observed.  A single-leaf tree returns its target for
    every input, so the cold-start behaviour is an exactly constant
    prediction -- which makes every endpoint's headroom identical, which
    makes the scorer's ranges zero, which makes the weighted-random picker
    uniform.  llm-d starts out routing at random, on purpose."""

    def __init__(self, value: float = _DEFAULT_PREDICTION_MS):
        self.value = float(value)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return np.full(len(rows), self.value, dtype=np.float64)


class _Booster:
    """Thin wrapper so prediction avoids the sklearn/pandas call path."""

    def __init__(self, booster):
        self._booster = booster

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._booster.inplace_predict(rows), dtype=np.float64)


class _QueueGatedModel:
    """Upstream's `QueueGatedModel`: a pair of sub-models selected per row
    by `num_request_waiting`, the noqueue half trained without the two
    columns that are constant inside its regime."""

    def __init__(self, noqueue, queued, columns: tuple[str, ...]):
        self.noqueue = noqueue
        self.queued = queued
        self._keep = [i for i, name in enumerate(columns)
                      if name not in _QUEUE_COLUMNS]
        self._waiting_col = columns.index("num_request_waiting")

    def predict(self, rows: np.ndarray) -> np.ndarray:
        out = np.empty(len(rows), dtype=np.float64)
        waiting = rows[:, self._waiting_col]
        idle = waiting <= 0
        if idle.any():
            out[idle] = self.noqueue.predict(
                np.ascontiguousarray(rows[idle][:, self._keep]))
        busy = ~idle
        if busy.any():
            out[busy] = self.queued.predict(np.ascontiguousarray(rows[busy]))
        return out


class LatencyPredictor:
    """Port of the llm-d latency-predictor sidecar's training/prediction
    loop, run in-process.

    Training happens on a daemon thread so a retrain can never stall the
    router's event loop -- upstream gets the same isolation for free by
    running the trainer in a separate pod.  Prediction is a numpy matrix
    into `inplace_predict`, which is the fastest path xgboost exposes and
    keeps the per-request cost off the critical section.
    """

    def __init__(
        self,
        retrain_interval_s: float = _RETRAIN_INTERVAL_S,
        # Upstream reads two thresholds but only ever uses one of them on a
        # deployment like ours.  `load_models` sets
        # MIN_SAMPLES_FOR_RETRAIN = MIN_SAMPLES_FOR_RETRAIN_FRESH when it
        # starts with no persisted joblib on disk, and never restores the
        # higher value -- so a fresh deployment, which every ssched run is,
        # retrains at 10 samples for the whole run, not at the chart's 100.
        min_samples_for_retrain: int = _MIN_SAMPLES_FRESH,
        bucket_size: int = _BUCKET_SIZE,
        train_tpot: bool = False,
        seed: int = 0,
    ):
        self.retrain_interval_s = float(retrain_interval_s)
        self.min_samples_for_retrain = int(min_samples_for_retrain)
        self.bucket_size = int(bucket_size)
        self.train_tpot = bool(train_tpot)
        self._rng = random.Random(seed)

        self._lock = threading.Lock()
        self._ttft_buckets: dict[tuple[int, int, int], deque] = {}
        self._tpot_buckets: dict[tuple[int, int, int], deque] = {}
        # Published models: plain attribute assignment, so route() never
        # takes a lock to read one and never sees a half-built pair.
        self.ttft_model = _ConstantModel()
        self.tpot_model = _ConstantModel()
        self._fresh = True          # telemetry: has anything been fitted yet
        self._trainings = 0
        self._samples_seen = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- sample intake -------------------------------------------------

    def add_sample(self, sample: dict) -> None:
        """One completed request, in the training server's schema."""
        ttft_ok = sample.get("actual_ttft_ms", 0.0) > 0
        tpot_ok = sample.get("actual_tpot_ms", 0.0) > 0
        if not ttft_ok and not tpot_ok:
            return
        if _is_holdout(sample):
            return  # upstream's held-out evaluation split
        key = (
            _queue_bucket(sample["num_request_waiting"]),
            _cache_bucket(sample["kv_cache_percentage"]),
            _prefix_bucket(sample["prefix_cache_score"]),
        )
        with self._lock:
            self._samples_seen += 1
            if ttft_ok:
                self._ttft_buckets.setdefault(
                    key, deque(maxlen=self.bucket_size)).append(sample)
            if tpot_ok and self.train_tpot:
                self._tpot_buckets.setdefault(
                    key, deque(maxlen=self.bucket_size)).append(sample)
        self._ensure_thread()

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="llmd-latency-predictor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.retrain_interval_s):
            try:
                self.train()
            except Exception:  # a failed retrain keeps the previous models
                continue

    def stop(self) -> None:
        self._stop.set()

    # -- training ------------------------------------------------------

    def train(self) -> bool:
        threshold = self.min_samples_for_retrain
        with self._lock:
            ttft = [s for bucket in self._ttft_buckets.values()
                    for s in bucket]
            tpot = [s for bucket in self._tpot_buckets.values()
                    for s in bucket]
        if len(ttft) + len(tpot) < threshold:
            return False

        trained = False
        if len(ttft) >= threshold:
            model = self._fit(ttft, _TTFT_COLUMNS, "actual_ttft_ms")
            if model is not None:
                self.ttft_model = model
                trained = True
        if self.train_tpot and len(tpot) >= threshold:
            model = self._fit(tpot, _TPOT_COLUMNS, "actual_tpot_ms")
            if model is not None:
                self.tpot_model = model
                trained = True
        if trained:
            self._fresh = False
            self._trainings += 1
        return trained

    def _fit(self, samples: list[dict], columns: tuple[str, ...],
             target: str):
        import xgboost as xgb

        rows = np.array(
            [[float(s[name]) for name in columns] for s in samples],
            dtype=np.float32)
        y = np.array([float(s[target]) for s in samples], dtype=np.float32)
        waiting = rows[:, columns.index("num_request_waiting")]
        keep = [i for i, name in enumerate(columns)
                if name not in _QUEUE_COLUMNS]

        idle = waiting <= 0
        halves = []
        for mask, cols in ((idle, keep), (~idle, list(range(len(columns))))):
            if not mask.any():
                # Nothing has been observed in this queue regime yet, so
                # it keeps the prior rather than borrowing the other
                # half's fit -- the two regimes do not share a feature
                # space, which is the point of the gate.
                halves.append(_ConstantModel())
                continue
            dtrain = xgb.DMatrix(
                np.ascontiguousarray(rows[mask][:, cols]), label=y[mask])
            booster = xgb.train(
                _XGB_NATIVE_PARAMS, dtrain, num_boost_round=_XGB_ROUNDS)
            halves.append(_Booster(booster))
        return _QueueGatedModel(halves[0], halves[1], columns)

    # -- prediction ----------------------------------------------------

    def predict_ttft_ms(self, rows: np.ndarray) -> np.ndarray:
        return self.ttft_model.predict(rows)

    def predict_tpot_ms(self, rows: np.ndarray) -> np.ndarray:
        return self.tpot_model.predict(rows)

    def stats(self) -> dict:
        return {
            "llmd_predictor_samples": self._samples_seen,
            "llmd_predictor_trainings": self._trainings,
            "llmd_predictor_fresh": self._fresh,
        }


def _endpoint_ttft(predicted_ms: float | None) -> float:
    """`prefixcacheaffinity.endpointTTFT`: an endpoint with no prediction
    contributes no signal to the load gate, which upstream spells as
    MaxFloat64 -- never the fastest, so it can neither be the cheap
    alternative that breaks stickiness nor make a sticky set look fast."""
    return math.inf if predicted_ms is None else predicted_ms


def _prefix_cache_score(req: RequestContext, inst: InstanceView) -> float:
    """Matched blocks over indexed blocks -- the same ratio the
    `llmd_prefix_load` port feeds its prefix scorer."""
    indexed = len(req.block_hashes) * HASH_BLOCK_TOKENS
    if indexed <= 0:
        return 0.0
    return min(1.0, max(0, inst.cache_hit_tokens) / indexed)


@register("llmd_predicted_latency")
class LlmdPredictedLatency:
    """llm-d's shipped predicted-latency profile, end to end.

    Parameters carry the chart's values where the chart sets one and the
    plugin's own default otherwise; the docstring above says which is
    which.  `slo_base_s` / `slo_input_tokens_per_s` / `slo_tpot_s` stand in
    for the `x-slo-ttft-ms` and `x-slo-tpot-ms` request headers upstream
    reads -- policy-side constants configured per arm, never read from the
    scoring convention (see ssched.scoring.slo_convention).
    """

    def __init__(
        self,
        # The SLO this arm declares per request.  llm-d takes it from
        # headers; ssched's replayer sends none, so the same convention
        # arrives as arm parameters.
        slo_base_s: float = 1.0,
        slo_input_tokens_per_s: float = 16000.0,
        slo_tpot_s: float = 0.020,
        # latency-scorer (plugin defaults; the chart passes no parameters).
        ttft_weight: float = 0.8,
        tpot_weight: float = 0.2,
        headroom_strategy: str = "least",
        # slo-headroom-tier-filter (plugin default).
        epsilon_explore_neg: float = 0.01,
        # prefix-cache-affinity-filter, both instances (chart values).
        strict_affinity_threshold: float = 0.99,
        loose_affinity_threshold: float = 0.80,
        max_ttft_penalty_ms: float = 5000.0,
        exploration_probability: float = 0.0,
        # predicted-latency-producer (plugin defaults).  streaming_mode
        # false is what the chart ships: TPOT is neutralized and the
        # profile is TTFT-only.
        streaming_mode: bool = False,
        slo_buffer_factor: float = 1.0,
        # latency-scorer's composite fallback, used only when NO endpoint
        # has a prediction (plugin defaults, all one).
        composite_kv_weight: float = 1.0,
        composite_queue_weight: float = 1.0,
        composite_prefix_weight: float = 1.0,
        # Predictor (chart's trainingServer config).
        retrain_interval_s: float = _RETRAIN_INTERVAL_S,
        min_samples_for_retrain: int = _MIN_SAMPLES_FRESH,
        seed: int = 0,
    ):
        # The predictor is this arm's whole point, so a missing xgboost
        # must fail here rather than silently leave the cold-start prior
        # in place for the whole run.  It is an extra, not a core
        # dependency: no other policy needs it.
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "llmd_predicted_latency needs xgboost "
                "(uv pip install --group latency-predictor)") from exc
        if headroom_strategy not in ("least", "most"):
            raise ValueError(
                f"headroom_strategy must be 'least' or 'most', "
                f"got {headroom_strategy!r}")
        self.slo_base_s = float(slo_base_s)
        self.slo_input_tokens_per_s = float(slo_input_tokens_per_s)
        self.slo_tpot_s = float(slo_tpot_s)
        self.ttft_weight = float(ttft_weight)
        self.tpot_weight = float(tpot_weight)
        self.headroom_strategy = headroom_strategy
        self.epsilon_explore_neg = float(epsilon_explore_neg)
        self.strict_affinity_threshold = float(strict_affinity_threshold)
        self.loose_affinity_threshold = float(loose_affinity_threshold)
        self.max_ttft_penalty_ms = float(max_ttft_penalty_ms)
        self.exploration_probability = float(exploration_probability)
        self.streaming_mode = bool(streaming_mode)
        self.slo_buffer_factor = float(slo_buffer_factor)
        self.composite_kv_weight = float(composite_kv_weight)
        self.composite_queue_weight = float(composite_queue_weight)
        self.composite_prefix_weight = float(composite_prefix_weight)
        self.predictor = LatencyPredictor(
            retrain_interval_s=retrain_interval_s,
            min_samples_for_retrain=min_samples_for_retrain,
            train_tpot=self.streaming_mode,
            seed=seed,
        )
        # Dispatch-time features, held until the request completes and can
        # be labelled.  Upstream keeps the same map with a 5 minute TTL.
        self._pending: dict[str, dict] = {}
        self._pending_order: deque[tuple[float, str]] = deque()
        self._context_ttl_s = 300.0

    # -- routing -------------------------------------------------------

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        instances = view.instances
        n = len(instances)
        rng = view.rng

        prefix = [_prefix_cache_score(req, inst) for inst in instances]
        raw = [self._features(req, inst, prefix[i])
               for i, inst in enumerate(instances)]
        # An endpoint whose model-server metrics upstream does not have is
        # an endpoint it cannot build a prediction request for, and every
        # downstream stage has an explicit rule for that case: the tier
        # filter puts it in the NEGATIVE tier, the affinity gate reads its
        # TTFT as MaxFloat64 so it can never be the fast alternative that
        # breaks stickiness, and the scorer drops to a composite score when
        # NO endpoint has a prediction.  A stale engine feed is that
        # condition here, so it produces no prediction rather than a
        # fabricated one from zeroed features.
        have = [i for i, row in enumerate(raw) if row is not None]

        ttft_ms: list[float | None] = [None] * n
        tpot_ms: list[float | None] = [None] * n
        if have:
            rows = np.array(
                [[raw[i][name] for name in _TTFT_COLUMNS] for i in have],
                dtype=np.float32)
            for i, value in zip(have, self.predictor.predict_ttft_ms(rows)):
                ttft_ms[i] = float(value)
            if self.streaming_mode:
                rows = np.array(
                    [[raw[i][name] for name in _TPOT_COLUMNS] for i in have],
                    dtype=np.float32)
                for i, value in zip(have,
                                    self.predictor.predict_tpot_ms(rows)):
                    tpot_ms[i] = float(value)

        slo_ttft_ms = 1000.0 * (
            self.slo_base_s + req.input_length / self.slo_input_tokens_per_s)
        ttft_headroom = [None if v is None else slo_ttft_ms - v
                         for v in ttft_ms]
        if self.streaming_mode:
            buffered = 1000.0 * self.slo_tpot_s * self.slo_buffer_factor
            tpot_headroom = [None if v is None else buffered - v
                             for v in tpot_ms]
        else:
            # The producer's non-streaming branch neutralizes TPOT for
            # every endpoint it DID predict; one it did not stays absent.
            tpot_headroom = [None if raw[i] is None else 0.0
                             for i in range(n)]

        dispatched = [inst.eff_num_requests() for inst in instances]
        kv_free = [
            1.0 - min(1.0, max(0.0, float(inst.real_state.gpu_kv_used_frac)))
            if inst.real_state is not None else 1.0
            for inst in instances
        ]

        candidates = list(range(n))
        candidates = self._affinity_filter(
            candidates, prefix, ttft_ms, self.strict_affinity_threshold, rng)
        gate = len(candidates)
        candidates = self._tier_filter(
            candidates, ttft_headroom, tpot_headroom, rng)
        tier = len(candidates)
        candidates = self._affinity_filter(
            candidates, prefix, ttft_ms, self.loose_affinity_threshold, rng)

        scores = self._latency_scores(
            candidates, ttft_headroom, tpot_headroom, dispatched,
            prefix, kv_free)
        winner = self._weighted_random_pick(candidates, scores, rng)

        # Hold the winner's dispatch-time features until the router
        # reports the realized latencies (upstream's PreRequest hook).
        self.note_dispatch(req.request_id, raw[winner])

        # `latency-slo-admitter` is NOT ported: a Decision cannot express
        # a reject, so admission control would need a router-side
        # mechanism, and adding one would change what the goodput
        # denominator counts for this arm alone.  Its verdict is logged
        # instead, so the question can later be answered from a landed
        # cell rather than by rerunning it.
        admissible = (ttft_headroom[winner] is not None
                      and tpot_headroom[winner] is not None
                      and ttft_headroom[winner] >= 0
                      and tpot_headroom[winner] >= 0)

        return Decision(
            instances[winner].idx,
            "llmd_predicted_latency",
            extra={
                # Headroom is slo - predicted, so it is not logged
                # separately: two 32-wide arrays per decision is already
                # the largest telemetry any port in this tree writes.
                "llmd_predicted_ttft_ms": [
                    None if v is None else round(v, 3) for v in ttft_ms],
                "llmd_ttft_slo_ms": round(slo_ttft_ms, 3),
                "llmd_prefix_scores": [round(v, 6) for v in prefix],
                "llmd_candidates_after_strict": gate,
                "llmd_candidates_after_tier": tier,
                "llmd_candidates": [instances[i].idx for i in candidates],
                "llmd_latency_scores": [round(scores[i], 6)
                                        for i in candidates],
                "llmd_slo_admissible": admissible,
                **({"llmd_predicted_tpot_ms": [
                    None if v is None else round(v, 4) for v in tpot_ms]}
                   if self.streaming_mode else {}),
                **self.predictor.stats(),
            },
        )

    def _features(self, req: RequestContext, inst: InstanceView,
                  prefix_score: float) -> dict | None:
        """The prediction request's feature row, or None when it cannot be
        built.

        Three of the columns are the model server's own scraped metrics.
        Without a fresh engine feed ssched does not have them, and zeroing
        them is not a neutral substitution: `num_request_waiting = 0` sends
        the row to the queue-gated model's NOQUEUE half, which was fitted
        only on genuinely idle instances, so a stale endpoint would come
        back predicted fast and act as the cheap alternative that breaks
        prefix stickiness.  Upstream's answer to a missing prediction is a
        defined state, not a fabricated number, so this returns None and
        each stage applies upstream's own rule.
        """
        state = inst.real_state
        if state is None:
            return None
        kv = min(1.0, max(0.0, float(state.gpu_kv_used_frac)))
        waiting = float(max(0, state.num_waiting))
        running = float(max(0, state.num_running))
        input_len = float(req.input_length)
        return {
            "is_queued": 1.0 if waiting > 0 else 0.0,
            "kv_cache_percentage": kv,
            "input_token_length": input_len,
            "num_request_waiting": waiting,
            "num_request_running": running,
            "prefill_tokens_in_flight": float(inst.eff_pending_prefill()),
            "prefix_cache_score": prefix_score,
            "effective_input_tokens": (1.0 - prefix_score) * input_len,
            "prefill_score_bucket": float(_prefix_bucket(prefix_score)),
            # Prediction time asks for the first generated token.
            "num_tokens_generated": 1.0,
        }

    # -- pipeline stages -----------------------------------------------

    def _affinity_filter(self, candidates: list[int], prefix: list[float],
                         ttft_ms, threshold: float,
                         rng: random.Random | None) -> list[int]:
        if len(candidates) <= 1 or threshold <= 0:
            return candidates
        if (self.exploration_probability > 0 and rng is not None
                and rng.random() < self.exploration_probability):
            return candidates

        sticky = [i for i in candidates if prefix[i] >= threshold]
        if not sticky:
            return candidates
        non_sticky = [i for i in candidates if prefix[i] < threshold]
        if self.max_ttft_penalty_ms > 0 and non_sticky:
            # `endpointTTFT` returns MaxFloat64 for an endpoint with no
            # prediction, so it never lowers either side of the gate.
            best_sticky = min(_endpoint_ttft(ttft_ms[i]) for i in sticky)
            best_other = min(_endpoint_ttft(ttft_ms[i]) for i in non_sticky)
            if best_sticky - best_other > self.max_ttft_penalty_ms:
                return candidates
        return sticky

    def _tier_filter(self, candidates: list[int], ttft_headroom: list[float],
                     tpot_headroom: list[float],
                     rng: random.Random | None) -> list[int]:
        if len(candidates) <= 1:
            return candidates
        # Upstream separates "predicted, and within SLO" from everything
        # else: an endpoint with no prediction cannot be confirmed to meet
        # the SLO, so it is appended to the negative tier.
        positive = [
            i for i in candidates
            if ttft_headroom[i] is not None and tpot_headroom[i] is not None
            and ttft_headroom[i] >= 0 and tpot_headroom[i] >= 0
        ]
        in_positive = set(positive)
        negative = [i for i in candidates if i not in in_positive]
        if not positive:
            return negative or candidates
        if not negative:
            return positive
        if rng is not None and rng.random() < self.epsilon_explore_neg:
            return negative
        return positive

    def _latency_scores(self, candidates: list[int],
                        ttft_headroom: list[float | None],
                        tpot_headroom: list[float | None],
                        dispatched: list[float],
                        prefix: list[float],
                        kv_free: list[float]) -> dict[int, float]:
        scores = {i: 0.0 for i in candidates}
        if all(ttft_headroom[i] is None for i in candidates):
            # `hasPredictions == false`: the sidecar is down or timed out,
            # and upstream scores on KV headroom, relative queue depth and
            # prefix instead of on latency at all.
            return self._composite_scores(candidates, dispatched, prefix,
                                          kv_free)

        # Upstream's split is `info != nil && (ttft < 0 || tpot < 0)` -> the
        # negative bucket, everything else positive.  An endpoint with no
        # prediction therefore lands in the POSITIVE bucket carrying Go's
        # zero-value headroom, which under the "least" strategy is the
        # smallest deficit and scores highest.  That is a real upstream
        # quirk, and it is reachable only through the tier filter's 1%
        # epsilon, since the filter otherwise removes those endpoints
        # first -- the scorer's own comment says it assumes homogeneous
        # input.  Reproduced rather than repaired: a port that quietly
        # fixed it would not be the policy being compared against.
        positive = [
            i for i in candidates
            if not (ttft_headroom[i] is not None
                    and tpot_headroom[i] is not None
                    and (ttft_headroom[i] < 0 or tpot_headroom[i] < 0))
        ]
        if positive:
            self._score_bucket(positive, ttft_headroom, tpot_headroom,
                               scores, force_least=False)
            return scores

        negative = list(candidates)
        idle = [i for i in negative if dispatched[i] == 0]
        if idle:
            self._score_bucket(idle, ttft_headroom, tpot_headroom, scores,
                               force_least=True)
            return scores

        # Deficit bucketing: least severe non-empty bucket wins outright.
        # Upstream files a nil-info endpoint under bothNeg here.
        tpot_only, ttft_only, both = [], [], []
        for i in negative:
            if ttft_headroom[i] is None or tpot_headroom[i] is None:
                both.append(i)
                continue
            ttft_neg = ttft_headroom[i] < 0
            tpot_neg = tpot_headroom[i] < 0
            if ttft_neg and tpot_neg:
                both.append(i)
            elif ttft_neg:
                ttft_only.append(i)
            elif tpot_neg:
                tpot_only.append(i)
            else:
                both.append(i)
        for bucket in (tpot_only, ttft_only, both):
            if bucket:
                self._score_bucket(bucket, ttft_headroom, tpot_headroom,
                                   scores, force_least=True)
                return scores
        self._score_bucket(negative, ttft_headroom, tpot_headroom, scores,
                           force_least=True)
        return scores

    def _score_bucket(self, bucket: list[int], ttft_headroom: list[float],
                      tpot_headroom: list[float], scores: dict[int, float],
                      force_least: bool) -> None:
        eps = 1e-9
        w_max = 100
        total = self.ttft_weight + self.tpot_weight
        if total <= 0:
            alpha, beta = 1.0, 0.0
        else:
            alpha, beta = self.ttft_weight / total, self.tpot_weight / total

        # A nil-info endpoint reads its headroom as Go's zero value.
        ttft_abs = [abs(ttft_headroom[i] or 0.0) for i in bucket]
        tpot_abs = [abs(tpot_headroom[i] or 0.0) for i in bucket]
        min_ttft, max_ttft = min(ttft_abs), max(ttft_abs)
        min_tpot, max_tpot = min(tpot_abs), max(tpot_abs)
        ttft_range = max_ttft - min_ttft
        tpot_range = max_tpot - min_tpot
        # Range-based weight re-normalization: a dimension with no spread
        # would otherwise compress every score to one value.
        if ttft_range <= eps and tpot_range > eps:
            alpha, beta = 0.0, 1.0
        elif tpot_range <= eps and ttft_range > eps:
            alpha, beta = 1.0, 0.0

        strategy = "least" if force_least else self.headroom_strategy
        for pos, i in enumerate(bucket):
            n_ttft = ((ttft_abs[pos] - min_ttft) / ttft_range
                      if ttft_range > eps else 0.5)
            n_tpot = ((tpot_abs[pos] - min_tpot) / tpot_range
                      if tpot_range > eps else 0.5)
            combined = alpha * n_ttft + beta * n_tpot
            if strategy == "most":
                w = int(combined * w_max) + 1
            else:
                w = int((1.0 - combined) * w_max) + 1
            scores[i] = w / w_max

    def _composite_scores(self, candidates: list[int],
                          dispatched: list[float], prefix: list[float],
                          kv_free: list[float]) -> dict[int, float]:
        """latency-scorer's no-prediction fallback: free KV, relative queue
        depth and prefix score at the plugin's default equal weights.

        Upstream reads `WaitingQueueSize` for the queue term; with no
        engine feed anywhere that number does not exist, so the queue term
        reads the router's own dispatch ledger -- the shared load signal
        every other port in this tree substitutes.  The KV term keeps the
        Go zero value (free) for an endpoint with no feed, which is what
        upstream's metrics object would carry."""
        w_kv = self.composite_kv_weight
        w_q = self.composite_queue_weight
        w_p = self.composite_prefix_weight
        total = w_kv + w_q + w_p
        if total <= 0:
            w_kv, w_q, w_p, total = 1.0, 0.0, 0.0, 1.0
        w_kv, w_q, w_p = w_kv / total, w_q / total, w_p / total

        max_q = max(dispatched[i] for i in candidates)
        scores = {}
        for i in candidates:
            rel_queue = 1.0 if max_q <= 0 else (max_q - dispatched[i]) / max_q
            composite = w_kv * kv_free[i] + w_q * rel_queue + w_p * prefix[i]
            # Go's math.Round is half-away-from-zero; Python's round()
            # is half-to-even, which differs on exact .5 weights.
            scores[i] = math.floor(100.0 * composite + 0.5) / 100.0
        return scores

    def _weighted_random_pick(self, candidates: list[int],
                              scores: dict[int, float],
                              rng: random.Random | None) -> int:
        """A-Res weighted reservoir sampling, k=1: key = U^(1/w)."""
        positive = [i for i in candidates if scores[i] > 0]
        if not positive:
            # Upstream delegates an all-zero score set to the random picker.
            if rng is None:
                return candidates[0]
            return rng.choice(candidates)
        if rng is None:
            # Deterministic fallback, matching the house convention that a
            # seedless snapshot never draws: highest score, first on ties.
            return max(positive, key=lambda i: (scores[i], -i))
        best_key, best = -1.0, positive[0]
        for i in positive:
            u = rng.random() or 1e-10
            key = math.pow(u, 1.0 / scores[i])
            if key > best_key:
                best_key, best = key, i
        return best

    # -- training feedback ---------------------------------------------

    def note_dispatch(self, request_id: str, features: dict) -> None:
        now = time.monotonic()
        self._pending[request_id] = features
        self._pending_order.append((now, request_id))
        cutoff = now - self._context_ttl_s
        while self._pending_order and self._pending_order[0][0] < cutoff:
            _, stale = self._pending_order.popleft()
            self._pending.pop(stale, None)

    def note_completion(self, request_id: str, ttft_s: float,
                        decode_s: float, output_tokens: int) -> None:
        """One labelled training row, from the router's completion path."""
        features = self._pending.pop(request_id, None)
        if features is None:
            return
        sample = dict(features)
        sample["actual_ttft_ms"] = max(0.0, float(ttft_s) * 1000.0)
        # Upstream samples per-token observations; one mean per stream is
        # the same label at coarser granularity (streaming mode only).
        steps = max(0, int(output_tokens) - 1)
        sample["actual_tpot_ms"] = (
            float(decode_s) * 1000.0 / steps if steps > 0 and decode_s > 0
            else 0.0)
        sample["num_tokens_generated"] = float(max(1, int(output_tokens)))
        self.predictor.add_sample(sample)
