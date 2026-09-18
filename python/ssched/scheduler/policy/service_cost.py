from __future__ import annotations

import bisect
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass

from ..core import prefill_work_units


@dataclass
class _Pending:
    idx: int
    input_length: int
    gpu_hit: int
    store_hit: int
    age_bin: int
    loss_probability: float
    hit_seconds: float
    miss_seconds: float
    started_at: float


class ServiceCostModel:
    """One service coordinate for resident, fetched and recomputed prefixes.

    The queue is an explicitly serial service approximation. Only its head
    ages: subtracting wall time independently from all admissions would drain
    queued requests before they run. Completion observations retire entries.
    """

    def __init__(self, drain_tps: float, store_tps: float, l_eq: float,
                 fixed_s: float = .28):
        self.drain_tps = drain_tps
        self.store_tps = store_tps
        self.l_eq = l_eq
        self.pending: dict[str, _Pending] = {}
        self.queues: dict[int, OrderedDict[str, None]] = defaultdict(OrderedDict)
        self.losses: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=256))
        self.loss_severity: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=256))
        self.fixed_s = fixed_s

    def _work_s(self, n: int, length: int) -> float:
        return prefill_work_units(n, length, self.l_eq) / self.drain_tps

    def estimate(self, req, inst) -> tuple[float, float, float, int]:
        gpu = max(0, min(req.input_length, inst.cache_hit_tokens))
        store = max(gpu, min(req.input_length, inst.store_hit_tokens))
        age_bin = bisect.bisect_right((10., 30., 60., 120.),
                                     inst.store_prefix_age_s)
        outcomes = self.losses[age_bin]
        # A weak, explicit Beta(1,1) prior; warmup replaces it with actual
        # recovered-prefix observations, not hit / full-prompt labels.
        loss = (1 + sum(outcomes)) / (2 + len(outcomes)) if store > gpu else 0.
        hit_s = (self.fixed_s + self._work_s(req.input_length - store,
                                            req.input_length)
                 + (store - gpu) / self.store_tps)
        severity = self.loss_severity[age_bin]
        lost_fraction = (1 + sum(severity)) / (1 + len(severity))
        hit_work = self._work_s(req.input_length - store, req.input_length)
        cold_work = self._work_s(req.input_length - gpu, req.input_length)
        miss_s = hit_s + lost_fraction * (cold_work - hit_work)
        return hit_s, miss_s, loss, age_bin

    @staticmethod
    def _survival_remaining(mean: float, age: float) -> tuple[float, float]:
        # Uniform [.5*mean, 1.5*mean] allows service variation. Conditioning
        # on still unfinished shifts a slow fetch toward the miss branch.
        lo, hi = .5 * mean, 1.5 * mean
        if age < lo:
            return 1., mean - age
        if age < hi:
            return (hi - age) / (hi - lo), (hi - age) / 2
        return 0., 0.

    def queue_seconds(self, inst, now: float) -> float:
        ids = self.queues[inst.idx]
        total = 0.
        for position, rid in enumerate(ids):
            p = self.pending[rid]
            if position:
                total += ((1 - p.loss_probability) * p.hit_seconds
                          + p.loss_probability * p.miss_seconds)
                continue
            age = max(0., now - p.started_at)
            sh, rh = self._survival_remaining(p.hit_seconds, age)
            sm, rm = self._survival_remaining(p.miss_seconds, age)
            ph, pm = (1 - p.loss_probability) * sh, p.loss_probability * sm
            if ph + pm > 0:
                total += (ph * rh + pm * rm) / (ph + pm)
            else:
                # A late response must not vanish. Use observed remaining
                # engine work, retaining a positive in-flight path cost.
                remaining = (inst.real_state.pending_prefill_tokens
                             if inst.real_state is not None else
                             p.input_length - p.gpu_hit)
                total += max(self.fixed_s, self._work_s(
                    min(p.input_length, max(0, remaining)), p.input_length))
        if not ids and inst.real_state is not None:
            remaining = max(0, inst.real_state.pending_prefill_tokens)
            if remaining:
                total = self._work_s(remaining, remaining)
        return total

    def reserve(self, req, inst, estimate, now: float) -> None:
        self.finish(req.request_id, None, now=now)
        hit_s, miss_s, loss, age_bin = estimate
        self.pending[req.request_id] = _Pending(
            inst.idx, req.input_length, inst.cache_hit_tokens,
            inst.store_hit_tokens, age_bin, loss, hit_s, miss_s, now)
        self.queues[inst.idx][req.request_id] = None

    def finish(self, request_id: str, cached_tokens: int | None,
               *, now: float | None = None) -> None:
        p = self.pending.pop(request_id, None)
        if p is None:
            return
        ids = self.queues[p.idx]
        was_head = next(iter(ids), None) == request_id
        ids.pop(request_id, None)
        if was_head and ids:
            self.pending[next(iter(ids))].started_at = (
                time.monotonic() if now is None else now)
        if cached_tokens is not None and p.store_hit > p.gpu_hit:
            hit_work = self._work_s(p.input_length - p.store_hit, p.input_length)
            miss_work = self._work_s(p.input_length - p.gpu_hit, p.input_length)
            actual_work = self._work_s(
                max(0, p.input_length - cached_tokens), p.input_length)
            fraction = max(0., min(1., (actual_work - hit_work)
                                    / max(1e-9, miss_work - hit_work)))
            lost = cached_tokens + 512 < p.store_hit
            self.losses[p.age_bin].append(float(lost))
            if lost:
                self.loss_severity[p.age_bin].append(fraction)
