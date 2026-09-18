"""Faithful policy-level ports of external routing baselines.

Source revisions frozen for this port:

* llm-d inference scheduler 08dce7a47a83ffffe83dd03fc522b8bc000647db
  (github.com/llm-d/llm-d-inference-scheduler; per-arm upstream file paths
  are listed in docs/external-routing-baselines.md)
* AIBrix e397f9866e19866abd943b93244f3dd03b5873cb

The ports reuse ssched's single-router request/cache/state substrate.  The
AIBrix Preble port deliberately keeps its upstream speculative radix tree and
fixed V100/Mistral-7B cost constants separate from ssched's completion-driven
shadow cache.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..core import ClusterSnapshot, InstanceView, RequestContext
from ...trace.schema import HASH_BLOCK_TOKENS
from .base import Decision, register


def _random_max_index(scores: list[float], view: ClusterSnapshot) -> int:
    """llm-d max-score picker: uniform random order before stable argmax."""
    best = max(scores)
    tied = [i for i, score in enumerate(scores) if score == best]
    if view.rng is None:
        return tied[0]
    return view.rng.choice(tied)


@register("aibrix_power_of_two")
class AibrixPowerOfTwo:
    """AIBrix power-of-two choices over router-owned in-flight counts.

    AIBrix stores these counters in Redis for multiple gateway replicas.
    ssched conservatively blends its engine Redis feed with the local
    reservation ledger so admission lag cannot understate active requests.
    """

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        instances = view.instances
        if len(instances) == 1:
            target = instances[0]
            sampled = [target]
        else:
            rng = view.rng
            if rng is None:
                first, second = 0, 1
            else:
                first = rng.randrange(len(instances))
                second = rng.randrange(len(instances))
                while second == first:
                    second = rng.randrange(len(instances))
            sampled = [instances[first], instances[second]]
            # Upstream selects candidate 1 on an equal count.
            target = min(sampled, key=lambda inst: inst.eff_num_requests())

        return Decision(
            target.idx,
            "aibrix_power_of_two",
            extra={
                "sampled_instance_indices": [inst.idx for inst in sampled],
                "sampled_request_counts": [
                    inst.eff_num_requests() for inst in sampled
                ],
            },
        )


def _llmd_queue_scores(instances: list[InstanceView]) -> list[float]:
    # There is no shadow waiting/running split, so use the conservative
    # blended active-request count rather than allowing a fresh-but-lagging
    # waiting metric to make the endpoint look idle.
    queues = [inst.eff_num_requests() for inst in instances]
    lo, hi = min(queues), max(queues)
    if lo == hi:
        return [1.0] * len(instances)
    return [(hi - queue) / (hi - lo) for queue in queues]


def _llmd_kv_scores(instances: list[InstanceView]) -> list[float]:
    scores = []
    for inst in instances:
        used = (inst.real_state.gpu_kv_used_frac
                if inst.real_state is not None else 0.0)
        scores.append(1.0 - min(1.0, max(0.0, used)))
    return scores


@register("llmd_prefix_load")
class LlmdPrefixLoad:
    """llm-d precise-prefix sample profile, with its frozen 2:1:1 weights.

    score = 2 * prefix-match ratio + 1 * free-KV + 1 * inverse queue.
    The prefix score uses ssched's common cache view so this remains a policy
    comparison rather than simultaneously changing the cache-state producer.
    """

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        # The upstream scorer divides matched blocks by total indexed blocks.
        # ssched's common view indexes only complete 512-token trace blocks, so
        # the prompt tail is likewise excluded from this ratio.
        indexed_tokens = len(req.block_hashes) * HASH_BLOCK_TOKENS
        inv_indexed = 1.0 / indexed_tokens if indexed_tokens > 0 else 0.0
        prefix = [
            min(1.0, max(0, inst.cache_hit_tokens) * inv_indexed)
            for inst in view.instances
        ]
        queue = _llmd_queue_scores(view.instances)
        kv = _llmd_kv_scores(view.instances)
        total = [2.0 * prefix[i] + kv[i] + queue[i]
                 for i in range(len(view.instances))]
        winner = _random_max_index(total, view)

        return Decision(
            view.instances[winner].idx,
            "llmd_prefix_load",
            extra={
                "llmd_prefix_scores": [round(score, 6) for score in prefix],
                "llmd_queue_scores": [round(score, 6) for score in queue],
                "llmd_kv_scores": [round(score, 6) for score in kv],
                "llmd_total_scores": [round(score, 6) for score in total],
            },
        )


def _llmd_active_request_scores(instances: list[InstanceView]) -> list[float]:
    """Upstream active-request scorer at its defaults (idle 0, busy max 1.0)."""
    counts = [inst.eff_num_requests() for inst in instances]
    max_count = max(counts)
    return [
        1.0 if count == 0 else (max_count - count) / max_count
        for count in counts
    ]


@register("llmd_session_load")
class LlmdSessionLoad:
    """llm-d payload-agnostic profile: session affinity + active requests.

    This is upstream's shipped `payload-agnostic` well-known profile
    (config/charts/routerlib/templates/_config.yaml), not a combination we
    assembled: both scorers carry weight one there.  The load term is part of
    that profile because session affinity alone scores every non-owner zero.
    Session affinity is therefore a soft preference: a maximally loaded
    session owner can tie an idle alternative.
    """

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        active = _llmd_active_request_scores(view.instances)
        affinity = [
            1.0 if inst.idx == view.affinity_instance else 0.0
            for inst in view.instances
        ]
        total = [active[i] + affinity[i]
                 for i in range(len(view.instances))]
        winner = _random_max_index(total, view)

        return Decision(
            view.instances[winner].idx,
            "llmd_session_load",
            extra={
                "llmd_active_request_scores": [
                    round(score, 6) for score in active
                ],
                "llmd_session_scores": affinity,
                "llmd_total_scores": [round(score, 6) for score in total],
            },
        )


@dataclass(eq=False)
class _PrebleNode:
    key: tuple[int, ...]
    parent: _PrebleNode | None
    context_length: int
    depth: int
    last_access: float
    children: dict[int, _PrebleNode] = field(default_factory=dict)
    # ssched is single-model, so this is AIBrix's modelToPods inner set.
    pods: set[int] = field(default_factory=set)

    @property
    def num_tokens(self) -> int:
        return len(self.key)


class _PrebleRadixTree:
    """Small Python translation of AIBrix's compressed LPRadixCache."""

    def __init__(self, now: float):
        self.root = _PrebleNode((), None, 0, 0, now)
        self.nodes: set[_PrebleNode] = {self.root}

    @staticmethod
    def _match_len(left: tuple[int, ...], right: tuple[int, ...]) -> int:
        size = min(len(left), len(right))
        i = 0
        while i < size and left[i] == right[i]:
            i += 1
        return i

    def add_prefix(
        self, tokens: tuple[int, ...], now: float,
    ) -> tuple[_PrebleNode, int]:
        current = self.root
        offset = 0

        while True:
            current.last_access = now
            remaining = tokens[offset:]
            if not remaining:
                return current, len(tokens)

            child = current.children.get(remaining[0])
            if child is None:
                node = _PrebleNode(
                    remaining,
                    current,
                    current.context_length + len(remaining),
                    current.depth + 1,
                    now,
                )
                current.children[remaining[0]] = node
                self.nodes.add(node)
                return node, offset

            matched = self._match_len(child.key, remaining)
            if matched == len(child.key):
                child.last_access = now
                offset += matched
                if offset == len(tokens):
                    return child, len(tokens)
                current = child
                continue

            # Split the existing compressed edge.  AIBrix copies the cached
            # pod mapping onto the new shared-prefix node.
            split = _PrebleNode(
                child.key[:matched],
                current,
                current.context_length + matched,
                current.depth + 1,
                now,
                pods=set(child.pods),
            )
            current.children[remaining[0]] = split
            child.key = child.key[matched:]
            child.parent = split
            split.children[child.key[0]] = child
            self.nodes.add(split)

            offset += matched
            if offset == len(tokens):
                return split, len(tokens)
            current = split

    def evict(self, now: float, ttl_s: float) -> set[_PrebleNode]:
        stale = {
            node for node in self.nodes
            if node is not self.root and now - node.last_access > ttl_s
        }
        roots = [node for node in stale if node.parent not in stale]
        evicted: set[_PrebleNode] = set()

        for root in roots:
            stack = [root]
            while stack:
                node = stack.pop()
                if node in evicted:
                    continue
                evicted.add(node)
                stack.extend(node.children.values())

            if root.parent is not None and root.key:
                root.parent.children.pop(root.key[0], None)

            # Match AIBrix's ancestor cleanup for the evicted mapping.
            ancestor = root.parent
            while ancestor is not None:
                ancestor.pods.difference_update(root.pods)
                ancestor = ancestor.parent

        self.nodes.difference_update(evicted)
        return evicted


@dataclass(frozen=True)
class _PrebleHistoryEntry:
    timestamp: float
    node: _PrebleNode
    leaf: _PrebleNode


class _PrebleHistogram:
    def __init__(self, window_s: float):
        self.window_s = window_s
        self.histogram: dict[_PrebleNode, int] = defaultdict(int)
        self.node_to_count: dict[_PrebleNode, int] = defaultdict(int)
        self.hit_tokens: dict[_PrebleNode, int] = defaultdict(int)
        self.prompt_tokens: dict[_PrebleNode, int] = defaultdict(int)
        self.decoding_size: dict[_PrebleNode, int] = {}
        self.timestamps: deque[_PrebleHistoryEntry] = deque()

    def update(
        self,
        now: float,
        node: _PrebleNode,
        leaf: _PrebleNode,
        decoding_length: int,
    ) -> None:
        self.timestamps.append(_PrebleHistoryEntry(now, node, leaf))
        self.histogram[node] += leaf.context_length
        self.node_to_count[node] += 1
        self.hit_tokens[node] += leaf.context_length - leaf.num_tokens
        self.prompt_tokens[node] += leaf.context_length
        self.decoding_size[node] = decoding_length

    def remove_evicted(self, evicted: set[_PrebleNode]) -> None:
        if not evicted:
            return
        self.timestamps = deque(
            entry for entry in self.timestamps if entry.node not in evicted
        )
        for node in evicted:
            self._delete(node)

    def remove_old(self, now: float) -> None:
        cutoff = now - self.window_s
        while self.timestamps and self.timestamps[0].timestamp <= cutoff:
            entry = self.timestamps.popleft()
            node, leaf = entry.node, entry.leaf
            self.histogram[node] -= leaf.context_length
            self.node_to_count[node] -= 1
            self.hit_tokens[node] -= leaf.context_length - leaf.num_tokens
            self.prompt_tokens[node] -= leaf.context_length
            if self.histogram[node] <= 0:
                self._delete(node)

    def _delete(self, node: _PrebleNode) -> None:
        self.histogram.pop(node, None)
        self.node_to_count.pop(node, None)
        self.hit_tokens.pop(node, None)
        self.prompt_tokens.pop(node, None)
        self.decoding_size.pop(node, None)

    def pod_load(self, instance_idx: int) -> int:
        return sum(
            count for node, count in self.node_to_count.items()
            if instance_idx in node.pods
        )

    @staticmethod
    def _v100_linear_time(num_tokens: int) -> float:
        if num_tokens >= 384:
            return (0.27106428 * num_tokens + 10.52444263) / 1000.0
        if num_tokens >= 192:
            return (-295 + 3.125 * num_tokens
                    - 6.4e-3 * num_tokens ** 2) / 1000.0
        return 55.0 / 1000.0

    @staticmethod
    def _v100_attention_time(context_length: int,
                             unique_kv: int) -> float:
        if context_length <= 1024:
            return 0.80 / 1000.0
        forward = 4.65e-4 * context_length + 0.398
        if unique_kv <= 1024 and unique_kv <= 32 * 256 * 2048:
            forward /= 2
        return forward / 1000.0

    @staticmethod
    def _v100_attention_quadratic(num_tokens: int) -> float:
        if num_tokens < 4096:
            return 0.0
        return (-18.425 + 9.65e-3 * num_tokens
                + 5.4e-6 * num_tokens ** 2) / 1000.0

    def _node_cost(self, node: _PrebleNode) -> float:
        miss_rate = 1.0
        if self.prompt_tokens[node] > 0:
            miss_rate -= self.hit_tokens[node] / self.prompt_tokens[node]

        prefill_time = (
            self._v100_linear_time(node.num_tokens)
            + self._v100_attention_time(
                node.context_length, node.num_tokens)
            + self._v100_attention_quadratic(node.num_tokens)
        ) / 0.9
        # AIBrix GetModelToPodCount() returns the number of models, not the
        # number of replicas.  ssched is single-model, so the divisor is one.
        prefill_cost = (
            miss_rate * self.node_to_count[node] * prefill_time
        )
        decode_cost = self.decoding_size[node] * 0.15
        return prefill_cost + decode_cost

    def allocation_costs(self) -> dict[int, float]:
        costs: dict[int, float] = defaultdict(float)
        for node in self.histogram:
            cost = self._node_cost(node)
            for instance_idx in node.pods:
                costs[instance_idx] += cost
        return costs


@dataclass(eq=False)
class _AibrixHashBlock:
    pods: set[int] = field(default_factory=set)
    last_access: float = 0.0


class _AibrixPrefixHashTable:
    """Chained-hash block table over AIBrix's LRUStore semantics.

    Upstream lrustore: Get never refreshes recency or the TTL clock; only
    Put (AddPrefix after every route) moves an entry to the head and stamps
    last_access.  Insertion order therefore equals last-access order, so
    TTL eviction can pop expired entries from the cold end exactly.
    """

    def __init__(self, block_size: int, capacity: int, ttl_s: float):
        self.block_size = block_size
        self.capacity = capacity
        self.ttl_s = ttl_s
        self._blocks: dict[int, _AibrixHashBlock] = {}

    def prefix_hashes(self, tokens: tuple[int, ...]) -> list[int]:
        # Upstream chains xxhash(parent_hash || block); incomplete final
        # blocks are never hashed.  Python tuple hashing over int token
        # ids is the deterministic stand-in (PYTHONHASHSEED does not
        # randomize int/tuple-of-int hashes).
        hashes: list[int] = []
        parent = 0
        for i in range(0, len(tokens) - self.block_size + 1,
                       self.block_size):
            parent = hash((parent, tokens[i:i + self.block_size]))
            hashes.append(parent)
        return hashes

    def evict_expired(self, now: float) -> None:
        for h in list(self._blocks):
            if now - self._blocks[h].last_access > self.ttl_s:
                del self._blocks[h]
            else:
                break

    def match(self, hashes: list[int],
              candidates: set[int]) -> dict[int, int]:
        """map[instance]%prefixmatch, upstream seqSearchPrefix.

        Walks blocks in order, destructively intersecting the candidate
        set; a pod that falls out keeps the percent from its last matched
        depth.  Stops at the first missing block or empty intersection.
        """
        matched: dict[int, int] = {}
        remaining = set(candidates)
        total = len(hashes)
        for i, h in enumerate(hashes):
            block = self._blocks.get(h)
            if block is None or not block.pods:
                break
            percent = (i + 1) * 100 // total
            hit = False
            for idx in list(remaining):
                if idx in block.pods:
                    matched[idx] = percent
                    hit = True
                else:
                    remaining.discard(idx)
            if not hit:
                break
        return matched

    def add_prefix(self, hashes: list[int], target: int,
                   now: float) -> None:
        for h in hashes:
            block = self._blocks.pop(h, None)
            if block is None:
                block = _AibrixHashBlock()
            block.pods.add(target)
            block.last_access = now
            self._blocks[h] = block
            if len(self._blocks) > self.capacity:
                del self._blocks[next(iter(self._blocks))]


@register("aibrix_prefix_cache")
class AibrixPrefixCache:
    """Faithful port of AIBrix ``prefix-cache`` default behavior.

    Upstream fixed defaults preserved: 4-unit hash blocks, a 200k-block
    LRU with 20-minute idle TTL, request-count imbalance threshold 8
    (matching then restricted to the least-loaded pods), and the
    mean + 1 stddev running-request guard on prefix winners; unlike
    preble there is no minimum match-ratio gate.  Prompt token ids stand
    in for AIBrix's character tokenization of the raw message (same
    substitution as AibrixPreble), so one block is 4 tokens rather than
    upstream's 4 characters.
    """

    block_size = 4
    capacity = 200_000
    ttl_s = 20 * 60.0
    load_imbalance_threshold = 8
    stddev_factor = 1.0

    def __init__(self):
        self._table = _AibrixPrefixHashTable(
            self.block_size, self.capacity, self.ttl_s)

    @staticmethod
    def _running_requests(inst: InstanceView) -> int:
        # Shadow has no running/waiting split; conservatively use blended
        # active requests, matching the other AIBrix ports.
        return int(inst.eff_num_requests())

    def _pick_within_stddev(self, matched: dict[int, int],
                            counts: dict[int, int],
                            view: ClusterSnapshot) -> int | None:
        # Mean/stddev are computed over ALL ready pods (sample stddev,
        # n-1), even when the imbalance filter narrowed the candidates.
        values = [float(count) for count in counts.values()]
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            stddev = var ** 0.5
        else:
            stddev = 0.0
        order = list(matched)
        # Upstream shuffles before a stable sort so full ties are random.
        if view.rng is not None:
            view.rng.shuffle(order)
        order.sort(key=lambda idx: (-matched[idx], counts[idx]))
        for idx in order:
            if counts[idx] <= mean + self.stddev_factor * stddev:
                return idx
        return None

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        now = time.time()
        self._table.evict_expired(now)

        counts = {
            inst.idx: self._running_requests(inst)
            for inst in view.instances
        }
        least = min(counts.values())
        load_imbalanced = max(counts.values()) - least \
            > self.load_imbalance_threshold
        if load_imbalanced:
            candidates = {idx for idx, count in counts.items()
                          if count == least}
        else:
            candidates = set(counts)

        hashes = self._table.prefix_hashes(req.token_ids)
        matched = self._table.match(hashes, candidates)

        target = None
        selection = "prefix_match"
        if matched:
            target = self._pick_within_stddev(matched, counts, view)
        if target is None:
            # Upstream falls back over ALL ready pods, not the
            # imbalance-filtered list, with a random min-count tie-break.
            tied = [idx for idx, count in counts.items() if count == least]
            if view.rng is None or len(tied) == 1:
                target = tied[0]
            else:
                target = tied[view.rng.randrange(len(tied))]
            selection = ("prefix_match_skipped" if matched
                         else "least_request_fallback")

        # AIBrix's PostRouteUpdate maps every block of the prompt to the
        # selected pod before backend success is known, with no rollback.
        self._table.add_prefix(hashes, target, now)

        return Decision(
            target,
            f"aibrix_prefix_cache_{selection}",
            extra={
                "aibrix_pc_selection": selection,
                "aibrix_pc_match_percent": matched.get(target, 0),
                "aibrix_pc_load_imbalanced": load_imbalanced,
                "aibrix_pc_matched_pods": dict(matched),
            },
        )


@register("aibrix_preble")
class AibrixPreble:
    """Faithful port of AIBrix ``prefix-cache-preble`` default behavior.

    The upstream fixed defaults are preserved: V100/Mistral-7B cost curves,
    decode length 45, a three-minute load window, five-minute radix TTL,
    strict 50% prefix gate, and request-count imbalance threshold 8.  Prompt
    token ids replace AIBrix's cl100k re-tokenization because ssched traces do
    not carry the original text.
    """

    decoding_length = 45
    history_window_s = 3 * 60.0
    radix_ttl_s = 5 * 60.0
    prefix_threshold = 0.5
    load_imbalance_threshold = 8

    def __init__(self):
        now = time.time()
        self._tree = _PrebleRadixTree(now)
        self._history = _PrebleHistogram(self.history_window_s)

    @staticmethod
    def _running_requests(inst: InstanceView) -> int:
        # Shadow has no running/waiting split; conservatively use blended
        # active requests for the load-imbalance guard.
        return int(inst.eff_num_requests())

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        now = time.time()
        evicted = self._tree.evict(now, self.radix_ttl_s)
        self._history.remove_evicted(evicted)
        self._history.remove_old(now)

        node, matched_tokens = self._tree.add_prefix(req.token_ids, now)
        candidates = list(view.instances)
        counts = [self._running_requests(inst) for inst in candidates]
        load_imbalanced = bool(counts) and (
            max(counts) - min(counts) > self.load_imbalance_threshold
        )
        if load_imbalanced:
            least = min(counts)
            candidates = [inst for inst, count in zip(candidates, counts)
                          if count == least]

        match_ratio = (
            matched_tokens / len(req.token_ids) if req.token_ids else 0.0
        )
        target = None
        selected_value = None
        branch = "cost"

        if match_ratio > self.prefix_threshold:
            current = node
            while current is not None:
                matched = [inst for inst in candidates
                           if inst.idx in current.pods]
                if matched:
                    target = min(
                        matched,
                        key=lambda inst: self._history.pod_load(inst.idx),
                    )
                    selected_value = self._history.pod_load(target.idx)
                    branch = "prefix"
                    break
                current = current.parent

        if target is None:
            costs = self._history.allocation_costs()
            target = min(candidates, key=lambda inst: costs[inst.idx])
            selected_value = costs[target.idx]

        # AIBrix updates the radix placement and histogram immediately after
        # route selection, before backend success is known, with no rollback.
        current = node
        while current is not None:
            current.pods.add(target.idx)
            current = current.parent
        self._history.update(
            now, node, node, decoding_length=self.decoding_length)

        return Decision(
            target.idx,
            f"aibrix_preble_{branch}",
            extra={
                "aibrix_preble_branch": branch,
                "aibrix_preble_match_ratio": round(match_ratio, 6),
                "aibrix_preble_matched_tokens": matched_tokens,
                "aibrix_preble_load_imbalanced": load_imbalanced,
                "aibrix_preble_candidate_indices": [
                    inst.idx for inst in candidates
                ],
                "aibrix_preble_selected_value": round(
                    float(selected_value), 6),
            },
        )


@register("dynamo_prefill_router")
class DynamoPrefillRouter:
    """Port of NVIDIA Dynamo's prefill-pool KV router cost.

    Upstream (``lib/kv-router/src/scheduling/``) scores every candidate with

        raw_prefill_tokens   = active_prefill_tokens + uncached + cached
                             = active_prefill_tokens + max(isl, cached)
        overlap_credit_blocks = overlap_score_credit * decay * device_blocks
                              + host_cache_hit_weight * host_blocks
        adjusted             = max(0, raw_prefill_blocks - overlap_credit_blocks)
        logit                = prefill_load_scale * adjusted
                             + decode_cost_blocks
                             + decode_active_request_weight * active_requests

    and takes the argmin (``router_temperature`` defaults to 0, so the
    softmax path is off).  Note the shape of the first line: ``selector/mod.rs``
    computes ``uncached = isl - min(cached, isl)`` and then adds ``cached``
    straight back, so the raw term is cache-INDEPENDENT.  All cache credit is
    applied once, later, in block units and with per-tier weights.  That is
    the substantive difference from a plain ``uncached + queued`` score: a
    host-tier hit is credited at 0.75, so it still costs a quarter of a fresh
    block, whereas netting cached tokens out of the work term treats it as
    free.

    Tier mapping for this cluster: ``device`` = the instance's own GPU prefix
    cache, ``host`` = the mooncake/LMCache store (CPU DRAM, shared, hence the
    same value on every instance).  The deployment has no disk tier, so
    ``disk_cache_hit_weight`` is not carried.

    Defaults are upstream's ``KvRouterConfig::default()``; every one is
    overridable so the decay term can be ablated (upstream ships it off).
    """

    def __init__(
        self,
        block_size: int = 16,               # vLLM KV block, not the 512-token
                                            # trace hash block
        overlap_score_credit: float = 1.0,
        overlap_score_credit_decay: float = 0.0,   # upstream default: off
        prefill_load_scale: float = 1.0,
        decode_active_request_weight: float = 0.0,
        host_cache_hit_weight: float = 0.75,
        track_prefill_tokens: bool = True,
    ):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        for name, value in (
            ("overlap_score_credit", overlap_score_credit),
            ("overlap_score_credit_decay", overlap_score_credit_decay),
            ("prefill_load_scale", prefill_load_scale),
            ("decode_active_request_weight", decode_active_request_weight),
            ("host_cache_hit_weight", host_cache_hit_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.block_size = block_size
        self.overlap_score_credit = overlap_score_credit
        self.overlap_score_credit_decay = overlap_score_credit_decay
        self.prefill_load_scale = prefill_load_scale
        self.decode_active_request_weight = decode_active_request_weight
        self.host_cache_hit_weight = host_cache_hit_weight
        self.track_prefill_tokens = track_prefill_tokens

    def _tiers(self, req: RequestContext, inst: InstanceView) -> tuple[int, int]:
        """(device_tokens, host_tokens) as a disjoint split of the prefix hit.

        The store shadow reports a GLOBAL prefix hit, so the host tier is only
        the part it can serve beyond what this instance already holds on
        device -- otherwise a block resident on both tiers is credited twice.
        """
        device = max(0, min(int(inst.cache_hit_tokens), req.input_length))
        store_total = max(0, min(int(inst.store_hit_tokens), req.input_length))
        return device, max(0, store_total - device)

    def route(
        self,
        req: RequestContext,
        view: ClusterSnapshot,
        candidate_positions: list[int] | None = None,
    ) -> Decision:
        """Route across all instances or an SMetric-provided safe subset.

        ``candidate_positions`` uses positions in ``view.instances`` rather
        than instance IDs, matching SMetric's contract-safe mask.  The
        standalone Dynamo baseline leaves it unset and therefore retains its
        complete upstream candidate set.
        """
        block = float(self.block_size)
        request_blocks = max(1.0, req.input_length / block)

        positions = (list(range(len(view.instances)))
                     if candidate_positions is None
                     else list(candidate_positions))
        if not positions:
            raise ValueError("candidate_positions must not be empty")
        if any(pos < 0 or pos >= len(view.instances) for pos in positions):
            raise ValueError("candidate position is outside the cluster view")

        rows = []
        for pos in positions:
            inst = view.instances[pos]
            device_tokens, host_tokens = self._tiers(req, inst)
            cached = device_tokens + host_tokens
            active_prefill = max(0.0, inst.eff_pending_prefill())
            if self.track_prefill_tokens:
                # uncached + cached, per upstream's saturating add-back.
                uncached = req.input_length - min(cached, req.input_length)
                raw_tokens = active_prefill + uncached + cached
            else:
                raw_tokens = 0.0
            rows.append({
                "idx": inst.idx,
                "device_blocks": device_tokens / block,
                "host_blocks": host_tokens / block,
                "active_prefill": active_prefill,
                "raw_prefill_blocks": raw_tokens / block,
                "decode_cost_blocks": max(0.0, inst.eff_ongoing_decode()) / block,
                "active_requests": max(0.0, inst.eff_num_requests()),
            })

        # Backlog above the least-loaded eligible worker, normalised by this
        # request's size; softly trades locality for prefill balance.
        min_active = min(r["active_prefill"] for r in rows) if rows else 0.0

        keys = []
        for row in rows:
            if self.track_prefill_tokens and self.overlap_score_credit_decay > 0.0:
                excess_blocks = max(
                    0.0, row["active_prefill"] - min_active) / block
                decay = 1.0 / (
                    1.0 + self.overlap_score_credit_decay
                    * (excess_blocks / request_blocks)
                )
            else:
                decay = 1.0
            credit = (
                self.overlap_score_credit * decay * row["device_blocks"]
                + self.host_cache_hit_weight * row["host_blocks"]
            )
            adjusted = max(0.0, row["raw_prefill_blocks"] - credit)
            logit = (
                self.prefill_load_scale * adjusted
                + row["decode_cost_blocks"]
                + self.decode_active_request_weight * row["active_requests"]
            )
            row["decay"] = decay
            row["credit_blocks"] = credit
            row["logit"] = logit
            keys.append((logit, row["idx"]))

        best = min(keys)
        # Upstream is a plain argmin at temperature 0, which would pin index 0
        # whenever every candidate is identical (notably a cold cluster).  Use
        # the harness's shared round-robin on EXACT ties only, as the other
        # ports do; strict ordering is unchanged.
        tied = [key for key in keys if key[0] == best[0]]
        used_rr = False
        if len(tied) > 1 and view.next_rr is not None:
            target = tied[view.next_rr() % len(tied)][1]
            used_rr = True
        else:
            target = best[1]

        chosen = next(r for r in rows if r["idx"] == target)
        reason = "dynamo_prefill_router" + ("_rr" if used_rr else "")
        return Decision(
            int(target),
            reason,
            extra={
                "dynamo_schema": 1,
                "dynamo_block_size": self.block_size,
                "dynamo_overlap_score_credit": self.overlap_score_credit,
                "dynamo_overlap_score_credit_decay": (
                    self.overlap_score_credit_decay),
                "dynamo_host_cache_hit_weight": self.host_cache_hit_weight,
                "dynamo_prefill_load_scale": self.prefill_load_scale,
                "dynamo_selected_logit": round(float(chosen["logit"]), 6),
                "dynamo_selected_device_blocks": round(
                    float(chosen["device_blocks"]), 4),
                "dynamo_selected_host_blocks": round(
                    float(chosen["host_blocks"]), 4),
                "dynamo_selected_credit_blocks": round(
                    float(chosen["credit_blocks"]), 4),
                "dynamo_selected_decay": round(float(chosen["decay"]), 4),
                "dynamo_candidate_logit": [
                    round(float(r["logit"]), 6) for r in rows],
                "dynamo_candidate_active_prefill_tokens": [
                    float(r["active_prefill"]) for r in rows],
                "dynamo_candidate_decay": [
                    round(float(r["decay"]), 4) for r in rows],
                "dynamo_candidate_credit_blocks": [
                    round(float(r["credit_blocks"]), 4) for r in rows],
            },
        )
