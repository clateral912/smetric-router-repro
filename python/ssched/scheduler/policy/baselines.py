"""Baseline routing policies. Old-proxy parity notes on each class. All
randomness comes from ClusterSnapshot.rng (seeded per run)."""

from __future__ import annotations

from ..core import ClusterSnapshot, RequestContext
from .base import Decision, register


def _unprefilled_token_score(req: RequestContext, inst) -> float:
    new_uncached = max(0, req.input_length - inst.cache_hit_tokens)
    pending = max(inst.eff_pending_prefill(), inst.pending_prefill_tokens)
    return pending + new_uncached


@register("load_balance_requests")
class LoadBalanceRequests:
    """Old `load_only`: argmin eff_num_requests, first-index tie-break."""

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        best = min(view.instances, key=lambda i: i.eff_num_requests())
        return Decision(best.idx, "least_requests")


@register("load_balance_tokens")
class LoadBalanceTokens:
    """Old `linear` with alpha=0: argmin eff_ongoing_tokens."""

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        best = min(view.instances, key=lambda i: i.eff_ongoing_tokens())
        return Decision(best.idx, "least_tokens")


@register("load_balance_tokens_blend")
class LoadBalanceTokensBlend:
    """Compatibility alias for the now-blended load_balance_tokens."""

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        best = min(view.instances, key=lambda i: i.eff_ongoing_tokens())
        return Decision(best.idx, "least_tokens_blend")


@register("load_balance_unprefilled_tokens")
class LoadBalanceUnprefilledTokens:
    """Pure prefill-work balance with cache discount on the new request.

    Score = max(real pending-prefill, shadow pending-prefill)
            + input tokens - reusable cache tokens.
    Decode backlog and request count are deliberately excluded.
    """

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        best = min(
            view.instances,
            key=lambda inst: _unprefilled_token_score(req, inst),
        )
        return Decision(best.idx, "least_unprefilled_tokens")


@register("lmetric")
class LMetric:
    """LMetric: f = P_i x BS_i.

    P = max(real_pending, shadow_pending) + max(0, L - c_i)
    BS = max(real_batch_size, shadow_batch_size)

    Set blend_state=false only for a causal ablation of the previous
    fresh-real-else-shadow behavior.
    """

    def __init__(self, blend_state: bool = True):
        self.blend_state = blend_state

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        best_idx, best_score = view.instances[0].idx, float("inf")
        for inst in view.instances:
            new_prefill = max(0, req.input_length - inst.cache_hit_tokens)
            if self.blend_state:
                pending = inst.eff_pending_prefill()
                batch_size = inst.eff_num_requests()
            else:
                rs = inst.real_state
                if rs is None:
                    pending = max(0, inst.pending_prefill_tokens)
                    batch_size = max(0, inst.num_requests)
                else:
                    pending = max(0, rs.pending_prefill_tokens)
                    batch_size = (
                        max(0, rs.num_running) + max(0, rs.num_waiting)
                    )
            score = (pending + new_prefill) * batch_size
            if score < best_score:
                best_score = score
                best_idx = inst.idx
        return Decision(best_idx, "lmetric")


@register("ali")
class Ali:
    """\\company production method (D2 tuned defaults = ali_a50b20k3):

    w_i = 2^(alpha*reuse_i + beta*(mean_load - load_i)),
    reuse = c_i / L, load = eff_num_requests / max; weighted sample
    over the top-K weights. No session pinning.
    """

    def __init__(self, alpha: float = 50.0, beta: float = 20.0,
                 topk: int = 3):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        n = len(view.instances)
        loads = [inst.eff_num_requests() for inst in view.instances]
        max_load = max(loads) or 1.0
        norm = [x / max_load for x in loads]
        mean_l = sum(norm) / n
        inv_in = 1.0 / req.input_length if req.input_length > 0 else 0.0

        weights = [
            2.0 ** (self.alpha * min(1.0, inst.cache_hit_tokens * inv_in)
                    + self.beta * (mean_l - norm[i]))
            for i, inst in enumerate(view.instances)
        ]
        k = min(self.topk, n)
        top = sorted(range(n), key=lambda j: weights[j], reverse=True)[:k]
        if view.rng is None:
            pick = top[0]
        else:
            pick = view.rng.choices(top, weights=[weights[j] for j in top],
                                    k=1)[0]
        return Decision(view.instances[pick].idx, "ali_sample")


@register("sticky")
class Sticky:
    """Hard session affinity: first turn argmin eff_num_requests, then
    always return to the same instance regardless of load."""

    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision:
        if view.affinity_instance is not None:
            return Decision(view.affinity_instance, "sticky")
        best = min(view.instances, key=lambda i: i.eff_num_requests())
        return Decision(best.idx, "sticky_first")
