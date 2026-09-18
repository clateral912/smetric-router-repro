"""Model-derived prefill constants, so no policy carries a 30B number.

Two quantities in the routing path are physical rather than dimensionless,
and both were literals calibrated on Qwen3-30B-A3B:

* ``L_EQ`` -- the context at which attention FLOPs equal linear FLOPs for
  one token, which converts a prefill's ``n * (L - n/2)`` attention moment
  into cold-token-equivalents.
* ``drain_tps`` -- how many of those cold-token-equivalents one instance
  retires per second.

Carrying them as constants makes the policy non-transferable: on
Qwen3-235B-A22B the true values are 13568 and ~13100, i.e. 2.0x and 0.56x
the 30B ones.  Both follow from the model's shape, so this module derives
them instead.

``L_EQ`` needs no fitting at all.  Per layer and per token, the linear
part costs ``2 * P`` FLOPs for ``P`` active parameters, and attending to
one token of context costs ``4 * n_q_heads * head_dim`` (QK^T and AV, two
FLOPs per MAC).  Layer count cancels::

    L_EQ = 2 * P_layer / (4 * n_q_heads * head_dim)

For Qwen3-30B-A3B that is 6912.0 and for Qwen3-235B-A22B 13568.0, both
exact.  The shipped 30B value is 6923 -- a TTFT-residual fit, 0.16% above
the shape value -- and the registry keeps it so already-landed cells stay
comparable; see ``KNOWN_SHAPES``.

``drain_tps`` needs one measured number, but only one for the whole
cluster rather than one per model.  A cold-token-equivalent costs
``2 * P_layer * n_layers`` FLOPs, so::

    drain_tps = achieved_flops_per_gpu * tp_size / flops_per_work_unit

Against the two measured rates (30B-PO/PD 23248 work/s on TP1, 235B-PD
13100 work/s on TP4) the implied per-GPU figures are 126.4 and 136.9
TFLOPS -- 8.3% apart across models differing 7.7x in FLOPs per token and
4x in tensor-parallel width.  Their geometric mean, 131.5 TFLOPS, predicts
30B at +4.1% and 235B at -3.9%, inside the spread of the measurement
itself.  That is one hardware constant instead of one constant per model,
and it is falsifiable: a third model that misses by more than ~10% means
the FLOP model is missing a term (memory-bound layers, quantisation, MoE
routing overhead), not that the constant needs retuning.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

# Achieved prefill FLOPS per GPU, from the two calibration points above.
# Hardware- and precision-specific (H20, FP8 weights), NOT model-specific:
# that is the whole point.  Override via SchedulerConfig.gpu_prefill_tflops
# when the fleet changes.
DEFAULT_GPU_PREFILL_FLOPS = 131.5e12


@dataclass(frozen=True)
class ModelShape:
    """The activated-parameter shape of one transformer layer."""

    name: str
    n_layers: int
    hidden: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    # Activated FFN parameters per layer: for a dense model the MLP, for a
    # mixture of experts only the experts a token actually routes to.
    ffn_params_per_layer: int
    # Set only where a measurement beats the shape calculation; see
    # KNOWN_SHAPES for why 30B carries one.
    l_eq_override: float | None = None

    @property
    def attn_params_per_layer(self) -> int:
        """Q, K, V and O projections."""
        q = self.hidden * self.n_q_heads * self.head_dim
        kv = 2 * self.hidden * self.n_kv_heads * self.head_dim
        o = self.n_q_heads * self.head_dim * self.hidden
        return q + kv + o

    @property
    def active_params_per_layer(self) -> int:
        return self.attn_params_per_layer + self.ffn_params_per_layer

    @property
    def attention_break_even_context(self) -> float:
        """``L_EQ``: context where attention FLOPs equal linear FLOPs.

        Independent of layer count -- both sides scale with it.
        """
        if self.l_eq_override is not None:
            return float(self.l_eq_override)
        return self.shape_break_even_context

    @property
    def shape_break_even_context(self) -> float:
        """``L_EQ`` from the shape alone, ignoring any override."""
        return (2.0 * self.active_params_per_layer
                / (4.0 * self.n_q_heads * self.head_dim))

    @property
    def flops_per_work_unit(self) -> float:
        """FLOPs to prefill one cold-token-equivalent, whole model."""
        return 2.0 * self.active_params_per_layer * self.n_layers

    def prefill_work_tps(
        self,
        tp_size: int = 1,
        gpu_flops: float = DEFAULT_GPU_PREFILL_FLOPS,
    ) -> float:
        """Cold-token-equivalents one instance retires per second."""
        if tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        return gpu_flops * float(tp_size) / self.flops_per_work_unit


def from_hf_config(cfg: dict[str, Any], name: str = "") -> ModelShape:
    """Build a shape from a HuggingFace ``config.json`` dict.

    Handles dense and Qwen3-style MoE.  A model whose MoE layers are
    interleaved with dense ones (``mlp_only_layers`` /
    ``decoder_sparse_step``) is averaged over layers, because the load
    model only ever needs the per-token mean.
    """
    hidden = int(cfg["hidden_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads", n_q))
    head_dim = int(cfg.get("head_dim") or (hidden // n_q))

    dense_ffn = 3 * hidden * int(cfg.get("intermediate_size") or 0)
    moe_int = int(cfg.get("moe_intermediate_size") or 0)
    top_k = int(cfg.get("num_experts_per_tok") or 0)
    moe_ffn = 3 * hidden * moe_int * top_k if (moe_int and top_k) else 0

    if moe_ffn:
        step = int(cfg.get("decoder_sparse_step") or 1)
        dense_only = set(cfg.get("mlp_only_layers") or ())
        moe_layers = sum(
            1 for i in range(n_layers)
            if i not in dense_only and step > 0 and (i + 1) % step == 0)
        ffn = int(round((moe_ffn * moe_layers
                         + dense_ffn * (n_layers - moe_layers)) / n_layers))
    else:
        ffn = dense_ffn
    if ffn <= 0:
        raise ValueError(f"cannot size the FFN of {name or 'model'}")

    return ModelShape(
        name=name or str(cfg.get("_name_or_path") or "model"),
        n_layers=n_layers, hidden=hidden, n_q_heads=n_q, n_kv_heads=n_kv,
        head_dim=head_dim, ffn_params_per_layer=ffn)


# Fallbacks for when the model directory is not readable from the
# scheduler host.  Keyed by a substring of the model path.
KNOWN_SHAPES: dict[str, ModelShape] = {
    # 6923 is the TTFT-residual fit; the shape gives 6912, 0.16% below.
    # Every landed 30B cell ran 6923, and the difference moves the cost of
    # a 4k append on a 60k prefix by 0.1%, so the measurement stays and the
    # cells stay comparable.
    "Qwen3-30B-A3B": ModelShape(
        name="Qwen3-30B-A3B", n_layers=48, hidden=2048, n_q_heads=32,
        n_kv_heads=4, head_dim=128, ffn_params_per_layer=8 * 3 * 2048 * 768,
        l_eq_override=6923.0),
    "Qwen3-235B-A22B": ModelShape(
        name="Qwen3-235B-A22B", n_layers=94, hidden=4096, n_q_heads=64,
        n_kv_heads=4, head_dim=128,
        ffn_params_per_layer=8 * 3 * 4096 * 1536),
}


def resolve(model: str | None) -> ModelShape | None:
    """Best shape for a model path or name, or None if unknown.

    Reads ``config.json`` under the path when it is there, so a model the
    registry has never seen still works; falls back to the registry by
    substring, which is what the scheduler host needs when the weights
    live only on the engine hosts.
    """
    if not model:
        return None
    cfg_path = os.path.join(model, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as fh:
                return from_hf_config(json.load(fh), name=os.path.basename(
                    model.rstrip("/")))
        except (OSError, ValueError, KeyError):
            pass
    base = os.path.basename(model.rstrip("/"))
    for key, shape in KNOWN_SHAPES.items():
        if key in model or key in base:
            return shape
    return None
