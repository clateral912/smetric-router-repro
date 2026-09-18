"""RoutingPolicy protocol and registry.

A policy is a pure function: given request context and a cluster snapshot,
it returns a routing decision. No I/O, no sleep, no retry — those belong
to the router core. All load reads go through the snapshot's eff_*
accessors, which conservatively blend Redis and shadow state.

The registry maps names to policy *classes*; ``create(name, **params)``
instantiates one with per-run parameters (e.g. ali alpha/beta/topk).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..core import ClusterSnapshot, RequestContext


@dataclass(frozen=True)
class Decision:
    instance_idx: int
    reason: str
    # Optional per-decision telemetry merged into the decisions log
    # (e.g. SMetric's smetric_gate for gate-boundary analysis).
    extra: dict | None = None


@runtime_checkable
class RoutingPolicy(Protocol):
    def route(self, req: RequestContext, view: ClusterSnapshot) -> Decision: ...


_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator: @register("smetric")."""
    def wrap(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls
    return wrap


def create(name: str, **params) -> RoutingPolicy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown policy {name!r}; available: {available()}")
    return _REGISTRY[name](**params)


def available() -> list[str]:
    return sorted(_REGISTRY)


def policy_class(name: str) -> type:
    """The registered class, for introspecting which kwargs it takes."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown policy {name!r}; available: {available()}")
    return _REGISTRY[name]
