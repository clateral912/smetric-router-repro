"""ssched CLI.

Subcommands:
  ssched trace sample    window+session sampling of a cluster-scale trace
  ssched trace annotate  join time_to_parent_chat from the raw trace
  ssched scheduler       run the global scheduler proxy
  ssched replay          closed-loop trace replay against a scheduler
  ssched srr             open-loop Poisson SRR loadgen (TPS-within-SLO)
  ssched run             config-driven run -> self-contained artifact dir
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_kv(pairs: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"ssched: bad {what} {pair!r}, expected k=v")
        key, value = pair.split("=", 1)
        out[key] = value
    return out


def _cmd_scheduler(args: argparse.Namespace) -> None:
    import json

    import uvicorn
    import yaml

    from .scheduler.app import SchedulerConfig, create_app

    if args.config_json:
        config = SchedulerConfig(**json.loads(args.config_json.read_text()))
    else:
        instances = [
            {"engine_id": eid, "url": url.rstrip("/")}
            for eid, url in _parse_kv(args.instance, "--instance").items()]
        if not instances:
            raise SystemExit(
                "ssched scheduler: at least one --instance required")
        policy_params = {
            k: yaml.safe_load(v)  # "0.5" -> float, "3" -> int
            for k, v in _parse_kv(args.policy_param, "--policy-param").items()
        }
        config = SchedulerConfig(
            policy=args.policy,
            policy_params=policy_params,
            instances=instances,
            redis_url=args.redis_url,
            seed=args.seed,
            engine_state_poll_ms=args.engine_state_poll_ms,
            request_timeout_s=args.request_timeout,
        )
    print(f"[scheduler] policy={config.policy} params={config.policy_params} "
          f"instances={len(config.instances)} "
          f"redis={config.redis_url or 'OFF'}", flush=True)
    uvicorn.run(create_app(config), host=args.host, port=args.port,
                log_level="warning")


def _cmd_trace_sample(args: argparse.Namespace) -> None:
    from .trace.loader import group_by_session, load_trace
    from .trace.sampler import (
        build_output, sample_sessions, summarize, write_jsonl,
    )

    if args.sample_ratio is None and args.target_requests is None:
        raise SystemExit(
            "ssched trace sample: must specify --sample-ratio or "
            "--target-requests")

    print(f"Loading trace from {args.input} ...")
    rows_by_session = group_by_session(load_trace(args.input))
    total_requests = sum(len(v) for v in rows_by_session.values())
    print(f"Full trace: {len(rows_by_session)} sessions, "
          f"{total_requests} requests")

    selection = sample_sessions(
        rows_by_session,
        sample_ratio=args.sample_ratio,
        target_requests=args.target_requests,
        max_single_turn_ratio=args.max_single_turn_ratio,
        window_seconds=args.window_seconds,
        admit=args.admit,
        seed=args.seed,
    )
    window = selection.window if args.admit == "active" else None
    out_rows = build_output(rows_by_session, selection.session_ids,
                            window=window)
    print(summarize(rows_by_session, selection.session_ids, out_rows).format())

    write_jsonl(out_rows, args.output)
    print(f"Wrote {len(out_rows)} rows to {args.output}")


def _cmd_trace_codex_import(args: argparse.Namespace) -> None:
    from .trace.codex_sharegpt import (
        ImportParams, NominalService, ThinkTimeModel, import_codex_traces,
    )

    params = ImportParams(
        span_seconds=args.span_seconds,
        sessions=args.sessions,
        seed=args.seed,
        think=ThinkTimeModel(
            median_s=args.think_median,
            p90_s=args.think_p90,
            service_offset_s=args.think_service_offset,
        ),
        service=NominalService(
            ttft_s=args.nominal_ttft,
            prefill_tokens_per_s=args.nominal_prefill_tps,
            tpot_s=args.nominal_tpot,
        ),
        shared_system_prefix_tokens=args.shared_system_prefix_tokens,
        replicas=args.replicas,
        pre_roll_seconds=args.pre_roll_seconds,
        replay_pre_roll=args.replay_pre_roll,
        arrival_scheme=args.arrival_scheme,
    )
    stats = import_codex_traces(args.input, args.tokenizer, args.output,
                                params)
    print(stats.format())
    print(f"Wrote {args.output}")


def _cmd_trace_annotate(args: argparse.Namespace) -> None:
    from .trace.annotate import annotate_trace

    stats = annotate_trace(args.input, args.output, args.raw)
    print(stats.format())
    print(f"Wrote {args.output}")


def _cmd_replay(args: argparse.Namespace) -> None:
    import asyncio
    import logging

    from .replayer.replay import ReplayConfig, replay_trace

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = ReplayConfig(
        trace_path=args.trace,
        output_path=args.output,
        endpoint_url=args.endpoint.rstrip("/"),
        model_name=args.model,
        dispatch_mode=args.dispatch_mode,
        request_timeout_s=args.request_timeout,
        request_limit=args.request_limit,
        max_inflight_sessions=args.max_inflight_sessions,
        inter_turn_think_s=args.inter_turn_think,
        max_duration_s=args.max_duration,
        prefill_only=args.prefill_only,
    )
    results = asyncio.run(replay_trace(config))
    ok = sum(1 for r in results if r.error is None)
    print(f"Done: {ok}/{len(results)} requests succeeded")
    print(f"Metrics: {args.output}")


def _cmd_srr(args: argparse.Namespace) -> None:
    import asyncio
    import json
    import logging

    from .replayer.srr import SrrConfig, run_srr

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = SrrConfig(
        trace_path=args.trace,
        output_path=args.output,
        endpoint_url=args.endpoint.rstrip("/"),
        model_name=args.model,
        arrival_rate=args.arrival_rate,
        warmup_s=args.warmup_s,
        steady_s=args.steady_s,
        drain_s=args.drain_s,
        request_timeout_s=args.request_timeout,
        session_pool_size=args.session_pool_size,
        rng_seed=args.rng_seed,
    )
    summary = asyncio.run(run_srr(config))
    print(json.dumps(summary, indent=2, sort_keys=True))


def _cmd_run(args: argparse.Namespace) -> None:
    import signal

    from .run import RunSpec, execute

    def handle_term(signum, _frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, handle_term)
    spec = RunSpec.from_yaml(args.config)
    try:
        run_dir = execute(spec, output_root=args.output_root,
                          redis_url=args.redis_url)
        print(f"Run artifacts: {run_dir}")
    finally:
        signal.signal(signal.SIGTERM, previous)


def _cmd_instances(args: argparse.Namespace) -> None:
    from .instances import patcher

    argv = [args.action, "--venv", str(args.venv)]
    if args.action == "check":
        argv.extend(["--expect", args.expect])
    patcher.main(argv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ssched", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    trace = sub.add_parser("trace", help="offline trace tools")
    tsub = trace.add_subparsers(dest="trace_command", required=True)

    sample = tsub.add_parser(
        "sample", help="sample sessions preserving KV reuse patterns")
    sample.add_argument("--input", type=Path, required=True,
                        help="full trace JSONL")
    sample.add_argument("--output", type=Path, required=True,
                        help="sampled trace JSONL")
    sample.add_argument("--sample-ratio", type=float, default=None,
                        help="fraction of sessions to keep (window+thin)")
    sample.add_argument("--target-requests", type=int, default=None,
                        help="target request count (legacy, no sharing "
                             "preservation)")
    sample.add_argument("--max-single-turn-ratio", type=float, default=None,
                        help="cap single-turn sessions to this fraction")
    sample.add_argument("--window-seconds", type=float, default=None,
                        help="fixed window duration (s)")
    sample.add_argument("--admit", choices=("born", "active"), default="born",
                        help="window admission: 'born' keeps whole sessions "
                             "first-seen in the window; 'active' keeps the "
                             "in-window turns of every session active there "
                             "(stationary arrival curve, mid-flight entries)")
    sample.add_argument("--seed", type=int, default=42)
    sample.set_defaults(func=_cmd_trace_sample)

    codex = tsub.add_parser(
        "codex-import",
        help="import the codex SWE-Bench-Pro ShareGPT dump as a trace")
    codex.add_argument("--input", type=Path, required=True,
                       help="codex_swebenchpro.json from the HF dataset")
    codex.add_argument("--tokenizer", type=Path, required=True,
                       help="tokenizer.json of the SERVED model")
    codex.add_argument("--output", type=Path, required=True,
                       help="full-population trace JSONL")
    codex.add_argument("--span-seconds", type=float, default=3600.0,
                       help="spread session arrivals over this span (s); "
                            "shorter span = denser offered load")
    codex.add_argument("--pre-roll-seconds", type=float, default=0.0,
                       help="draw arrivals over [-P, span); crop negative "
                            "turns unless --replay-pre-roll is set")
    codex.add_argument("--replay-pre-roll", action="store_true",
                       help="retain negative-time turns for real warmup; "
                            "score only after P seconds of replay")
    codex.add_argument("--arrival-scheme", choices=("uniform", "stratified"),
                       default="uniform",
                       help="random or evenly distributed nested arrivals")
    codex.add_argument("--sessions", type=int, default=None,
                       help="admit only this many trials over the whole "
                            "[-P, span) arrival span (default: all)")
    codex.add_argument("--think-median", type=float, default=5.2,
                       help="median inter-call gap (s), dataset card 5.2")
    codex.add_argument("--think-p90", type=float, default=23.0,
                       help="p90 inter-call gap (s), dataset card 23.0")
    codex.add_argument("--think-service-offset", type=float, default=0.0,
                       help="service time to subtract from the card's "
                            "start-to-start gap to get think time")
    codex.add_argument("--shared-system-prefix-tokens", type=int,
                       default=0,
                       help="pin this many head tokens of every "
                            "session to one canonical block chain "
                            "(dataset card reports 11520 cached "
                            "cross-trial on call 1; the published "
                            "text only shares 902)")
    codex.add_argument("--replicas", type=int, default=1,
                       help="admit each trial this many times, "
                            "each with its own session, arrival, "
                            "think draws and block salt; raises "
                            "the load ceiling past the 610-trial "
                            "pool")
    codex.add_argument("--nominal-ttft", type=float, default=1.0,
                       help="offline service model, row timestamps only")
    codex.add_argument("--nominal-prefill-tps", type=float,
                       default=8000.0,
                       help="offline service model, row timestamps only")
    codex.add_argument("--nominal-tpot", type=float, default=0.02,
                       help="offline service model, row timestamps "
                            "only; use ~0 for a prefill-only run")
    codex.add_argument("--seed", type=int, default=42)
    codex.set_defaults(func=_cmd_trace_codex_import)

    annotate = tsub.add_parser(
        "annotate", help="join time_to_parent_chat from the raw trace")
    annotate.add_argument("--input", type=Path, required=True,
                          help="sampled trace JSONL")
    annotate.add_argument("--output", type=Path, required=True,
                          help="annotated trace JSONL")
    annotate.add_argument("--raw", type=Path, required=True,
                          help="raw trace JSONL with request_ready/end_time_ms")
    annotate.set_defaults(func=_cmd_trace_annotate)

    sched = sub.add_parser("scheduler", help="run the global scheduler proxy")
    sched.add_argument("--config-json", type=Path, default=None,
                       help="full SchedulerConfig as JSON (overrides the "
                            "individual flags; used by ssched run)")
    sched.add_argument("--policy", default="smetric",
                       help="routing policy name (see ssched.scheduler.policy)")
    sched.add_argument("--policy-param", action="append", default=[],
                       metavar="K=V",
                       help="policy constructor override, repeatable "
                            "(e.g. alpha=50 topk=3 overload_factor=2.0)")
    sched.add_argument("--instance", action="append", default=[],
                       metavar="ENGINE_ID=URL",
                       help="backend instance, repeatable "
                            "(e.g. engine_0=http://host:8000)")
    sched.add_argument("--redis-url", default=None,
                       help="engine-state feed (redis://host:port/0); "
                            "empty = shadow-only routing")
    sched.add_argument("--host", default="0.0.0.0")
    sched.add_argument("--port", type=int, default=9300)
    sched.add_argument("--seed", type=int, default=0)
    sched.add_argument("--engine-state-poll-ms", type=int, default=50)
    sched.add_argument("--request-timeout", type=float, default=600.0)
    sched.set_defaults(func=_cmd_scheduler)

    replay = sub.add_parser("replay",
                            help="closed-loop trace replay via a scheduler")
    replay.add_argument("--trace", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True,
                        help="requests JSONL output")
    replay.add_argument("--endpoint", required=True,
                        help="scheduler URL (single endpoint)")
    replay.add_argument("--model", default="default")
    replay.add_argument("--dispatch-mode", choices=["thinktime", "tracets"],
                        default="thinktime",
                        help="thinktime = faithful closed-loop (default); "
                             "tracets = absolute schedule (bursty stress)")
    replay.add_argument("--request-timeout", type=float, default=600.0)
    replay.add_argument("--request-limit", type=int, default=None)
    replay.add_argument("--max-inflight-sessions", type=int, default=None)
    replay.add_argument("--inter-turn-think", type=float, default=None,
                        help="fixed closed-loop think-time (s); use with "
                             "--dispatch-mode tracets")
    replay.add_argument("--max-duration", type=float, default=None)
    replay.add_argument("--prefill-only", action="store_true")
    replay.set_defaults(func=_cmd_replay)

    srr = sub.add_parser("srr", help="open-loop Poisson SRR loadgen")
    srr.add_argument("--trace", type=Path, required=True)
    srr.add_argument("--output", type=Path, required=True)
    srr.add_argument("--endpoint", required=True)
    srr.add_argument("--model", default="default")
    srr.add_argument("--arrival-rate", type=float, required=True,
                     help="sessions per second (Poisson)")
    srr.add_argument("--warmup-s", type=float, default=60.0)
    srr.add_argument("--steady-s", type=float, default=300.0)
    srr.add_argument("--drain-s", type=float, default=60.0)
    srr.add_argument("--request-timeout", type=float, default=600.0)
    srr.add_argument("--session-pool-size", type=int, default=None)
    srr.add_argument("--rng-seed", type=int, default=42)
    srr.set_defaults(func=_cmd_srr)

    run = sub.add_parser("run",
                         help="config-driven run -> artifact directory")
    run.add_argument("--config", type=Path, required=True,
                     help="RunSpec YAML (see examples/smoke.yaml)")
    run.add_argument("--output-root", type=Path, default=None)
    run.add_argument("--redis-url", default=None)
    run.set_defaults(func=_cmd_run)

    inst = sub.add_parser(
        "instances",
        help="apply/verify the pinned vLLM patch series on an engine venv")
    inst.add_argument("action", choices=["apply", "check", "revert"])
    inst.add_argument("--venv", type=Path, required=True)
    inst.add_argument("--expect", choices=["pristine", "applied"],
                      default="applied",
                      help="required state for check (default: applied)")
    inst.set_defaults(func=_cmd_instances)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
