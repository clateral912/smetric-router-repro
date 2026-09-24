"""Run the two workload settings x three native Router policies x N replicas.

The script deliberately requires a reset hook. It must clear engine KV state,
Mooncake, and restart workers before every arm; silently reusing state would
invalidate the comparison.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICIES = (
    "cache_aware",
    "power_of_two",
    "consistent_hash",
    "rendezvous_hash",
    "smetric_default",
    "smetric_optimized",
)
DEFAULT_POLICIES = ("cache_aware", "smetric_default", "smetric_optimized")
SETTINGS = {"po": ROOT / "configs/prefill-400k.yaml",
            "pd110": ROOT / "configs/pd-mixed-400k-n110.yaml"}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=(*SETTINGS, "all"), default="all")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(DEFAULT_POLICIES),
        help="Policies to run in the supplied order",
    )
    ap.add_argument("--reset-hook", required=True,
                    help="Executable called as RESET_HOOK setting policy replicate")
    ap.add_argument("--router-binary", type=Path, default=ROOT / "router/target/release/vllm-router")
    ap.add_argument("--output-root", type=Path, default=ROOT / "results")
    ap.add_argument("--kv-events", action="store_true")
    ap.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip an arm when its replicate directory already contains a summary.json",
    )
    args = ap.parse_args()
    settings = SETTINGS if args.setting == "all" else {args.setting: SETTINGS[args.setting]}
    runner = ROOT / "python/run_router.py"
    for setting, config in settings.items():
        for rep in range(1, args.replicates + 1):
            for policy in args.policies:
                out = args.output_root / setting / f"replicate-{rep}"
                out.mkdir(parents=True, exist_ok=True)
                policy_slug = policy.replace("_", "-")
                completed = list(out.glob(f"*-vllm-router-native-{policy_slug}_*/summary.json"))
                if args.skip_completed and completed:
                    print("SKIP completed", setting, policy, rep, completed[-1], flush=True)
                    continue
                subprocess.run([args.reset_hook, setting, policy, str(rep)], check=True)
                port = 18090 + rep * 10 + (0 if setting == "po" else 1)
                metrics = 19090 + rep * 10 + (0 if setting == "po" else 1)
                cmd = [sys.executable, str(runner), "--config", str(config),
                       "--router-binary", str(args.router_binary),
                       "--router-source", str(ROOT / "router"), "--arm", policy,
                       "--output-root", str(out),
                       "--port", str(port), "--metrics-port", str(metrics)]
                if args.kv_events:
                    cmd.append("--kv-events")
                env = dict(os.environ); env["PYTHONPATH"] = str(ROOT / "python")
                print("RUN", setting, policy, rep, flush=True)
                subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    main()
