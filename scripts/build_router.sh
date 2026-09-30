#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}
TARGET=${CARGO_TARGET_DIR:-$ROOT/router/target}
case "$TARGET" in /*) ;; *) TARGET="$ROOT/$TARGET" ;; esac
export CARGO_TARGET_DIR="$TARGET"
cargo build --manifest-path "$ROOT/router/Cargo.toml" --release --locked --bin vllm-router "$@"
"$PYTHON" "$ROOT/scripts/source_identity.py" --binary "$TARGET/release/vllm-router"
printf 'Router binary: %s\n' "$TARGET/release/vllm-router"
