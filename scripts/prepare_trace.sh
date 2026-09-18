#!/usr/bin/env bash
set -euo pipefail
# Downloads the public source trace and creates the exact 220/110-session
# replay traces used by the matrix. The generated JSONL is intentionally not
# committed; its SHA256 is recorded in provenance/trace-lock.json.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${MODEL_TOKENIZER:?Set MODEL_TOKENIZER to the local tokenizer.json}"
RAW=${RAW_TRACE:-$ROOT/traces/codex_swebenchpro.json}
mkdir -p "$(dirname "$RAW")"
if [[ ! -f "$RAW" ]]; then
  HF_DATASET=${HF_DATASET:-Inferact/codex_swebenchpro_traces}
  HF_FILE=${HF_FILE:-codex_swebenchpro.json}
  python - "$HF_DATASET" "$HF_FILE" "$RAW" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, filename, output = sys.argv[1:]
source = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename)
from pathlib import Path
Path(output).write_bytes(Path(source).read_bytes())
PY
fi
export PYTHONPATH="$ROOT/python"
python "$ROOT/python/build_trace.py" --input "$RAW" --tokenizer "$MODEL_TOKENIZER" --sessions 220 \
  --output "$ROOT/traces/codex/steady_warm1200_measure600_n220_seed42.jsonl" \
  --span-seconds 900 --pre-roll-seconds 1200 --seed 42
python "$ROOT/python/build_trace.py" --input "$RAW" --tokenizer "$MODEL_TOKENIZER" --sessions 110 \
  --output "$ROOT/traces/codex/steady_warm1200_measure600_n110_seed42.jsonl" \
  --span-seconds 900 --pre-roll-seconds 1200 --seed 42
sha256sum "$ROOT"/traces/codex/steady_warm1200_measure600_n{220,110}_seed42.jsonl
