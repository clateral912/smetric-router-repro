#!/usr/bin/env bash
set -euo pipefail

# Site-specific reset hook for scripts/run_matrix.py.
#
# The matrix invokes this file as:
#   reset_hook.example.sh SETTING POLICY REPLICATE
# It must stop the previous router/workers, clear the engine KV state and
# Mooncake namespace, start eight healthy workers with the same configuration,
# and return only after the KV-event endpoints and health checks are ready.
#
# Keep this file as a template. Do not commit credentials, model paths, or
# cluster-specific process IDs. A real deployment should copy it to a private
# site directory and set RESET_IMPL to an implementation owned by that site.

if [[ $# != 3 ]]; then
  echo "usage: $0 SETTING POLICY REPLICATE" >&2
  exit 2
fi

: "${RESET_IMPL:?Set RESET_IMPL to your site-specific stop/clear/start script}"
exec "$RESET_IMPL" "$@"
