#!/usr/bin/env bash
# Run all Part E analyses. Paths/params come from config.yml.
# Usage: ./run_all.sh [path/to/config.yml]   (defaults to ./config.yml)
#
# E1's analyze.py lands as T20 is implemented; until then it is skipped
# rather than failing, so this script stays runnable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
CONFIG="${1:-config.yml}"

if [[ -f "E1_denoise_check/analyze.py" ]]; then
  echo "== E1 denoise sanity check (T20) =="
  python E1_denoise_check/analyze.py --config "$CONFIG"
else
  echo "== E1 denoise sanity check (T20) == SKIP (not yet implemented: E1_denoise_check/analyze.py)"
fi
echo "All done (implemented steps only). Reports are in Part_E/E1_denoise_check/reports/."
