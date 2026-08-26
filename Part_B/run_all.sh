#!/usr/bin/env bash
# Run wav2vec2 first (fast), then Whisper, then the combined report.
# Usage: ./run_all.sh [path/to/config.yml]   (defaults to ./config.yml)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
CONFIG="${1:-config.yml}"
echo "== wav2vec2 (fast) =="     ; python wav2vec2_stt/analyze.py --config "$CONFIG"
echo "== whisper (slow) =="      ; python whisper_stt/analyze.py  --config "$CONFIG"
echo "== combined report =="     ; python create_report.py        --config "$CONFIG"
echo "Done. See wav2vec2_stt/reports/, whisper_stt/reports/, combined_report/."
