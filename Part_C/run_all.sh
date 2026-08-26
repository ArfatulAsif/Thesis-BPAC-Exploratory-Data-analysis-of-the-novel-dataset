#!/usr/bin/env bash
# C0 normalize the corpus, then C1 (distance) and C2 (separability) on the clean tree.
# Usage: ./run_all.sh [path/to/config.yml]   (defaults to ./config.yml)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
CONFIG="${1:-config.yml}"
echo "== C0 normalize corpus + diagnostics =="
python C0_Normalize_corpus/analyze.py --config "$CONFIG"
echo "== C1 region accent-distance matrix (clean) =="
python C1_Region_accent_distance_matrix/analyze.py --config "$CONFIG"
echo "== C2 speaker-vs-accent separability (clean) =="
python C2_Speaker_vs_accent_embedding_separability/analyze.py --config "$CONFIG"
echo "Done. See C0_/reports, C1_/reports, C2_/reports."
