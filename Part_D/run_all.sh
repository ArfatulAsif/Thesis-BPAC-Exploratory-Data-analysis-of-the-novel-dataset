#!/usr/bin/env bash
# Run all Part D analyses. Paths/params come from config.yml.
# Usage: ./run_all.sh [path/to/config.yml]   (defaults to ./config.yml)
#
# D0-D7 analyze.py scripts land as T11-T19 are implemented (see repo
# README task table); until then each step is skipped rather than failing,
# so this script stays runnable throughout Phase 4.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
CONFIG="${1:-config.yml}"

run_step () {
  local label="$1" script="$2"
  if [[ -f "$script" ]]; then
    echo "== $label =="
    python "$script" --config "$CONFIG"
  else
    echo "== $label == SKIP (not yet implemented: $script)"
  fi
}

run_step "D0 lexicon (T11)"                 "D0_lexicon/analyze.py"
run_step "D1 forced alignment (T12)"        "D1_alignment/analyze.py"
run_step "D2 duration & rhythm (T13-T14)"   "D2_duration_rhythm/analyze.py"
run_step "D3 vowel space (T15)"             "D3_vowel_space/analyze.py"
run_step "D4 consonants (T16)"              "D4_consonants/analyze.py"
run_step "D5 intonation (T17)"              "D5_intonation/analyze.py"
run_step "D6 phonetic distance (T18)"       "D6_phonetic_distance/analyze.py"
run_step "D7 convergence test (T19)"        "D7_convergence/analyze.py"
echo "All done (implemented steps only). Reports are in each Part_D/D*/reports/ folder."
