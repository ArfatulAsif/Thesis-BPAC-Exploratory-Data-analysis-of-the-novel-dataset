#!/usr/bin/env bash
# Run all Part A analyses. Paths/params come from config.yml.
# Usage: ./run_all.sh [path/to/config.yml]   (defaults to ./config.yml)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
CONFIG="${1:-config.yml}"

echo "== A1 unique sentences =="          ; python A1_unique_sentences/analyze.py     --config "$CONFIG"
echo "== A2 audio length =="              ; python A2_audio_length/analyze.py         --config "$CONFIG"
echo "== A3 utterance count =="           ; python A3_utterance_count/analyze.py      --config "$CONFIG"
echo "== A4 character count =="           ; python A4_character_count/analyze.py      --config "$CONFIG"
echo "== A5 word count =="                ; python A5_word_count/analyze.py           --config "$CONFIG"
echo "== A6 speaking rate & pitch =="     ; python A6_speaking_rate_pitch/analyze.py  --config "$CONFIG"
echo "All done. Reports are in each Part_A/A*/reports/ folder."
