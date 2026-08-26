#!/usr/bin/env python3
"""Whisper STT — transcribe + score the Whisper model only.

All paths/models/params come from config.yml. Writes an independent report into
./reports. Uses the shared cache, so running this and wav2vec2 in any order is
fine; the corpus is transcribed once per model.

Run:  python whisper_stt/analyze.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpac_stt.cli import add_config_arg  # noqa: E402
from bpac_stt.model_report import analyze_model  # noqa: E402
from bpac_stt.settings import DEFAULT_CONFIG_PATH, load_config, select_model  # noqa: E402

TARGET_TYPE = "whisper"


def main():
    ap = argparse.ArgumentParser(description="Whisper STT (single model)")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--model-key", default=None,
                    help="Override which config model to use (default: first 'whisper' model).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = select_model(cfg, type_=TARGET_TYPE, key=args.model_key)
    out = cfg.reports_dir_for(__file__)
    analyze_model(cfg, model, out)


if __name__ == "__main__":
    main()
