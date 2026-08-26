#!/usr/bin/env python3
"""wav2vec2 STT — transcribe + score the wav2vec2 (CTC) model only.

All paths/models/params come from config.yml. Writes an independent report into
./reports. Faster than Whisper, so run this first to check improvement.

Run:  python wav2vec2_stt/analyze.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpac_stt.cli import add_config_arg  # noqa: E402
from bpac_stt.model_report import analyze_model  # noqa: E402
from bpac_stt.settings import DEFAULT_CONFIG_PATH, load_config, select_model  # noqa: E402

TARGET_TYPE = "wav2vec2_ctc"


def main():
    ap = argparse.ArgumentParser(description="wav2vec2 STT (single model)")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--model-key", default=None,
                    help="Override which config model to use (default: first 'wav2vec2_ctc' model).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = select_model(cfg, type_=TARGET_TYPE, key=args.model_key)
    out = cfg.reports_dir_for(__file__)
    analyze_model(cfg, model, out)


if __name__ == "__main__":
    main()
