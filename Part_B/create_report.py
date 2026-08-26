#!/usr/bin/env python3
"""create_report.py — combined STT report across all transcribed models.

Reads the shared predictions cache (does NOT transcribe) and builds:
  - a whole-corpus model-comparison table (WER / CER per model)
  - region x model WER and CER matrices
  - a region ranking ("distance from standard") averaged across models
  - figures + report.md

Works with whatever models are present in the cache, so you can run it after
just wav2vec2, or after both. Place this file in Part_B/ and run:

    python create_report.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bpac_stt.cli import add_config_arg  # noqa: E402
from bpac_stt.reporting import ReportWriter, ensure_dir, save_fig, save_json, save_table  # noqa: E402
from bpac_stt.scoring import overall_metrics, per_region_metrics  # noqa: E402
from bpac_stt.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combined_report")


def _bar_by_region(matrix, metric, path):
    regions = list(matrix.index)
    models = list(matrix.columns)
    x = np.arange(len(regions))
    w = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(7, len(regions) * 1.1), 5))
    for j, mdl in enumerate(models):
        ax.bar(x + (j - (len(models) - 1) / 2) * w, matrix[mdl].values, w, label=mdl)
    ax.set_xticks(x); ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_ylabel(metric.upper()); ax.set_title(f"Per-region {metric.upper()} by model")
    ax.legend(fontsize=8)
    save_fig(fig, path)


def main():
    ap = argparse.ArgumentParser(description="Combined STT report")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(args.out)

    cache = cfg.predictions_cache
    if not os.path.exists(cache):
        raise SystemExit(f"No predictions cache at {cache}. Run a model first "
                         f"(e.g. python wav2vec2_stt/analyze.py).")
    pred = pd.read_csv(cache)

    hyp_cols = [c for c in pred.columns if c.startswith("hyp_")]
    # only models that are actually populated
    hyp_cols = [c for c in hyp_cols if pred[c].notna().any()]
    if not hyp_cols:
        raise SystemExit("Cache has no populated hypothesis columns yet.")
    key2label = {m["key"]: m["label"] for m in cfg.models}
    key2type = {m["key"]: m["type"] for m in cfg.models}
    primary = cfg.primary_metric.upper()

    # ---- whole-corpus comparison + per-region per model ----
    comp_rows, region_long = [], []
    for col in hyp_cols:
        key = col[len("hyp_"):]
        ov = overall_metrics(pred, col, cfg)
        comp_rows.append({"model_key": key, "label": key2label.get(key, key),
                          "type": key2type.get(key, "?"),
                          "WER": round(ov["wer"], 4), "CER": round(ov["cer"], 4),
                          "n_scored": ov["n"], "n_excluded_empty": ov["n_excluded_empty"]})
        reg = per_region_metrics(pred, col, cfg)
        for _, r in reg.iterrows():
            region_long.append({"model_key": key, "region": r["region"],
                                "WER": r["WER"], "CER": r["CER"], "n": r["n"]})

    comp = pd.DataFrame(comp_rows).sort_values(primary).reset_index(drop=True)
    long_df = pd.DataFrame(region_long)
    save_table(comp, out / "model_comparison.csv", index=False)
    save_table(long_df, out / "region_metrics_long.csv", index=False)

    regions = sorted(long_df["region"].unique())
    wer_matrix = long_df.pivot(index="region", columns="model_key", values="WER").reindex(regions)
    cer_matrix = long_df.pivot(index="region", columns="model_key", values="CER").reindex(regions)
    save_table(wer_matrix, out / "wer_matrix.csv")
    save_table(cer_matrix, out / "cer_matrix.csv")

    metric_matrix = cer_matrix if primary == "CER" else wer_matrix
    ranking = metric_matrix.copy()
    ranking["mean_across_models"] = ranking.mean(axis=1)
    ranking = ranking.sort_values("mean_across_models", ascending=False)
    save_table(ranking, out / "region_ranking.csv")

    if len(regions):
        _bar_by_region(cer_matrix, "cer", out / "region_cer_by_model.png")
        _bar_by_region(wer_matrix, "wer", out / "region_wer_by_model.png")

    summary = {
        "models_present": [c[len("hyp_"):] for c in hyp_cols],
        "n_utterances_cached": int(len(pred)),
        "primary_metric": primary,
        "exclude_empty_audio": cfg.exclude_empty_audio,
        "best_overall_by_primary": comp.iloc[0]["model_key"] if len(comp) else None,
        "hardest_region": ranking.index[0] if len(ranking) else None,
        "easiest_region": ranking.index[-1] if len(ranking) else None,
    }
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "Combined STT Report")
    rep.p(f"Models present in cache: **{', '.join(summary['models_present'])}**. "
          + ("Empty-audio utterances excluded from scoring." if cfg.exclude_empty_audio
             else "Empty-audio utterances NOT excluded."))
    if len(hyp_cols) < 2:
        rep.p("> Only one model is present so far. Run the other model "
              "(`whisper_stt/analyze.py` or `wav2vec2_stt/analyze.py`) and re-run "
              "this script for the full cross-model comparison.")
    rep.h2("Whole-corpus comparison")
    rep.table(comp, index=False)
    if len(regions):
        rep.h2(f"Region ranking — distance from standard (by {primary})")
        rep.p("Higher = farther from standard Bengali (more ASR degradation).")
        rep.table(ranking.round(4))
        rep.h2("WER matrix (region x model)")
        rep.table(wer_matrix.round(4))
        rep.h2("CER matrix (region x model)")
        rep.table(cer_matrix.round(4))
        rep.image("region_cer_by_model.png", "Per-region CER by model.")
        rep.image("region_wer_by_model.png", "Per-region WER by model.")
    rep.h2("Read")
    rep.bullet("A consistent region ranking across models is a robust signal; it "
               "should corroborate Part C. Disagreement = model-dependence.")
    rep.bullet("Lean on CER for the ranking (Bangla orthography makes WER noisier).")
    rep.write()

    print(f"[create_report] models={summary['models_present']} "
          f"best={summary['best_overall_by_primary']} hardest={summary['hardest_region']}")


if __name__ == "__main__":
    main()
