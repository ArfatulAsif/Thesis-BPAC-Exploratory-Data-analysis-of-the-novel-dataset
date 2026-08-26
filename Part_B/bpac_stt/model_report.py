"""Shared logic for a single-model STT report (used by whisper_stt and
wav2vec2_stt). Transcribes one model (resumable), scores overall + per region,
and writes CSVs, a bar chart, and report.md into the given reports dir.
"""
import matplotlib.pyplot as plt
import numpy as np

from .asr import ensure_predictions
from .reporting import ReportWriter, ensure_dir, save_fig, save_json, save_table
from .scoring import overall_metrics, per_region_metrics


def analyze_model(cfg, model, out_dir):
    out = ensure_dir(out_dir)
    col = f"hyp_{model['key']}"

    pred = ensure_predictions(cfg, models=[model])
    ov = overall_metrics(pred, col, cfg)
    reg = per_region_metrics(pred, col, cfg)
    primary = cfg.primary_metric.upper()
    reg = reg.sort_values(primary, ascending=False).reset_index(drop=True)

    summary = {
        "model_key": model["key"], "label": model["label"], "hf_id": model["hf_id"],
        "type": model["type"],
        "WER": round(ov["wer"], 4), "CER": round(ov["cer"], 4),
        "n_scored": ov["n"], "n_transcribed": ov["n_transcribed"],
        "n_excluded_empty_audio": ov["n_excluded_empty"], "n_empty_hyp": ov["n_empty_hyp"],
        "exclude_empty_audio": cfg.exclude_empty_audio,
        "hardest_region": reg.iloc[0]["region"] if len(reg) else None,
        "easiest_region": reg.iloc[-1]["region"] if len(reg) else None,
    }
    save_json(summary, out / "summary.json")
    save_table(reg, out / "region_metrics.csv", index=False)

    # Per-region WER/CER bar chart (single model)
    if len(reg):
        x = np.arange(len(reg)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(7, len(reg) * 1.1), 5))
        ax.bar(x - w / 2, reg["WER"], w, label="WER")
        ax.bar(x + w / 2, reg["CER"], w, label="CER")
        ax.set_xticks(x); ax.set_xticklabels(reg["region"], rotation=30, ha="right")
        ax.set_ylabel("error rate")
        ax.set_title(f"{model['label']} — per-region WER / CER")
        ax.legend()
        save_fig(fig, out / "region_metrics.png")

    rep = ReportWriter(out, f"STT — {model['label']}")
    rep.p(f"**Model:** `{model['hf_id']}` ({model['type']}).")
    rep.p("**Preprocessing:** 16 kHz mono; identical NFC + punctuation normalization "
          "on reference and hypothesis; no per-region denoising."
          + (f" Empty-audio utterances excluded from scoring "
             f"({summary['n_excluded_empty_audio']} dropped)." if cfg.exclude_empty_audio
             else " Empty-audio utterances NOT excluded."))
    rep.h2("Overall")
    rep.bullet(f"WER **{summary['WER']}**, CER **{summary['CER']}** over "
               f"{summary['n_scored']} scored utterances "
               f"(of {summary['n_transcribed']} transcribed; "
               f"{summary['n_excluded_empty_audio']} excluded as empty audio).")
    rep.h2(f"Per-region (sorted by {primary}, hardest first)")
    rep.table(reg, index=False)
    if len(reg):
        rep.image("region_metrics.png", f"{model['label']} per-region WER/CER.")
    rep.h2("Notes")
    rep.bullet("Higher per-region error = farther from standard Bengali.")
    rep.bullet("Run the other model and `create_report.py` for the cross-model "
               "comparison and the region ranking averaged across models.")
    rep.write()

    print(f"[{model['key']}] WER={summary['WER']} CER={summary['CER']} "
          f"hardest={summary['hardest_region']} excluded_empty={summary['n_excluded_empty_audio']}")
    return summary, reg
