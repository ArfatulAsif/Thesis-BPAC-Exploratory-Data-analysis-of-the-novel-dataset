#!/usr/bin/env python3
"""A6 — Speaking rate (WPM) & pitch (F0), per region / speaker / gender.

All paths/params come from config.yml (a6_speaking_rate_pitch.top_db,
.max_per_speaker, .compute_f0; audio.pitch_ranges). Reads every audio file
(slow); set a6_speaking_rate_pitch.max_per_speaker for a quick pass.

Outputs (in ./reports): summary.json, utterance_metrics.csv,
per_region_gender.csv, wpm_by_region.png, f0_by_region.png, report.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bpac_eda.audio import mean_f0, speech_duration_seconds  # noqa: E402
from bpac_eda.cli import add_config_arg  # noqa: E402
from bpac_eda.manifest import build_manifest  # noqa: E402
from bpac_eda.reporting import (ReportWriter, ensure_dir, maybe_tqdm, save_fig,  # noqa: E402
                                save_json, save_table)
from bpac_eda.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from bpac_eda.textnorm import syllable_count_bangla, tokenize  # noqa: E402


def _violin_by_region(df, value, title, ylabel, path):
    genders = sorted(df["gender"].dropna().unique())
    regions = sorted(df["region"].unique())
    fig, axes = plt.subplots(1, max(1, len(genders)),
                             figsize=(7 * max(1, len(genders)), 5), squeeze=False)
    for gi, g in enumerate(genders):
        ax = axes[0][gi]
        
        for i, r in enumerate(regions):
            # Extract clean data for this specific gender and region
            d = df[(df["gender"] == g) & (df["region"] == r)][value].dropna().values
            
            if len(d) > 1:
                # Need at least 2 points for a violin plot
                ax.violinplot(d, positions=[i + 1], showmeans=True, showextrema=False)
            elif len(d) == 1:
                # If there's exactly 1 point, KDE fails. Plot it as a single scatter dot.
                ax.scatter([i + 1], d, color='C0', zorder=3)
                
        ax.set_xticks(range(1, len(regions) + 1))
        ax.set_xticklabels(regions, rotation=35, ha="right", fontsize=8)
        
        # Hardcode X limits so the graph renders correctly even if the first/last region is empty
        ax.set_xlim(0.5, len(regions) + 0.5)
        
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {g}")
        
    fig.suptitle(title)
    plt.tight_layout()  # Ensures X-axis labels aren't cut off
    save_fig(fig, path)


def main():
    ap = argparse.ArgumentParser(description="A6 speaking rate & pitch (F0)")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))

    a6 = cfg["a6_speaking_rate_pitch"]
    top_db = a6["top_db"]
    max_per_speaker = a6["max_per_speaker"]
    compute_f0 = a6["compute_f0"]
    pitch_ranges = cfg.pitch_ranges

    df = build_manifest(cfg.data_root, cfg.bpac_subdir, cfg.sentences_csv, verbose=cfg.verbose)
    if df.empty:
        raise SystemExit("No utterances found — check paths.data_root in config.yml.")
    df = df.dropna(subset=["sentence"])

    if max_per_speaker:
        df = (df.sort_values(["speaker_uid", "sentence_id"])
                .groupby("speaker_uid", group_keys=False)
                .head(max_per_speaker).reset_index(drop=True))

    df["n_words"] = df["sentence"].map(lambda s: len(tokenize(s)))
    df["n_syllables"] = df["sentence"].map(syllable_count_bangla)

    rows = []
    for r in maybe_tqdm(df.itertuples(index=False), desc="A6 audio", total=len(df)):
        dur = speech_duration_seconds(r.filepath, sr=cfg.analysis_sr, top_db=top_db)
        wpm = (r.n_words / dur * 60.0) if dur and dur > 0 else np.nan
        sps = (r.n_syllables / dur) if dur and dur > 0 else np.nan
        rec = {"speech_dur_s": dur, "wpm": wpm, "syll_per_s": sps}
        if compute_f0:
            rec.update(mean_f0(r.filepath, gender=r.gender, sr=cfg.analysis_sr,
                               top_db=top_db, pitch_ranges=pitch_ranges))
        rows.append(rec)
    metrics = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    base_cols = ["utterance_id", "region", "gender", "speaker_uid", "sentence_id",
                 "n_words", "n_syllables", "speech_dur_s", "wpm", "syll_per_s"]
    f0_cols = [] if not compute_f0 else ["f0_mean", "f0_sd", "voiced_fraction", "f0_method"]
    save_table(metrics[base_cols + f0_cols], out / "utterance_metrics.csv", index=False)

    agg_cols = {"wpm": ["mean", "std"], "syll_per_s": ["mean", "std"], "speech_dur_s": ["mean"]}
    if compute_f0:
        agg_cols["f0_mean"] = ["mean", "std"]
    rg = metrics.groupby(["region", "gender"]).agg(agg_cols)
    rg.columns = ["_".join(c) for c in rg.columns]
    rg = rg.reset_index()
    save_table(rg, out / "per_region_gender.csv", index=False)

    _violin_by_region(metrics, "wpm", "A6 speaking rate (WPM)", "words/min", out / "wpm_by_region.png")
    if compute_f0:
        _violin_by_region(metrics, "f0_mean", "A6 mean F0", "Hz", out / "f0_by_region.png")

    summary = {
        "preprocessing": f"speech-only duration (top_db={top_db}); "
                         f"F0 per-gender floor/ceiling {pitch_ranges}",
        "n_utterances_used": int(len(metrics)),
        "max_per_speaker": max_per_speaker,
        "f0_computed": bool(compute_f0),
        "overall_wpm_mean": round(float(metrics["wpm"].mean(skipna=True)), 2),
        "overall_syll_per_s_mean": round(float(metrics["syll_per_s"].mean(skipna=True)), 3),
    }
    if compute_f0 and "f0_method" in metrics and not metrics.empty:
        summary["f0_method"] = metrics["f0_method"].mode().iat[0]
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "A6 — Speaking Rate & Pitch (F0)")
    rep.p(f"**Preprocessing:** WPM uses speech-only duration (silence trimmed, "
          f"top_db={top_db}); F0 uses per-gender floor/ceiling {pitch_ranges}.")
    if max_per_speaker:
        rep.p(f"> Quick pass: limited to {max_per_speaker} utterances/speaker. Set "
              "`a6_speaking_rate_pitch.max_per_speaker: null` in config.yml for final numbers.")
    rep.h2("Headline")
    rep.bullet(f"Utterances analyzed: {summary['n_utterances_used']}.")
    rep.bullet(f"Overall mean WPM: {summary['overall_wpm_mean']}; "
               f"mean syllables/s: {summary['overall_syll_per_s_mean']}.")
    rep.h2("Per region x gender"); rep.table(rg.round(2), index=False)
    rep.h2("Figures")
    rep.image("wpm_by_region.png", "Speaking rate (WPM) by region, faceted by gender.")
    if compute_f0:
        rep.image("f0_by_region.png", "Mean F0 by region, faceted by gender.")
    rep.h2("Critical caveat")
    rep.p("F0 is dominated by gender (male ≈ low, female ≈ high). **Read F0 region "
          "effects *within* each gender panel** — a pooled per-region F0 mean mostly "
          "reflects the male/female ratio. Per-utterance values: `utterance_metrics.csv`.")
    rep.write()

    print(f"[A6] done. utterances={summary['n_utterances_used']} "
          f"wpm_mean={summary['overall_wpm_mean']} f0={compute_f0}")


if __name__ == "__main__":
    main()