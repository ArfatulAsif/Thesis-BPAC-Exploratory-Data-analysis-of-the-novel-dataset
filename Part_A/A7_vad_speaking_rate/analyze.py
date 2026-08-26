#!/usr/bin/env python3
"""T6 -- VAD-based speaking rate (fixes A6's energy-threshold contamination).

A6 trims silence with librosa.effects.trim(top_db=30), an energy heuristic
that can remove nearly all of a quiet recording, sending WPM to absurd
values (A6's corpus max is ~31,000 WPM). This replaces it with Silero VAD
(a trained voice-activity model) and adds sanity guards that exclude
implausible cells from the WPM aggregates instead of silently keeping them.

Reads verified_manifest.csv (Phase 1) for the corrected sentence_id/text
mapping and data_clean/ (frozen, T5) for audio -- NOTE: if verified_manifest
still has unreviewed T3 flags (see its `provisional` status), this run
reflects the corpus as currently understood, not the final post-review
state; re-run after T4 review completes.

Outputs (in ./reports): utterance_metrics.csv, per_region_gender_rate.csv,
excluded_outliers.csv, method_comparison.csv, figures/*, report.md, summary.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.provenance import base_summary  # noqa: E402

from bpac_eda.audio import load_vad_model, mean_f0, speech_duration_vad  # noqa: E402
from bpac_eda.cli import add_config_arg  # noqa: E402
from bpac_eda.manifest import build_manifest  # noqa: E402
from bpac_eda.reporting import (ReportWriter, ensure_dir, maybe_tqdm, save_fig,  # noqa: E402
                                save_json, save_table)
from bpac_eda.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from bpac_eda.textnorm import syllable_count_bangla, tokenize  # noqa: E402


def semitone_re_100hz(f0_hz):
    if f0_hz is None or not np.isfinite(f0_hz) or f0_hz <= 0:
        return np.nan
    return 12.0 * np.log2(f0_hz / 100.0)


def main():
    ap = argparse.ArgumentParser(description="T6 VAD-based speaking rate")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")
    a7 = cfg["a7_vad_speaking_rate"]
    pitch_ranges = cfg.pitch_ranges

    # -- clean-corpus manifest (filepaths) merged with verified metadata (corrected sentence_id/text) --
    clean_mani = build_manifest(cfg.data_clean_root, cfg.bpac_subdir, None, verbose=False)
    if os.path.exists(cfg.verified_manifest):
        vm = pd.read_csv(cfg.verified_manifest, encoding="utf-8-sig")
        vm = vm[~vm["excluded"].astype(bool)]
        df = clean_mani[["utterance_id", "filepath"]].merge(
            vm[["utterance_id", "region", "gender", "speaker_uid", "sentence_id", "sentence", "needs_review"]],
            on="utterance_id", how="inner",
        )
        provisional_note = bool(vm["needs_review"].any()) if "needs_review" in vm.columns else None
    else:
        print("[T6] WARNING verified_manifest.csv not found -- using data_clean/ manifest as-is (pre-Phase-1).")
        df = clean_mani
        provisional_note = None
    df = df.dropna(subset=["sentence"]).reset_index(drop=True)
    n_in = len(df)

    max_per_speaker = a7.get("max_per_speaker")
    if max_per_speaker:
        df = (df.sort_values(["speaker_uid", "sentence_id"])
                .groupby("speaker_uid", group_keys=False)
                .head(max_per_speaker).reset_index(drop=True))

    df["n_words"] = df["sentence"].map(lambda s: len(tokenize(s)))
    df["n_syllables"] = df["sentence"].map(syllable_count_bangla)

    vad_model = load_vad_model()
    threshold = float(a7.get("vad_threshold", 0.5))
    compute_f0 = bool(a7.get("compute_f0", True))

    rows = []
    for r in maybe_tqdm(df.itertuples(index=False), desc="T6 VAD", total=len(df)):
        vd = speech_duration_vad(r.filepath, model=vad_model, sr=16000, threshold=threshold)
        dur = vd["speech_dur_s"]
        wpm = (r.n_words / dur * 60.0) if dur and dur > 0 else np.nan
        sps = (r.n_syllables / dur) if dur and dur > 0 else np.nan
        rec = {"speech_dur_s": dur, "total_dur_s": vd["total_dur_s"], "speech_ratio": vd["speech_ratio"],
               "wpm": wpm, "articulation_rate_sps": sps}
        if compute_f0:
            f0 = mean_f0(r.filepath, gender=r.gender, sr=16000, pitch_ranges=pitch_ranges)
            f0["f0_semitone_re100"] = semitone_re_100hz(f0.get("f0_mean"))
            rec.update(f0)
        rows.append(rec)
    metrics = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    # -- sanity guards (T6): exclude from WPM aggregates, but keep the row for transparency --
    lo_wpm, hi_wpm = a7["wpm_valid_range"]
    metrics["exclude_reason"] = ""
    cond_dur = metrics["speech_dur_s"] < float(a7["min_speech_dur_s"])
    cond_ratio = (metrics["speech_ratio"] < float(a7["min_speech_ratio"])) | \
                 (metrics["speech_ratio"] > float(a7["max_speech_ratio"]))
    cond_wpm = ~metrics["wpm"].between(lo_wpm, hi_wpm)
    metrics.loc[cond_dur, "exclude_reason"] += "short_speech;"
    metrics.loc[cond_ratio, "exclude_reason"] += "bad_speech_ratio;"
    metrics.loc[cond_wpm, "exclude_reason"] += "wpm_out_of_range;"
    metrics["excluded_from_agg"] = metrics["exclude_reason"] != ""

    base_cols = ["utterance_id", "region", "gender", "speaker_uid", "sentence_id",
                 "n_words", "n_syllables", "speech_dur_s", "total_dur_s", "speech_ratio",
                 "wpm", "articulation_rate_sps", "excluded_from_agg", "exclude_reason"]
    f0_cols = [] if not compute_f0 else ["f0_mean", "f0_sd", "voiced_fraction", "f0_method", "f0_semitone_re100"]
    save_table(metrics[base_cols + f0_cols], out / "utterance_metrics.csv", index=False)

    excluded = metrics[metrics["excluded_from_agg"]]
    save_table(
        excluded.groupby(["region", "gender", "exclude_reason"]).size()
        .reset_index(name="n").sort_values("n", ascending=False),
        out / "excluded_outliers.csv", index=False,
    )

    used = metrics[~metrics["excluded_from_agg"]]
    agg_cols = {"wpm": ["mean", "std", "median"], "articulation_rate_sps": ["mean", "std"],
                "speech_dur_s": ["mean"], "speech_ratio": ["mean"]}
    if compute_f0:
        agg_cols["f0_mean"] = ["mean", "std"]
        agg_cols["f0_semitone_re100"] = ["mean", "std"]
    rg = used.groupby(["region", "gender"]).agg(agg_cols)
    rg.columns = ["_".join(c) for c in rg.columns]
    rg = rg.reset_index()
    save_table(rg, out / "per_region_gender_rate.csv", index=False)

    # -- before/after comparison against A6's already-computed top_db=30 metrics --
    method_comparison = None
    old_path = a7.get("old_method_metrics_csv")
    if old_path:
        old_path_abs = (out.parent / old_path).resolve() if not os.path.isabs(old_path) else old_path
        if os.path.exists(old_path_abs):
            old = pd.read_csv(old_path_abs)[["utterance_id", "speech_dur_s", "wpm"]].rename(
                columns={"speech_dur_s": "speech_dur_s_old_topdb30", "wpm": "wpm_old_topdb30"}
            )
            cmp = used.merge(old, on="utterance_id", how="inner")
            method_comparison = (
                cmp.groupby(["region", "gender"])
                .agg(wpm_new_vad_mean=("wpm", "mean"),
                     wpm_old_topdb30_mean=("wpm_old_topdb30", "mean"),
                     wpm_new_vad_median=("wpm", "median"),
                     wpm_old_topdb30_median=("wpm_old_topdb30", "median"))
                .reset_index()
            )
            method_comparison["wpm_mean_reduction_pct"] = (
                100 * (1 - method_comparison["wpm_new_vad_mean"] / method_comparison["wpm_old_topdb30_mean"])
            ).round(2)
            save_table(method_comparison, out / "method_comparison.csv", index=False)
        else:
            print(f"[T6] WARNING old-method comparison file not found: {old_path_abs}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, val, ylabel in [(axes[0], "wpm", "words/min (VAD)"),
                             (axes[1], "articulation_rate_sps", "syllables/s")]:
        regions = sorted(used["region"].unique())
        genders = sorted(used["gender"].dropna().unique())
        positions = range(len(regions))
        for g in genders:
            means = [used[(used["region"] == r) & (used["gender"] == g)][val].mean() for r in regions]
            ax.plot(positions, means, marker="o", label=g)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(regions, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle("T6 -- VAD-corrected speaking/articulation rate by region")
    plt.tight_layout()
    save_fig(fig, fig_dir / "wpm_by_region.png")
    fig.savefig(fig_dir / "wpm_by_region.pdf", bbox_inches="tight")
    plt.close(fig)

    if compute_f0:
        fig, ax = plt.subplots(figsize=(9, 5))
        regions = sorted(used["region"].unique())
        genders = sorted(used["gender"].dropna().unique())
        for g in genders:
            means = [used[(used["region"] == r) & (used["gender"] == g)]["f0_semitone_re100"].mean()
                     for r in regions]
            ax.plot(range(len(regions)), means, marker="o", label=g)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("semitones re 100 Hz")
        ax.set_title("T6 -- median F0 (semitones re 100 Hz) by region")
        ax.legend()
        save_fig(fig, fig_dir / "f0_semitone_by_region.png")
        fig.savefig(fig_dir / "f0_semitone_by_region.pdf", bbox_inches="tight")
        plt.close(fig)

    summary = base_summary("T6_A7_vad_speaking_rate", cfg.config_path, n_in, int(len(used)))
    summary.update({
        "n_excluded_from_agg": int(len(excluded)),
        "pct_excluded": round(100 * len(excluded) / len(metrics), 3) if len(metrics) else None,
        "overall_wpm_mean": round(float(used["wpm"].mean()), 2),
        "overall_wpm_median": round(float(used["wpm"].median()), 2),
        "overall_articulation_rate_mean": round(float(used["articulation_rate_sps"].mean()), 3),
        "vad_threshold": threshold,
        "guards": {"min_speech_dur_s": a7["min_speech_dur_s"], "min_speech_ratio": a7["min_speech_ratio"],
                   "max_speech_ratio": a7["max_speech_ratio"], "wpm_valid_range": a7["wpm_valid_range"]},
        "verified_manifest_still_has_pending_review": provisional_note,
    })
    if method_comparison is not None:
        summary["wpm_mean_reduction_pct_overall"] = round(
            100 * (1 - used["wpm"].mean() /
                   used.merge(pd.read_csv(old_path_abs)[["utterance_id", "wpm"]].rename(
                       columns={"wpm": "wpm_old"}), on="utterance_id")["wpm_old"].mean()), 2
        )
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T6 -- VAD-Based Speaking Rate")
    rep.p(
        f"A6's `librosa.effects.trim(top_db=30)` speech-duration heuristic explodes on quiet "
        f"recordings (WPM up to ~31,000 in that table). This replaces it with Silero VAD and "
        f"excludes implausible cells from the aggregates instead of silently keeping them: "
        f"**{summary['n_excluded_from_agg']}/{n_in}** utterances ({summary['pct_excluded']}%) are "
        f"excluded (short speech / bad speech ratio / WPM outside "
        f"{a7['wpm_valid_range']}). Overall mean WPM (VAD, cleaned): "
        f"**{summary['overall_wpm_mean']}** (median {summary['overall_wpm_median']})."
    )
    if provisional_note:
        rep.p(
            "**Note:** `verified_manifest.csv` still has unreviewed T3 alignment flags "
            "(`needs_review=True` rows exist) -- re-run this analysis after T4 review completes "
            "for the final numbers."
        )
    rep.h2("Headline numbers")
    for k in ["n_utterances_in", "n_utterances_used", "n_excluded_from_agg", "pct_excluded",
              "overall_wpm_mean", "overall_wpm_median", "overall_articulation_rate_mean"]:
        rep.bullet(f"`{k}` = {summary[k]}")
    rep.h2("Excluded outliers, by region x gender x reason")
    rep.table(pd.read_csv(out / "excluded_outliers.csv").head(30), index=False)
    if method_comparison is not None:
        rep.h2("Before / after: A6 (top_db=30) vs. T6 (VAD)")
        rep.p(
            f"Overall mean WPM dropped {summary.get('wpm_mean_reduction_pct_overall')}% after switching "
            "to VAD + guards -- direct evidence the old energy-trim heuristic was inflating WPM."
        )
        rep.table(method_comparison.round(2), index=False)
    rep.h2("Per region x gender (VAD-corrected)")
    rep.table(rg.round(3), index=False)
    rep.h2("Figures")
    rep.image("figures/wpm_by_region.png", "VAD-corrected WPM and articulation rate by region.")
    if compute_f0:
        rep.image("figures/f0_semitone_by_region.png", "Median F0 (semitones re 100 Hz) by region.")
    rep.h2("For the paper")
    rep.p(
        "S4 corpus statistics: this table replaces thesis Table 4.4. Cite `method_comparison.csv` as "
        "evidence for the correction, and `overall_articulation_rate_mean` (syllables/s) as the more "
        "cross-study-comparable rate metric alongside WPM."
    )
    rep.write()

    print(f"[T6] done. used={len(used)}/{n_in}  excluded={summary['n_excluded_from_agg']}  "
          f"wpm_mean={summary['overall_wpm_mean']}")


if __name__ == "__main__":
    main()
