#!/usr/bin/env python3
"""A2 — Audio length (total / per speaker / per region / per gender).

All paths/params come from config.yml (a2_audio_length.mode, .top_db). Outputs
(in ./reports): summary.json, per_region_duration.csv, per_speaker_duration.csv,
per_gender_duration.csv, region_gender_duration.csv, per_speaker_duration.png,
per_region_duration.png, report.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from bpac_eda.audio import speech_duration_seconds  # noqa: E402
from bpac_eda.cli import add_config_arg  # noqa: E402
from bpac_eda.manifest import build_manifest  # noqa: E402
from bpac_eda.reporting import (ReportWriter, ensure_dir, maybe_tqdm, save_fig,  # noqa: E402
                                save_json, save_table)
from bpac_eda.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def _hms(seconds):
    s = int(round(seconds))
    return f"{s // 3600}h {(s % 3600) // 60}m {s % 60}s"


def main():
    ap = argparse.ArgumentParser(description="A2 audio length")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))

    mode = cfg["a2_audio_length"]["mode"]
    top_db = cfg["a2_audio_length"]["top_db"]

    df = build_manifest(cfg.data_root, cfg.bpac_subdir, cfg.sentences_csv,
                        probe_durations=(mode == "raw"), verbose=cfg.verbose)
    if df.empty:
        raise SystemExit("No utterances found — check paths.data_root in config.yml.")

    if mode == "speech":
        df["duration_s"] = [
            speech_duration_seconds(p, sr=cfg.analysis_sr, top_db=top_db)
            for p in maybe_tqdm(df["filepath"], desc="speech-only durations")
        ]

    df = df.dropna(subset=["duration_s"])
    total_s = float(df["duration_s"].sum())

    per_region = (
        df.groupby("region")["duration_s"].agg(["sum", "mean", "count"]).reset_index()
        .rename(columns={"sum": "total_s", "mean": "mean_s", "count": "n_utt"})
        .sort_values("total_s", ascending=False)
    )
    per_region["total_min"] = per_region["total_s"] / 60.0
    per_gender = (
        df.groupby("gender")["duration_s"].agg(["sum", "mean", "count"]).reset_index()
        .rename(columns={"sum": "total_s", "mean": "mean_s", "count": "n_utt"})
    )
    region_gender_min = df.groupby(["region", "gender"])["duration_s"].sum().unstack(fill_value=0.0) / 60.0
    per_speaker = (
        df.groupby(["speaker_uid", "region", "gender"])["duration_s"]
        .agg(["sum", "mean", "count"]).reset_index()
        .rename(columns={"sum": "total_s", "mean": "mean_s", "count": "n_utt"})
        .sort_values(["region", "speaker_uid"])
    )
    per_speaker["total_min"] = per_speaker["total_s"] / 60.0

    summary = {
        "mode": mode,
        "top_db": top_db if mode == "speech" else None,
        "total_seconds": round(total_s, 2),
        "total_hms": _hms(total_s),
        "total_hours": round(total_s / 3600.0, 3),
        "n_utterances": int(len(df)),
        "n_speakers": int(df["speaker_uid"].nunique()),
        "n_regions": int(df["region"].nunique()),
        "mean_utt_seconds": round(float(df["duration_s"].mean()), 3),
        "min_speaker_total_min": round(float(per_speaker["total_min"].min()), 2),
        "max_speaker_total_min": round(float(per_speaker["total_min"].max()), 2),
    }
    save_json(summary, out / "summary.json")
    save_table(per_region, out / "per_region_duration.csv", index=False)
    save_table(per_gender, out / "per_gender_duration.csv", index=False)
    save_table(region_gender_min, out / "region_gender_duration.csv")
    save_table(per_speaker, out / "per_speaker_duration.csv", index=False)

    regions = sorted(df["region"].unique())
    cmap = plt.get_cmap("tab10")
    rcolor = {r: cmap(i % 10) for i, r in enumerate(regions)}
    ps = per_speaker.sort_values(["region", "total_min"])
    fig, ax = plt.subplots(figsize=(12, max(4, len(ps) * 0.22)))
    ax.barh(range(len(ps)), ps["total_min"].values, color=[rcolor[r] for r in ps["region"]])
    ax.set_yticks(range(len(ps)))
    ax.set_yticklabels(ps["speaker_uid"], fontsize=6)
    ax.set_xlabel("minutes")
    ax.set_title(f"A2 per-speaker audio length ({mode})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=rcolor[r]) for r in regions]
    ax.legend(handles, regions, fontsize=7, ncol=2, loc="lower right")
    save_fig(fig, out / "per_speaker_duration.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(per_region["region"], per_region["total_min"], color=[rcolor[r] for r in per_region["region"]])
    ax.set_ylabel("minutes")
    ax.set_title(f"A2 per-region audio length ({mode})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, out / "per_region_duration.png")

    rep = ReportWriter(out, "A2 — Audio Length")
    rep.p(f"**Duration mode:** `{mode}` "
          + ("(file-header duration, includes silence)." if mode == "raw"
             else f"(silence-trimmed, top_db={top_db})."))
    rep.h2("Headline")
    rep.bullet(f"Total: **{summary['total_hms']}** ({summary['total_hours']} h) across "
               f"{summary['n_utterances']} utterances, {summary['n_speakers']} speakers, "
               f"{summary['n_regions']} regions.")
    rep.bullet(f"Mean utterance: {summary['mean_utt_seconds']} s. Speaker totals range "
               f"{summary['min_speaker_total_min']}–{summary['max_speaker_total_min']} min.")
    rep.h2("Per region"); rep.table(per_region.round(2), index=False)
    rep.h2("Per gender"); rep.table(per_gender.round(2), index=False)
    rep.h2("Region x gender (minutes)"); rep.table(region_gender_min.round(2))
    rep.h2("Figures")
    rep.image("per_region_duration.png", "Total audio per region.")
    rep.image("per_speaker_duration.png", "Total audio per speaker (colored by region).")
    rep.h2("Look for")
    rep.p("Speakers or regions with markedly less audio — imbalance worsens the "
          "speaker–accent confound. Full table: `per_speaker_duration.csv`.")
    rep.write()

    print(f"[A2] done. total={summary['total_hms']} mode={mode}")


if __name__ == "__main__":
    main()
