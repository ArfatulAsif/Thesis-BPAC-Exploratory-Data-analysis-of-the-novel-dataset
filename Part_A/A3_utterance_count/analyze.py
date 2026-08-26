#!/usr/bin/env python3
"""A3 — Utterance counts (total / per region / per speaker / per gender).

All paths/params come from config.yml. Outputs (in ./reports): summary.json,
per_region_counts.csv, per_speaker_counts.csv, region_gender_counts.csv,
speakers_per_region.csv, per_region_counts.png, report.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from bpac_eda.cli import add_config_arg  # noqa: E402
from bpac_eda.manifest import build_manifest  # noqa: E402
from bpac_eda.reporting import (ReportWriter, ensure_dir, save_fig,  # noqa: E402
                                save_json, save_table)
from bpac_eda.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="A3 utterance counts")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    n_expected = cfg.n_sentences

    df = build_manifest(cfg.data_root, cfg.bpac_subdir, cfg.sentences_csv, verbose=cfg.verbose)
    if df.empty:
        raise SystemExit("No utterances found — check paths.data_root in config.yml.")

    per_region = df.groupby("region").size().reset_index(name="n_utt").sort_values("n_utt", ascending=False)
    per_gender = df.groupby("gender").size().reset_index(name="n_utt")
    region_gender = df.groupby(["region", "gender"]).size().unstack(fill_value=0)
    per_speaker = (
        df.groupby(["speaker_uid", "region", "gender"]).size().reset_index(name="n_utt")
        .sort_values(["region", "speaker_uid"])
    )
    per_speaker["expected"] = n_expected
    per_speaker["delta_vs_expected"] = per_speaker["n_utt"] - n_expected
    speakers_per_region = df.groupby("region")["speaker_uid"].nunique().reset_index(name="n_speakers")

    summary = {
        "total_utterances": int(len(df)),
        "n_speakers": int(df["speaker_uid"].nunique()),
        "n_regions": int(df["region"].nunique()),
        "expected_per_speaker": int(n_expected),
        "speakers_below_expected": int((per_speaker["n_utt"] < n_expected).sum()),
        "speakers_above_expected": int((per_speaker["n_utt"] > n_expected).sum()),
        "min_region_utt": int(per_region["n_utt"].min()),
        "max_region_utt": int(per_region["n_utt"].max()),
    }
    save_json(summary, out / "summary.json")
    save_table(per_region, out / "per_region_counts.csv", index=False)
    save_table(per_speaker, out / "per_speaker_counts.csv", index=False)
    save_table(region_gender, out / "region_gender_counts.csv")
    save_table(speakers_per_region, out / "speakers_per_region.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    region_gender.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("utterances")
    ax.set_title("A3 utterances per region (stacked by gender)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, out / "per_region_counts.png")

    rep = ReportWriter(out, "A3 — Utterance Counts")
    rep.h2("Headline")
    rep.bullet(f"Total utterances: **{summary['total_utterances']}** "
               f"({summary['n_speakers']} speakers x up to {n_expected} sentences).")
    rep.bullet(f"Speakers below expected: {summary['speakers_below_expected']}; "
               f"above: {summary['speakers_above_expected']}.")
    rep.h2("Speakers per region"); rep.table(speakers_per_region, index=False)
    rep.h2("Utterances per region"); rep.table(per_region, index=False)
    rep.h2("Region x gender"); rep.table(region_gender)
    rep.image("per_region_counts.png", "Utterances per region, stacked by gender.")
    rep.h2("Look for")
    rep.p("Uneven counts across regions, and speakers whose `delta_vs_expected` is "
          "non-zero (see `per_speaker_counts.csv`).")
    rep.write()

    print(f"[A3] done. total_utterances={summary['total_utterances']}")


if __name__ == "__main__":
    main()
