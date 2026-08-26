#!/usr/bin/env python3
"""A1 — Unique sentences & parallel-coverage check.

All paths/params come from config.yml (see --config). Outputs (in ./reports):
summary.json, per_speaker_completeness.csv, missing_cells.csv,
coverage_heatmap.png, report.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bpac_eda.cli import add_config_arg  # noqa: E402
from bpac_eda.manifest import build_manifest, load_sentences  # noqa: E402
from bpac_eda.reporting import ReportWriter, ensure_dir, save_fig, save_json, save_table  # noqa: E402
from bpac_eda.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="A1 unique sentences & coverage")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))

    sentences = load_sentences(cfg.sentences_csv)
    n_ids = len(sentences)
    unique_texts = len(set(sentences.values()))
    dup_texts = n_ids - unique_texts

    df = build_manifest(cfg.data_root, cfg.bpac_subdir, cfg.sentences_csv, verbose=cfg.verbose)
    if df.empty:
        raise SystemExit("No utterances found — check paths.data_root in config.yml.")

    speakers = sorted(df["speaker_uid"].unique())
    all_ids = sorted(sentences.keys())
    present = df.groupby(["speaker_uid", "sentence_id"]).size().unstack(fill_value=0).gt(0)
    present = present.reindex(index=speakers, columns=all_ids, fill_value=False)

    completeness = pd.DataFrame({
        "speaker_uid": speakers,
        "region": [s.split("_")[0] for s in speakers],
        "n_present": present.sum(axis=1).values,
        "n_expected": len(all_ids),
    })
    completeness["n_missing"] = completeness["n_expected"] - completeness["n_present"]
    completeness["complete"] = completeness["n_missing"] == 0

    miss = present.eq(False)
    missing_cells = (
        miss.stack()[lambda s: s].reset_index()[["speaker_uid", "sentence_id"]]
        if miss.values.any() else pd.DataFrame(columns=["speaker_uid", "sentence_id"])
    )

    summary = {
        "sentences_in_csv": int(n_ids),
        "unique_sentence_texts": int(unique_texts),
        "duplicate_text_rows": int(dup_texts),
        "expected_sentences_per_speaker": int(cfg.n_sentences),
        "n_speakers": int(len(speakers)),
        "n_regions": int(df["region"].nunique()),
        "total_utterances": int(len(df)),
        "expected_if_full_parallel": int(len(speakers) * len(all_ids)),
        "speakers_complete": int(completeness["complete"].sum()),
        "speakers_incomplete": int((~completeness["complete"]).sum()),
        "total_missing_cells": int(missing_cells.shape[0]),
        "fully_parallel": bool(missing_cells.empty and dup_texts == 0),
    }

    save_json(summary, out / "summary.json")
    save_table(completeness, out / "per_speaker_completeness.csv", index=False)
    save_table(missing_cells, out / "missing_cells.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, max(4, len(speakers) * 0.22)))
    ax.imshow(present.values.astype(int), aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers, fontsize=6)
    ax.set_xlabel(f"sentence_id (1..{len(all_ids)})")
    ax.set_title("A1 parallel-coverage matrix (green = present)")
    save_fig(fig, out / "coverage_heatmap.png")

    rep = ReportWriter(out, "A1 — Unique Sentences & Parallel Coverage")
    rep.p("**Preprocessing:** Unicode NFC applied to all transcript text before counting.")
    rep.h2("Headline numbers")
    for k, v in summary.items():
        rep.bullet(f"`{k}` = {v}")
    rep.gap()
    rep.h2("Interpretation")
    if summary["fully_parallel"]:
        rep.p("Every speaker has every sentence and there are no duplicate sentence "
              "texts — the corpus is fully parallel as designed.")
    else:
        if summary["duplicate_text_rows"]:
            rep.p(f"{summary['duplicate_text_rows']} CSV rows share identical text — "
                  "the count of *unique* sentences is lower than the number of ids.")
        if summary["total_missing_cells"]:
            rep.p(f"{summary['total_missing_cells']} (speaker, sentence) cells are missing "
                  "— see `missing_cells.csv`. These break strict parallelism.")
    rep.h2("Per-speaker completeness")
    rep.p("Full table: `per_speaker_completeness.csv`. Incomplete speakers (if any):")
    inc = completeness[~completeness["complete"]]
    rep.table(inc if not inc.empty else pd.DataFrame({"note": ["all speakers complete"]}), index=False)
    rep.h2("Coverage matrix")
    rep.image("coverage_heatmap.png", "Speaker x sentence presence (green = present).")
    rep.write()

    print(f"[A1] done. fully_parallel={summary['fully_parallel']}  "
          f"speakers={summary['n_speakers']}  utterances={summary['total_utterances']}")


if __name__ == "__main__":
    main()
