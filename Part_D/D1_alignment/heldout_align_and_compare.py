#!/usr/bin/env python3
"""T12 held-out-region robustness check, steps 2-3.

Step 2: align the Sylhet-only corpus using the acoustic model trained
WITHOUT Sylhet (bpac_mfa_acoustic_model_no_sylhet.zip, from
prepare_heldout_split.py + `mfa train`).

Step 3: compare the resulting phone boundaries against the original
full-corpus model's Sylhet boundaries (already in reports/textgrids/).
High agreement = the aligner learned genuine Bangla phonetics rather than
partially memorizing Sylhet-specific pronunciation; this is what makes the
phonetic measurements in D2-D7 trustworthy rather than circular.

Run (after `mfa train` on mfa_corpus_no_sylhet has produced the model zip):
    conda run -n aligner mfa align D1_alignment/mfa_corpus_sylhet_only \
        D0_lexicon/reports/bpac_lexicon.txt \
        D1_alignment/reports/bpac_mfa_acoustic_model_no_sylhet.zip \
        D1_alignment/reports/textgrids_sylhet_heldout --clean
    python D1_alignment/heldout_align_and_compare.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.provenance import base_summary  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FULL_TG_DIR = os.path.join(HERE, "reports", "textgrids")
HELDOUT_TG_DIR = os.path.join(HERE, "reports", "textgrids_sylhet_heldout")
OUT_DIR = os.path.join(HERE, "reports")


def parse_sylhet_textgrids(tg_dir, label):
    sys.path.insert(0, os.path.dirname(HERE))
    from bpac_phon.textgrid_io import parse_textgrid_dir

    from pathlib import Path
    files = sorted(Path(tg_dir).rglob("*.TextGrid"))
    sylhet_files = [f for f in files if f.parent.name.startswith("sylhet")]
    print(f"[{label}] {len(sylhet_files)} Sylhet TextGrids found in {tg_dir}")

    class _Files(list):
        pass
    # Reuse parse_one_textgrid directly per-file to avoid re-walking the whole dir
    from bpac_phon.textgrid_io import parse_one_textgrid
    all_phones = []
    for f in sylhet_files:
        try:
            _, phones = parse_one_textgrid(f)
            for p in phones:
                p["source"] = label
            all_phones.extend(phones)
        except Exception as e:
            print(f"  failed to parse {f}: {e}")
    return pd.DataFrame(all_phones)


def reindex_sequential(df):
    """phone_idx from textgrid_io.py is the RAW pre-silence-filter tier
    position, so it desyncs across two independently-aligned models
    whenever they disagree on inserting an optional silence/pause
    somewhere in an utterance (the phone CONTENT is fully determined by
    the shared lexicon+transcript and cannot itself differ; only where
    each model places pauses can). Renumbering to "nth retained (non-
    silence) phone in this utterance" makes the two models' sequences
    comparable position-by-position again. Done locally here rather than
    in textgrid_io.py so it cannot change phone_tokens.parquet or any of
    the already-finalized T13-T19 numbers, which use the original scheme."""
    df = df.sort_values(["utterance_id", "phone_idx"]).copy()
    df["seq_idx"] = df.groupby("utterance_id").cumcount()
    return df


def main():
    if not os.path.isdir(HELDOUT_TG_DIR):
        raise SystemExit(
            f"{HELDOUT_TG_DIR} not found -- run `mfa align` on "
            "mfa_corpus_sylhet_only with the no-Sylhet model first (see module docstring)."
        )

    full_df = parse_sylhet_textgrids(FULL_TG_DIR, "full_model")
    heldout_df = parse_sylhet_textgrids(HELDOUT_TG_DIR, "heldout_model")

    if not len(full_df) or not len(heldout_df):
        raise SystemExit("One of the two TextGrid sets produced 0 phone tokens -- check paths.")

    full_df = reindex_sequential(full_df)
    heldout_df = reindex_sequential(heldout_df)

    # match by (utterance_id, seq_idx) -- the "nth retained phone in this
    # utterance" scheme above, robust to differential silence insertion.
    full_k = full_df.set_index(["utterance_id", "seq_idx"])
    held_k = heldout_df.set_index(["utterance_id", "seq_idx"])
    common_idx = full_k.index.intersection(held_k.index)
    print(f"[compare] {len(common_idx)} matched (utterance_id, phone_idx) phone tokens")

    f = full_k.loc[common_idx]
    h = held_k.loc[common_idx]

    same_label = (f["phone"].values == h["phone"].values)
    start_diff_ms = np.abs(f["start_s"].values - h["start_s"].values) * 1000
    end_diff_ms = np.abs(f["end_s"].values - h["end_s"].values) * 1000
    boundary_diff_ms = np.concatenate([start_diff_ms, end_diff_ms])

    rows = pd.DataFrame({
        "utterance_id": [i[0] for i in common_idx],
        "phone_idx": [i[1] for i in common_idx],
        "phone_full": f["phone"].values,
        "phone_heldout": h["phone"].values,
        "same_phone_label": same_label,
        "start_diff_ms": start_diff_ms,
        "end_diff_ms": end_diff_ms,
    })
    rows.to_csv(os.path.join(OUT_DIR, "heldout_region_agreement.csv"), index=False, encoding="utf-8-sig")

    thresholds = [10, 20, 30, 50]
    pct_within = {f"pct_start_within_{t}ms": float(100 * np.mean(start_diff_ms <= t)) for t in thresholds}

    summary = base_summary(
        "T12_heldout_region_check", os.path.join(_REPO_ROOT, "Part_D", "config.yml"),
        len(full_df["utterance_id"].unique()), len(common_idx), count_unit="phone_tokens",
    )
    summary.update({
        "held_out_region": "sylhet",
        "n_sylhet_utterances_full_model": int(full_df["utterance_id"].nunique()),
        "n_sylhet_utterances_heldout_model": int(heldout_df["utterance_id"].nunique()),
        "n_matched_phone_tokens": int(len(common_idx)),
        "pct_same_phone_label": float(100 * np.mean(same_label)),
        "mean_start_boundary_diff_ms": float(np.mean(start_diff_ms)),
        "median_start_boundary_diff_ms": float(np.median(start_diff_ms)),
        "mean_end_boundary_diff_ms": float(np.mean(end_diff_ms)),
        **pct_within,
    })

    from bpac_phon.reporting import save_json, save_table
    save_table(rows.describe(), os.path.join(OUT_DIR, "heldout_region_agreement_stats.csv"))
    save_json(summary, os.path.join(OUT_DIR, "heldout_region_summary.json"))

    print(f"[heldout-compare] done. n_matched={len(common_idx)}  "
          f"pct_same_label={summary['pct_same_phone_label']:.2f}%  "
          f"mean_start_diff={summary['mean_start_boundary_diff_ms']:.2f}ms  "
          f"pct_within_20ms={summary['pct_start_within_20ms']:.2f}%")


if __name__ == "__main__":
    main()
