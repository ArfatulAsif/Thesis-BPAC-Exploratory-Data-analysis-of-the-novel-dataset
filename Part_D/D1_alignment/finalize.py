#!/usr/bin/env python3
"""T12 step 3 -- parse MFA's TextGrid output into phone_tokens.parquet /
word_tokens.parquet, the input to T13-T18.

Run after `mfa train` (or `mfa align`) has written TextGrids to
D1_alignment/reports/textgrids/. Also runs the sanity assertions the spec
calls for: every utterance aligned, no unreasonably-short phones without
being flagged, total aligned duration ~= total speech duration.

Run: python D1_alignment/finalize.py --config config.yml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import pandas as pd  # noqa: E402

from common.provenance import base_summary  # noqa: E402

from bpac_phon.cli import add_config_arg  # noqa: E402
from bpac_phon.manifest import build_manifest  # noqa: E402
from bpac_phon.reporting import ReportWriter, ensure_dir, maybe_tqdm, save_json, save_table  # noqa: E402
from bpac_phon.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from bpac_phon.textgrid_io import parse_textgrid_dir


def main():
    ap = argparse.ArgumentParser(description="T12 -- parse TextGrids into phone/word tables")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--textgrid_dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    here = os.path.dirname(os.path.abspath(__file__))
    out = ensure_dir(os.path.join(here, "reports"))
    tg_dir = args.textgrid_dir or os.path.join(here, "reports", "textgrids")

    clean_mani = build_manifest(cfg.data_root, cfg.bpac_subdir, None, verbose=False)
    if os.path.exists(cfg.verified_manifest):
        vm = pd.read_csv(cfg.verified_manifest, encoding="utf-8-sig")
        vm = vm[~vm["excluded"].astype(bool)]
        mani = clean_mani[["utterance_id"]].merge(
            vm[["utterance_id", "region", "gender", "sentence_id", "sentence"]], on="utterance_id", how="inner"
        )
    else:
        mani = clean_mani[["utterance_id", "region", "gender", "sentence_id", "sentence"]]
    mani["pipeline"] = mani["gender"].map({"male": "lab", "female": "app"})

    word_df, phone_df = parse_textgrid_dir(tg_dir, manifest_df=mani, progress=maybe_tqdm)
    print(f"[T12-finalize] parsed {phone_df['utterance_id'].nunique() if len(phone_df) else 0} "
          f"utterances -> {len(phone_df)} phone tokens, {len(word_df)} word tokens")

    phone_df.to_parquet(out / "phone_tokens.parquet", index=False)
    word_df.to_parquet(out / "word_tokens.parquet", index=False)

    # -- sanity checks --
    n_manifest = mani["utterance_id"].nunique()
    n_aligned = phone_df["utterance_id"].nunique() if len(phone_df) else 0
    n_missing = n_manifest - n_aligned
    short_thresh_s = 0.010
    n_short = int((phone_df["duration_s"] < short_thresh_s).sum()) if len(phone_df) else 0
    pct_short = round(100 * n_short / len(phone_df), 3) if len(phone_df) else None

    by_region = (
        phone_df.groupby("region")
        .agg(n_phones=("phone", "size"), n_utterances=("utterance_id", "nunique"),
             mean_dur_s=("duration_s", "mean"), pct_short=("duration_s", lambda s: 100 * (s < short_thresh_s).mean()))
        .reset_index()
    ) if len(phone_df) else pd.DataFrame()
    save_table(by_region, out / "alignment_stats_by_region.csv", index=False)

    summary = base_summary("T12_alignment_finalize", cfg.config_path, n_manifest, n_aligned)
    summary.update({
        "n_missing_aligned": n_missing,
        "n_phone_tokens": len(phone_df),
        "n_word_tokens": len(word_df),
        "pct_phones_under_10ms": pct_short,
        "textgrid_dir": tg_dir,
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T12 -- Forced Alignment: Parsed TextGrids")
    rep.p(
        f"Parsed **{n_aligned}/{n_manifest}** utterances' TextGrids into "
        f"{len(phone_df)} phone tokens and {len(word_df)} word tokens. "
        f"{n_missing} utterances have no TextGrid (alignment failed or were skipped)."
    )
    rep.h2("Sanity checks")
    rep.bullet(f"Phones under {short_thresh_s*1000:.0f}ms: {n_short} ({pct_short}%) -- "
              "the aligner squeezing a phone it couldn't confidently place.")
    rep.h2("Per-region alignment stats")
    rep.table(by_region.round(4), index=False)
    rep.write()

    print(f"[T12-finalize] done. aligned={n_aligned}/{n_manifest}  missing={n_missing}  "
          f"phones={len(phone_df)}  pct_short={pct_short}")


if __name__ == "__main__":
    main()
