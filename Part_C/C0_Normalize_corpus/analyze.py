#!/usr/bin/env python3
"""C0 — Normalize the corpus + data diagnostics.

Writes a CLEAN copy of the corpus (uniform 16 kHz mono, empty/dead files
dropped, optional peak normalization) to paths.data_root, removing the
recording-format confound before C1/C2. Also reports per-region empty/dead
breakdown and a duration-based parallel-alignment check (with focus on
Rajshahi/Dhaka).

T5 (BPAC Paper 1): pass --manifest verified_manifest.csv to re-freeze the
clean corpus from the Phase-1-verified manifest instead of a fresh raw
directory walk -- excluded=True rows are dropped and T4-corrected
sentence_id mappings are honored. Also emits checksums.csv and
CORPUS_FROZEN.json, the reproducibility anchors everything downstream reads.

Run:  python C0_Normalize_corpus/analyze.py [--manifest ../../verified_manifest.csv]
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

_PART_C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PART_C_DIR)
_REPO_ROOT = os.path.dirname(_PART_C_DIR)

from bpac_accent.cli import add_config_arg  # noqa: E402
from bpac_accent.normalize import (alignment_check, compute_checksums,  # noqa: E402
                                   normalize_corpus, per_region_summary)
from bpac_accent.reporting import (ReportWriter, ensure_dir, maybe_tqdm,  # noqa: E402
                                   save_json, save_table)
from bpac_accent.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="C0 normalize corpus + diagnostics")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--manifest", default=None,
                     help="T5: path to verified_manifest.csv. If given, re-freezes data_clean/ "
                          "from this manifest (excluded=True rows dropped, corrected sentence_id "
                          "honored) instead of a fresh raw directory walk.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    nz = cfg["normalize"]

    manifest_override = None
    n_excluded_by_manifest = 0
    if args.manifest:
        vm = pd.read_csv(args.manifest, encoding="utf-8-sig")
        n_excluded_by_manifest = int(vm["excluded"].sum()) if "excluded" in vm.columns else 0
        if "excluded" in vm.columns:
            vm = vm[~vm["excluded"].astype(bool)].reset_index(drop=True)
        manifest_override = vm
        print(f"[C0/T5] using verified manifest {args.manifest}: "
              f"{len(vm)} rows kept, {n_excluded_by_manifest} excluded")

    df, _ = normalize_corpus(cfg, manifest_override=manifest_override)
    save_table(df, out / "normalization_log.csv", index=False)

    summary = per_region_summary(df)
    save_table(summary, out / "per_region_summary.csv", index=False)

    by_region, cell = alignment_check(df, tol=nz.get("align_dur_tol", 0.40),
                                      focus=nz.get("rajshahi_focus"))
    if len(by_region):
        save_table(by_region, out / "alignment_by_region.csv", index=False)
        save_table(cell[cell["suspect"]].sort_values("rel_dev", ascending=False),
                   out / "alignment_suspect_cells.csv", index=False)

    totals = {
        "n_input": int(len(df)),
        "n_written": int((df["status"] == "written").sum()),
        "n_exists_skipped": int((df["status"] == "exists_skipped").sum()),
        "n_dead_skipped": int((df["status"] == "dead_skipped").sum()),
        "n_load_failed": int((df["status"] == "load_failed").sum()),
        "n_excluded_by_verified_manifest": n_excluded_by_manifest,
        "clean_root": str(cfg.data_root),
        "target_sr": nz.get("target_sr", 16000),
        "used_verified_manifest": bool(args.manifest),
    }

    checksums = None
    if args.manifest:
        # T5: checksums.csv + CORPUS_FROZEN.json -- the reproducibility anchors.
        import hashlib

        clean_bpac = os.path.join(str(cfg.data_root), cfg.bpac_subdir)
        checksums = compute_checksums(clean_bpac, progress=maybe_tqdm)
        save_table(checksums, out / "checksums.csv", index=False)
        combined = "".join(sorted(checksums["sha256"])).encode("utf-8")
        checksum_of_checksums = hashlib.sha256(combined).hexdigest()
        frozen = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_files": int(len(checksums)),
            "total_bytes": int(checksums["size_bytes"].sum()) if len(checksums) else 0,
            "checksum_of_checksums": checksum_of_checksums,
            "source_manifest": os.path.abspath(args.manifest),
            "clean_root": str(cfg.data_root),
        }
        frozen_path = os.path.join(_REPO_ROOT, "CORPUS_FROZEN.json")
        save_json(frozen, frozen_path)
        totals["checksum_of_checksums"] = checksum_of_checksums
        totals["corpus_frozen_path"] = frozen_path

    save_json(totals, out / "summary.json")

    rep = ReportWriter(out, "C0 — Corpus Normalization & Diagnostics")
    rep.p(f"Wrote a clean 16 kHz mono corpus to `{cfg.data_root}` "
          f"(peak-normalized={nz.get('peak_normalize', True)}). C1/C2 read this tree.")
    rep.h2("Normalization totals")
    rep.bullet(f"Input clips: **{totals['n_input']}**; written: **{totals['n_written']}**; "
               f"already present: {totals['n_exists_skipped']}; "
               f"dead/empty dropped: **{totals['n_dead_skipped']}**; "
               f"load failures: {totals['n_load_failed']}.")
    if args.manifest:
        rep.h2("T5 -- re-freeze from verified_manifest.csv")
        rep.p(
            f"Re-normalized from `{args.manifest}` (Phase 1): "
            f"**{totals['n_excluded_by_verified_manifest']}** utterances excluded by T4 review "
            "(vs. the plain dead-file drop above), corrected `sentence_id` mappings honored. "
            f"See Part_B/alignment_verification/reports/report.md for the ASR-based alignment "
            "verification rate (T3/T4) that this corpus reflects -- **check whether that rate is "
            "still `provisional` before citing this freeze as final.**"
        )
        rep.bullet(f"`checksums.csv`: sha256 for all **{totals.get('n_files', len(checksums) if checksums is not None else 0)}** files in `data_clean/`.")
        rep.bullet(f"`CORPUS_FROZEN.json` -> `{totals.get('corpus_frozen_path')}` "
                   f"(checksum_of_checksums=`{totals.get('checksum_of_checksums', '')[:16]}...`).")
    rep.h2("Per-region summary")
    rep.table(summary, index=False)
    rep.h2("Parallel-alignment check (duration-based)")
    rep.p("For each shared sentence, clip durations should be similar across "
          "regions. A region with many duration-deviant sentences may have "
          "`sound_NNN.wav` not matching sentence id N. Higher `pct_suspect` = more "
          "alignment-suspect.")
    if len(by_region):
        rep.table(by_region, index=False)
        rep.bullet("Inspect `alignment_suspect_cells.csv` for the specific "
                   "(region, sentence) pairs; spot-check those files against the CSV.")
    else:
        rep.p("Not enough live files to run the alignment check.")
    rep.h2("Next")
    rep.bullet("Run C1 and C2 — they now read the clean corpus and avoid the "
               "format confound.")
    rep.write()

    print(f"[C0] written={totals['n_written']} dead_dropped={totals['n_dead_skipped']} "
          f"load_failed={totals['n_load_failed']} -> {cfg.data_root}")
    if len(by_region):
        worst = by_region.iloc[0]
        print(f"[C0] alignment: highest suspect = {worst['region']} "
              f"({worst['pct_suspect']}% of sentences)")


if __name__ == "__main__":
    main()
