#!/usr/bin/env python3
"""
mark_empty_files.py — find every empty / dead .wav in the BPAC corpus.

Place this file in the project ROOT (the folder that contains `data/`) and run:

    python mark_empty_files.py
    python mark_empty_files.py --workers 4
    python mark_empty_files.py --min-ms 50      # also flag clips shorter than 50 ms

A file is "empty/dead" if any of these hold:
  - it fails to open / decode                      (corrupt)
  - it has 0 audio frames                          (zero-length)
  - it is shorter than --min-ms milliseconds       (too short to be speech)
  - its peak amplitude is ~0 (pure digital silence) (silent)

It also tells you, for each empty file, whether it is a CONFORMING file
(name matches `sound_<NNN>.wav` AND its id is in best_diverse_sentences.csv) —
i.e. whether it was part of the scored B1/B2 set or one of the skipped files.

Outputs (next to this script, in ./empty_files_report/):
  - empty_files.csv          one row per empty/dead file (the actionable list)
  - empty_files_by_region.csv  counts per region (conforming vs non-conforming)
  - all_files_flagged.csv    every .wav with an `is_empty` column (full audit)
  - delete_empty_files.sh / .ps1   ready-to-review removal commands (NOT run)
  - report.md                human-readable summary

Nothing is modified or deleted. Only needs: numpy, pandas, soundfile, tqdm (opt).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SPEAKER_RE = re.compile(r"^speaker_(?P<gender>male|female)_(?P<id>\d+)_(?P<name>.+)$", re.IGNORECASE)
SOUND_RE = re.compile(r"^sound_(?P<num>\d+)\.wav$", re.IGNORECASE)


def load_sentence_ids(data_root: Path):
    """Return the set of valid sentence ids from the transcript CSV (or None)."""
    csv = data_root / "best_diverse_sentences.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv, encoding="utf-8-sig")
        cols = {c.lower(): c for c in df.columns}
        return set(int(x) for x in df[cols["id"]].tolist())
    except Exception:
        return None


def find_wavs(bpac_dir: Path):
    rows = []
    for region_dir in sorted(p for p in bpac_dir.iterdir() if p.is_dir()):
        region = region_dir.name.lower()
        for spk_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            m = SPEAKER_RE.match(spk_dir.name)
            speaker = spk_dir.name
            for wav in sorted(spk_dir.glob("*.wav")):
                sm = SOUND_RE.match(wav.name)
                rows.append({
                    "filepath": str(wav),
                    "filename": wav.name,
                    "region": region,
                    "speaker_dir": speaker,
                    "conforming_name": bool(sm),
                    "sentence_id": int(sm.group("num")) if sm else None,
                })
    return pd.DataFrame(rows)


def inspect(path: str, min_samples_getter):
    """Return (is_empty, reason, n_samples, duration_s, peak, sample_rate)."""
    import soundfile as sf

    try:
        info = sf.info(path)
        sr = info.samplerate
    except Exception as e:
        return True, f"open_failed: {str(e)[:120]}", 0, 0.0, 0.0, None
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        return True, f"decode_failed: {str(e)[:120]}", 0, 0.0, 0.0, sr
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = int(y.size)
    if n == 0:
        return True, "zero_length", 0, 0.0, 0.0, sr
    dur = n / float(sr or 1)
    peak = float(np.max(np.abs(y)))
    min_samples = min_samples_getter(sr)
    if n < min_samples:
        return True, f"too_short(<{min_samples}smp)", n, dur, peak, sr
    if peak <= 1e-5:
        return True, "silent(peak~0)", n, dur, peak, sr
    return False, "ok", n, dur, peak, sr


def main():
    ap = argparse.ArgumentParser(description="Find empty/dead wav files in the BPAC corpus.")
    ap.add_argument("--data-root", default="data", help="Folder containing BPAC/ (default: data)")
    ap.add_argument("--bpac-subdir", default="BPAC")
    ap.add_argument("--min-ms", type=float, default=0.0,
                    help="Also flag clips shorter than this many ms (default 0 = only zero-length).")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1).")
    ap.add_argument("--out", default=None, help="Output dir (default ./empty_files_report)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    bpac = data_root / args.bpac_subdir
    if not bpac.is_dir():
        raise SystemExit(f"BPAC folder not found: {bpac}  (run from the project root, "
                         f"or pass --data-root)")
    out = Path(args.out) if args.out else here / "empty_files_report"
    out.mkdir(parents=True, exist_ok=True)

    valid_ids = load_sentence_ids(data_root)
    files = find_wavs(bpac)
    if files.empty:
        raise SystemExit(f"No wavs found under {bpac}")
    print(f"[scan] {len(files)} wav files across {files['region'].nunique()} regions")

    def min_samples_for(sr):
        return int(round((args.min_ms / 1000.0) * (sr or 16000)))

    paths = files["filepath"].tolist()
    try:
        from tqdm import tqdm
        it = tqdm(paths, desc="checking")
    except Exception:
        it = paths

    if args.workers and args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        fn = partial(inspect, min_samples_getter=min_samples_for)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(fn, paths, chunksize=32))
    else:
        res = [inspect(p, min_samples_for) for p in it]

    flagged = pd.DataFrame(res, columns=["is_empty", "reason", "n_samples",
                                         "duration_s", "peak", "sample_rate"])
    df = pd.concat([files.reset_index(drop=True), flagged], axis=1)

    # Was this file part of the scored set? (conforming name AND id in the CSV)
    if valid_ids is not None:
        df["in_scored_set"] = df["conforming_name"] & df["sentence_id"].isin(valid_ids)
    else:
        df["in_scored_set"] = df["conforming_name"]  # best effort if CSV missing

    df.to_csv(out / "all_files_flagged.csv", index=False, encoding="utf-8-sig")

    empties = df[df["is_empty"]].copy()
    empties = empties.sort_values(["in_scored_set", "region", "speaker_dir", "filename"],
                                  ascending=[False, True, True, True])
    empties.to_csv(out / "empty_files.csv", index=False, encoding="utf-8-sig")

    # Per-region breakdown
    def region_counts(g):
        return pd.Series({
            "n_files": len(g),
            "n_empty": int(g["is_empty"].sum()),
            "n_empty_scored": int((g["is_empty"] & g["in_scored_set"]).sum()),
            "n_empty_nonconforming": int((g["is_empty"] & ~g["in_scored_set"]).sum()),
            "pct_empty": round(100.0 * g["is_empty"].mean(), 2),
        })

    by_region = df.groupby("region").apply(region_counts).reset_index().sort_values(
        "n_empty", ascending=False)
    by_region.to_csv(out / "empty_files_by_region.csv", index=False, encoding="utf-8-sig")

    reason_counts = empties["reason"].str.replace(r"\(.*\)", "", regex=True).value_counts()

    # Ready-to-review removal scripts (NOT executed). Only target SCORED empties
    # by default — review before running either of these.
    scored_empty_paths = empties[empties["in_scored_set"]]["filepath"].tolist()
    sh = ["#!/usr/bin/env bash",
          "# REVIEW BEFORE RUNNING. Removes empty/dead files that were in the scored set.",
          "# (Non-conforming empties are listed in empty_files.csv but not removed here.)",
          "set -e"]
    ps = ["# REVIEW BEFORE RUNNING (PowerShell).",
          "# Removes empty/dead files that were in the scored set."]
    for p in scored_empty_paths:
        sh.append(f'rm -f "{p}"')
        ps.append(f'Remove-Item -LiteralPath "{p}" -Force')
    (out / "delete_empty_files.sh").write_text("\n".join(sh) + "\n", encoding="utf-8")
    (out / "delete_empty_files.ps1").write_text("\n".join(ps) + "\n", encoding="utf-8")

    # Report
    lines = ["# Empty / dead .wav files — inventory", ""]
    lines += [f"Scanned **{len(df)}** files. Found **{len(empties)}** empty/dead "
              f"({round(100*len(empties)/len(df),2)}%).", ""]
    lines += [f"- In the **scored** set (conforming name + valid sentence id): "
              f"**{int(empties['in_scored_set'].sum())}**  ← these affected B1/B2",
              f"- Non-conforming / not scored: **{int((~empties['in_scored_set']).sum())}**", ""]
    lines += ["## Reasons", "", reason_counts.to_frame("count").to_markdown(), ""]
    lines += ["## By region", "", by_region.to_markdown(index=False), ""]
    lines += ["## What to do",
              "- `empty_files.csv` is the actionable list (scored empties first).",
              "- **Empties in the scored set** are the ones that hurt your B1/B2 numbers "
              "(blank hypothesis vs a real reference = ~100% error for that clip). Either "
              "re-record them, or exclude them and re-run B1/B2.",
              "- `delete_empty_files.sh` / `.ps1` contain ready removal commands for the "
              "**scored** empties only — **review them before running**; nothing was deleted.",
              "- Non-conforming empties were never scored, but their count per region tells "
              "you which recording sessions produced the most failed captures.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # Console digest
    print("\n=== empty/dead files by region ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(by_region.to_string(index=False))
    print(f"\nTotal empty/dead: {len(empties)}  "
          f"(scored: {int(empties['in_scored_set'].sum())}, "
          f"non-conforming: {int((~empties['in_scored_set']).sum())})")
    print("\nReasons:")
    print(reason_counts.to_string())
    print(f"\nWrote inventory to: {out}")
    print("  - empty_files.csv            (actionable list)")
    print("  - empty_files_by_region.csv  (per-region counts)")
    print("  - delete_empty_files.sh/.ps1 (review before running)")


if __name__ == "__main__":
    main()
