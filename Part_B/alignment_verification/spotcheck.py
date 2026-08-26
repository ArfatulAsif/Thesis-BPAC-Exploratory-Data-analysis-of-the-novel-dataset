#!/usr/bin/env python3
"""T4 -- interactive spot-check reviewer for T3's flagged worklist.

Walks `reports/spotcheck_worklist.csv`, plays each clip with the OS default
audio player (Windows `os.startfile`), shows reference/hypothesis/best-match
text, and records your decision. Resumable: rows already decided by a given
--reviewer are skipped on the next run.

Decisions:
  ok        - correctly aligned (ASR flag was accent/noise-driven)
  shift:N   - file actually contains sentence N (metadata-only fix, no rename)
  bad       - unusable (truncated, wrong speaker, corrupt)
  unsure    - flag for a second opinion
  s / skip  - leave undecided, move on
  q / quit  - save and exit

Usage:
    python spotcheck.py --reviewer arif
    python spotcheck.py --reviewer arif --sample 0.2   # for the second-reviewer 20% agreement check
"""
import argparse
import csv
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WORKLIST = HERE / "reports" / "spotcheck_worklist.csv"
RESOLUTION = HERE / "reports" / "alignment_resolution.csv"

VALID_DECISIONS = {"ok", "bad", "unsure"}


def play(path):
    try:
        os.startfile(path)  # noqa: S606 -- Windows only, opens default media player
    except Exception as e:
        print(f"  (could not auto-play: {e}. Open manually: {path})")


def load_resolution():
    if RESOLUTION.exists():
        return pd.read_csv(RESOLUTION, encoding="utf-8-sig")
    return pd.DataFrame(columns=["utterance_id", "reviewer", "decision", "shift_to", "timestamp"])


def append_resolution(row):
    is_new = not RESOLUTION.exists()
    with open(RESOLUTION, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["utterance_id", "reviewer", "decision", "shift_to", "timestamp"])
        if is_new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="T4 interactive alignment spot-check")
    ap.add_argument("--reviewer", required=True, help="Your reviewer id (used for the kappa agreement check).")
    ap.add_argument("--sample", type=float, default=None,
                     help="Review only a random fraction (e.g. 0.2 for a second reviewer's 20%% overlap sample).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not WORKLIST.exists():
        raise SystemExit(f"{WORKLIST} not found -- run analyze.py (T3) first.")

    worklist = pd.read_csv(WORKLIST, encoding="utf-8-sig")
    done = load_resolution()
    done_by_this_reviewer = set(done.loc[done["reviewer"] == args.reviewer, "utterance_id"])
    todo = worklist[~worklist["utterance_id"].isin(done_by_this_reviewer)]

    if args.sample:
        rng = random.Random(args.seed)
        idx = sorted(rng.sample(range(len(worklist)), max(1, int(len(worklist) * args.sample))))
        sample_ids = set(worklist.iloc[idx]["utterance_id"])
        todo = todo[todo["utterance_id"].isin(sample_ids)]

    print(f"[spotcheck] reviewer={args.reviewer}  {len(todo)}/{len(worklist)} rows left to review.")
    print("Decisions: ok | shift:N | bad | unsure | s(kip) | q(uit)\n")

    for _, row in todo.iterrows():
        print("=" * 78)
        print(f"utterance_id: {row['utterance_id']}   region={row['region']}   sentence_id={row['sentence_id']}")
        print(f"flags: offdiag={row.get('flag_offdiag')} highcer={row.get('flag_highcer')} "
              f"agree={row.get('flag_agree')} known_suspect={row.get('is_known_suspect_sentence')}")
        print(f"reference     : {row['reference']}")
        print(f"hyp (primary) : {row['hyp_primary']}")
        print(f"hyp (confirm) : {row['hyp_confirm']}")
        print(f"best match    : sentence {row['best_match_sentence_id_primary']} "
              f"(cer={row['best_match_cer_primary']:.3f}) -- {row.get('best_match_text_primary')}")
        print(f"duration      : {row.get('duration_s')}s")
        print(f"file          : {row['filepath']}")
        play(row["filepath"])

        ans = input("decision> ").strip().lower()
        if ans in ("q", "quit"):
            print("Saved. Exiting.")
            break
        if ans in ("s", "skip", ""):
            continue
        shift_to = None
        decision = ans
        if ans.startswith("shift:"):
            shift_to = ans.split(":", 1)[1].strip()
            decision = "shift"
        elif ans not in VALID_DECISIONS:
            print(f"  ! unrecognized '{ans}', skipping this row (re-run to retry).")
            continue

        append_resolution({
            "utterance_id": row["utterance_id"],
            "reviewer": args.reviewer,
            "decision": decision,
            "shift_to": shift_to or "",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    print(f"\n[spotcheck] resolutions saved to {RESOLUTION}")
    print("Run resolve.py next to build verified_manifest.csv from all recorded decisions.")


if __name__ == "__main__":
    main()
