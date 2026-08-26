#!/usr/bin/env python3
"""T8 -- global count-consistency audit.

Walks every `reports/summary.json` in the repo, extracts
`n_utterances_in`/`n_utterances_used` (+ speaker/region counts where
present), and reconciles them against `CORPUS_FROZEN.json` (T5) and
`verified_manifest.csv` (T4). Different analyses legitimately use different
subsets (T6 drops WPM outliers, T3 drops empty-audio rows, ...) -- the point
is to make every drop explicit and justified, not to force one N everywhere.

Exit code:
  0  -- every summary.json's n_utterances_in is one of the corpus's known
        canonical raw sizes, and n_used never exceeds n_in.
  1  -- a genuine inconsistency (n_used > n_in, or n_in matches no known
        canonical corpus size at all -- almost certainly a bug, not a
        legitimate subset).

Rows where n_used < n_in are reported (not failed) with a best-effort guess
at whether the drop is documented in that same summary.json, so a human can
scan `reconciliation.csv` for anything that looks unexplained.

Usage: python common/consistency_check.py   (wired into `make check`)
"""
import glob
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__)).rsplit(os.sep + "common", 1)[0]

DROP_EVIDENCE_KEYS = (
    "n_excluded", "n_excluded_from_agg", "n_flagged", "n_dead_skipped",
    "n_corrected", "n_excluded_by_review", "guards", "exclusion_reason",
    "n_excluded_by_verified_manifest", "max_per_speaker",
    "n_missing_aligned", "n_speakers_with_median_f0",
)


def find_summaries(root):
    return sorted(glob.glob(os.path.join(root, "**", "reports", "summary.json"), recursive=True))


def canonical_sizes(root):
    sizes = set()
    frozen_path = os.path.join(root, "CORPUS_FROZEN.json")
    if os.path.exists(frozen_path):
        frozen = json.load(open(frozen_path, encoding="utf-8"))
        sizes.add(frozen["n_files"])
    vm_path = os.path.join(root, "verified_manifest.csv")
    if os.path.exists(vm_path):
        import csv

        with open(vm_path, encoding="utf-8-sig") as f:
            n = sum(1 for _ in csv.reader(f)) - 1
        sizes.add(n)
    # hard-coded corpus facts (spec S0.3) as a last-resort fallback if the
    # files above don't exist yet (early Phase 0/1 runs)
    sizes.update({12750, 12747})
    # T12's aligned-utterance count is a legitimate downstream "full corpus"
    # size for every task after it in the pipeline (T13+) -- MFA alignment
    # can fail on a few files, and that drop is T12's to explain, not theirs.
    align_summary = os.path.join(root, "Part_D", "D1_alignment", "reports", "summary.json")
    if os.path.exists(align_summary):
        try:
            sizes.add(json.load(open(align_summary, encoding="utf-8"))["n_utterances_used"])
        except Exception:
            pass
    return sizes


def main():
    root = _REPO_ROOT
    canon = canonical_sizes(root)
    summaries = find_summaries(root)

    rows = []
    n_hard_fail = 0
    for path in summaries:
        rel = os.path.relpath(path, root)
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            rows.append({"report": rel, "task_id": None, "n_in": None, "n_used": None,
                        "status": f"UNREADABLE ({e})"})
            n_hard_fail += 1
            continue

        n_in = data.get("n_utterances_in")
        n_used = data.get("n_utterances_used")
        task_id = data.get("task_id", "(no task_id)")
        count_unit = data.get("count_unit", "utterances")

        if n_in is None or n_used is None:
            rows.append({"report": rel, "task_id": task_id, "n_in": n_in, "n_used": n_used,
                        "count_unit": count_unit,
                        "status": "SKIP (no n_utterances_in/used -- pre-T2-contract report)"})
            continue

        status = "ok"
        if n_used > n_in:
            status = "HARD_FAIL (n_used > n_in)"
            n_hard_fail += 1
        elif count_unit != "utterances":
            # Non-utterance unit (word types, phone/vowel/sibilant/VOT tokens, regions, ...) --
            # canonical corpus-size check doesn't apply; just record for traceability.
            status = f"ok (non-utterance unit: {count_unit})"
        elif n_in not in canon:
            status = f"HARD_FAIL (n_in={n_in} matches no known canonical corpus size {sorted(canon)})"
            n_hard_fail += 1
        elif n_used < n_in:
            has_evidence = any(k in data for k in DROP_EVIDENCE_KEYS)
            status = "explained_drop" if has_evidence else "UNEXPLAINED_DROP (review manually)"

        rows.append({"report": rel, "task_id": task_id, "n_in": n_in, "n_used": n_used,
                    "count_unit": count_unit, "status": status})

    # -- write reconciliation.csv --
    out_dir = os.path.join(root, "common", "T8_consistency")
    os.makedirs(out_dir, exist_ok=True)
    import csv as _csv

    with open(os.path.join(out_dir, "reconciliation.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=["report", "task_id", "n_in", "n_used", "count_unit", "status"])
        w.writeheader()
        w.writerows(rows)

    # -- corpus_counts.tex --
    n_verified = None
    vm_path = os.path.join(root, "verified_manifest.csv")
    n_speakers = 51
    if os.path.exists(vm_path):
        import pandas as pd

        vm = pd.read_csv(vm_path, encoding="utf-8-sig")
        n_verified = int((~vm["excluded"].astype(bool)).sum())
        n_speakers = int(vm["speaker_uid"].nunique())
    tex_dir = os.path.join(root, "paper", "tables")
    os.makedirs(tex_dir, exist_ok=True)
    with open(os.path.join(tex_dir, "corpus_counts.tex"), "w", encoding="utf-8") as f:
        f.write("% Auto-generated by common/consistency_check.py (T8). Do not hand-edit.\n")
        f.write(r"\newcommand{\NTotal}{12750}" + "\n")
        f.write(r"\newcommand{\NClean}{12747}" + "\n")
        f.write(f"\\newcommand{{\\NVerified}}{{{n_verified if n_verified is not None else '?'}}}\n")
        f.write(f"\\newcommand{{\\NSpeakers}}{{{n_speakers}}}\n")

    n_unexplained = sum(1 for r in rows if r["status"].startswith("UNEXPLAINED"))
    print(f"[T8] scanned {len(summaries)} summary.json files")
    print(f"[T8] hard failures: {n_hard_fail}")
    print(f"[T8] unexplained drops (review manually): {n_unexplained}")
    print(f"[T8] wrote {os.path.join(out_dir, 'reconciliation.csv')}")
    print(f"[T8] wrote {os.path.join(tex_dir, 'corpus_counts.tex')}")

    if n_hard_fail:
        print("[T8] FAILED -- see reconciliation.csv for HARD_FAIL rows.")
        sys.exit(1)
    print("[T8] OK.")


if __name__ == "__main__":
    main()
