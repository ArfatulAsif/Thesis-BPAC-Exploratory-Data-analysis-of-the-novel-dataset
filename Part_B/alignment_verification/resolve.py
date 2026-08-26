#!/usr/bin/env python3
"""T4 -- resolve T3's flags into the canonical verified_manifest.csv.

Reads reports/alignment_resolution.csv (whatever has been reviewed so far --
may be empty) and applies it to the full raw manifest:
  shift:N -> corrects the sentence_id/sentence mapping for that utterance
             (metadata only; the on-disk file is never renamed)
  bad     -> excluded=True
  ok      -> needs_review cleared
  unsure  -> stays needs_review=True, not excluded

Utterances T3 never flagged are needs_review=False from the start. Anything
T3 flagged but nobody has reviewed yet passes through UNCHANGED
(excluded=False, needs_review=True) so downstream tasks can run on the
corpus as currently understood -- re-run this script after each review
session to tighten verified_manifest.csv.

Outputs: verified_manifest.csv (repo root -- the canonical Part D/E input),
reports/exclusions.csv, reports/reviewer_agreement.json, report.md, summary.json.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> Part_B
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from common.provenance import base_summary  # noqa: E402

from bpac_stt.manifest import build_manifest, load_sentences  # noqa: E402
from bpac_stt.reporting import ReportWriter, ensure_dir, save_json, save_table  # noqa: E402
from bpac_stt.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def cohen_kappa(a, b):
    """a, b: dict utterance_id -> decision, restricted to shared keys."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 5:
        return None, len(shared)
    from sklearn.metrics import cohen_kappa_score

    y1 = [a[k] for k in shared]
    y2 = [b[k] for k in shared]
    return float(cohen_kappa_score(y1, y2)), len(shared)


def main():
    ap = argparse.ArgumentParser(description="T4 resolve alignment flags into verified_manifest.csv")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(os.path.join(HERE, "reports"))

    worklist_path = os.path.join(HERE, "reports", "spotcheck_worklist.csv")
    resolution_path = os.path.join(HERE, "reports", "alignment_resolution.csv")
    if not os.path.exists(worklist_path):
        raise SystemExit("spotcheck_worklist.csv not found -- run analyze.py (T3) first.")

    worklist = pd.read_csv(worklist_path, encoding="utf-8-sig")
    flagged_ids = set(worklist["utterance_id"])

    if os.path.exists(resolution_path):
        resolution = pd.read_csv(resolution_path, encoding="utf-8-sig")
    else:
        resolution = pd.DataFrame(columns=["utterance_id", "reviewer", "decision", "shift_to", "timestamp"])
    n_resolution_rows = len(resolution)

    # One decision per utterance: prefer the latest timestamp if multiple reviewers/passes disagree.
    decided = pd.DataFrame(columns=["utterance_id", "decision", "shift_to"])
    if len(resolution):
        resolution_sorted = resolution.sort_values("timestamp")
        decided = resolution_sorted.groupby("utterance_id").last().reset_index()[
            ["utterance_id", "decision", "shift_to"]
        ]

    mani = build_manifest(cfg.data_root, cfg.bpac_subdir, cfg.sentences_csv, verbose=False)
    sentences = load_sentences(cfg.sentences_csv)
    n_total = len(mani)

    mani["needs_review"] = mani["utterance_id"].isin(flagged_ids)
    mani["excluded"] = False
    mani["exclusion_reason"] = ""
    mani["original_sentence_id"] = mani["sentence_id"]
    mani["corrected"] = False
    mani["review_decision"] = "unreviewed"
    mani.loc[~mani["needs_review"], "review_decision"] = "not_flagged"

    dec_map = decided.set_index("utterance_id")
    n_corrected, n_excluded_by_review, n_ok, n_unsure = 0, 0, 0, 0
    for uid, row in dec_map.iterrows():
        if uid not in mani.index.values and uid not in set(mani["utterance_id"]):
            continue
        mask = mani["utterance_id"] == uid
        decision = row["decision"]
        mani.loc[mask, "review_decision"] = decision
        if decision == "bad":
            mani.loc[mask, "excluded"] = True
            mani.loc[mask, "exclusion_reason"] = "reviewer_marked_bad"
            mani.loc[mask, "needs_review"] = False
            n_excluded_by_review += 1
        elif decision == "shift":
            new_id = int(row["shift_to"])
            mani.loc[mask, "sentence_id"] = new_id
            mani.loc[mask, "sentence"] = sentences.get(new_id, None)
            mani.loc[mask, "corrected"] = True
            mani.loc[mask, "needs_review"] = False
            n_corrected += 1
        elif decision == "ok":
            mani.loc[mask, "needs_review"] = False
            n_ok += 1
        elif decision == "unsure":
            mani.loc[mask, "needs_review"] = True
            n_unsure += 1

    n_still_pending = int(mani["needs_review"].sum())
    save_table(mani, os.path.join(_REPO_ROOT, "verified_manifest.csv"), index=False)
    save_table(mani[mani["excluded"]], out / "exclusions.csv", index=False)

    # -- reviewer agreement (Cohen's kappa) on any utterance reviewed by >=2 reviewers --
    kappa, n_shared = None, 0
    if len(resolution) and resolution["reviewer"].nunique() >= 2:
        reviewers = sorted(resolution["reviewer"].unique())[:2]
        r1 = resolution[resolution["reviewer"] == reviewers[0]].set_index("utterance_id")["decision"].to_dict()
        r2 = resolution[resolution["reviewer"] == reviewers[1]].set_index("utterance_id")["decision"].to_dict()
        kappa, n_shared = cohen_kappa(r1, r2)
    agreement = {
        "n_reviewers": int(resolution["reviewer"].nunique()) if len(resolution) else 0,
        "n_shared_utterances": n_shared,
        "cohen_kappa": kappa,
        "note": ("Need >=2 reviewers with overlapping utterance_ids (see spotcheck.py --sample 0.2) "
                 "for a real kappa; null until then.") if kappa is None else None,
    }
    save_json(agreement, out / "reviewer_agreement.json")

    verification_rate = 1 - (n_corrected + n_excluded_by_review) / n_total if n_total else float("nan")
    is_provisional = n_still_pending > 0

    summary = base_summary("T4_alignment_resolution", cfg.config_path, n_total, n_total)
    summary.update({
        "n_flagged_by_t3": len(flagged_ids),
        "n_resolution_rows": n_resolution_rows,
        "n_corrected": n_corrected,
        "n_excluded_by_review": n_excluded_by_review,
        "n_ok": n_ok,
        "n_unsure": n_unsure,
        "n_still_pending_review": n_still_pending,
        "alignment_verification_rate": round(verification_rate, 5),
        "provisional": is_provisional,
        "reviewer_agreement": agreement,
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T4 -- Alignment Resolution")
    if is_provisional:
        rep.p(
            f"**PROVISIONAL.** {n_still_pending}/{len(flagged_ids)} T3-flagged utterances have not been "
            "reviewed yet (`review_decision=unreviewed`) -- they pass through `verified_manifest.csv` "
            "unchanged (not excluded, not corrected) so downstream tasks can run on the corpus as "
            "currently understood. Run `spotcheck.py --reviewer <you>` and then re-run this script to "
            "tighten the numbers below. **Do not cite `alignment_verification_rate` in the paper until "
            "`n_still_pending_review` is 0.**"
        )
    else:
        rep.p(
            f"All {len(flagged_ids)} T3-flagged utterances have been reviewed. Final alignment "
            f"verification rate: **{verification_rate*100:.3f}%**."
        )
    rep.h2("Headline numbers")
    for k in ["n_flagged_by_t3", "n_resolution_rows", "n_corrected", "n_excluded_by_review",
              "n_ok", "n_unsure", "n_still_pending_review", "alignment_verification_rate", "provisional"]:
        rep.bullet(f"`{k}` = {summary[k]}")
    rep.h2("Reviewer agreement")
    rep.p(f"{agreement}")
    rep.h2("How to review")
    rep.p(
        "```\ncd Part_B/alignment_verification\n"
        "python spotcheck.py --reviewer <your_name>\n"
        "# optional second opinion on a 20% sample, for the kappa check:\n"
        "python spotcheck.py --reviewer <second_name> --sample 0.2\n"
        "python resolve.py --config ../config.yml\n```"
    )
    rep.h2("For the paper")
    rep.p(
        "S3.6: cite `alignment_verification_rate` as the corpus's alignment-verification rate ONLY once "
        "`provisional=false`. `verified_manifest.csv` (repo root) is the canonical manifest every "
        "downstream task (T5+, all of Part D) should read."
    )
    rep.write()

    print(f"[T4] done. provisional={is_provisional}  pending={n_still_pending}/{len(flagged_ids)}  "
          f"corrected={n_corrected}  excluded={n_excluded_by_review}  "
          f"rate={summary['alignment_verification_rate']}")


if __name__ == "__main__":
    main()
