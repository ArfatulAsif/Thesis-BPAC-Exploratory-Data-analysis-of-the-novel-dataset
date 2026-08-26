#!/usr/bin/env python3
"""T3 -- ASR-based parallel-alignment verification.

If sound_NNN.wav really contains sentence NNN, the ASR hypothesis should
resemble sentence NNN's text more than any other of the 250 script
sentences. This reuses Part_B's already-computed ASR predictions
(cache/predictions.csv) to test that directly, rather than relying on the
weaker duration-based proxy in Part_C/C0.

Method: for every utterance, compute character-error-rate (CER) between its
ASR hypothesis and ALL 250 script sentences (the "best-match search"), using
rapidfuzz's C-implemented Levenshtein distance via a single batched
edit-distance matrix (12,747 x 250 per model) rather than a python loop --
this makes the full 3.19M-comparison search run in seconds instead of hours,
so the trigram-Jaccard prefilter described in the original spec is not
needed for correctness (kept as a documented simplification).

Outputs (in ./reports): utterance_cer.csv, cell_flags.csv,
detector_agreement.csv, spotcheck_worklist.csv, figures/*, report.md,
summary.json.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> Part_B
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)  # -> repo root, for common.provenance

import matplotlib.pyplot as plt  # noqa: E402
from rapidfuzz import process  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from common.provenance import base_summary  # noqa: E402

from bpac_stt.manifest import build_manifest, load_sentences  # noqa: E402
from bpac_stt.reporting import ReportWriter, ensure_dir, save_fig, save_json, save_table  # noqa: E402
from bpac_stt.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from bpac_stt.textnorm import normalize_for_scoring  # noqa: E402


def cer_matrix_for_model(hyp_norm, candidate_texts):
    """Row i, col j = CER(hyp_norm[i], candidate_texts[j])."""
    dist = process.cdist(hyp_norm, candidate_texts, scorer=Levenshtein.distance, workers=-1)
    dist = dist.astype(np.float64)
    cand_len = np.array([max(len(c), 1) for c in candidate_texts], dtype=np.float64)
    return dist / cand_len[None, :]


def main():
    ap = argparse.ArgumentParser(description="T3 ASR-based parallel-alignment verification")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args()
    cfg = load_config(args.config)
    t3 = cfg.t3
    out = ensure_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"))
    fig_dir = ensure_dir(out / "figures")

    pred = pd.read_csv(cfg.predictions_cache, encoding="utf-8-sig")
    n_in = len(pred)
    pred = pred[~pred["is_empty_audio"].fillna(False).astype(bool)].reset_index(drop=True)
    n_used = len(pred)
    print(f"[T3] predictions: {n_in} -> {n_used} after dropping empty-audio rows")

    sentences = load_sentences(cfg.sentences_csv)
    candidate_ids = sorted(sentences.keys())
    candidate_texts = [normalize_for_scoring(sentences[i]) for i in candidate_ids]
    id_to_col = {sid: j for j, sid in enumerate(candidate_ids)}

    pred["reference_norm"] = pred["reference"].map(normalize_for_scoring)

    primary_key = t3["primary_model"]
    confirm_key = t3["confirm_model"]
    model_cols = {
        primary_key: f"hyp_{primary_key}",
        confirm_key: f"hyp_{confirm_key}",
    }

    per_model = {}
    for mkey, col in model_cols.items():
        hyp_norm = pred[col].fillna("").astype(str).map(normalize_for_scoring).tolist()
        cer_mat = cer_matrix_for_model(hyp_norm, candidate_texts)
        best_idx = cer_mat.argmin(axis=1)
        best_cer = cer_mat[np.arange(len(pred)), best_idx]
        own_col = pred["sentence_id"].map(id_to_col).to_numpy()
        self_cer = cer_mat[np.arange(len(pred)), own_col]
        best_sid = np.array(candidate_ids)[best_idx]
        per_model[mkey] = {
            "self_cer": self_cer,
            "best_match_sentence_id": best_sid,
            "best_match_cer": best_cer,
        }
        print(f"[T3] {mkey}: best-match search done "
              f"(median self_cer={np.median(self_cer):.3f}, median best_match_cer={np.median(best_cer):.3f})")

    df = pred[["utterance_id", "region", "gender", "speaker_uid", "sentence_id", "reference"]].copy()
    df["hyp_primary"] = pred[model_cols[primary_key]]
    df["hyp_confirm"] = pred[model_cols[confirm_key]]
    df["self_cer_primary"] = per_model[primary_key]["self_cer"]
    df["best_match_sentence_id_primary"] = per_model[primary_key]["best_match_sentence_id"]
    df["best_match_cer_primary"] = per_model[primary_key]["best_match_cer"]
    df["self_cer_confirm"] = per_model[confirm_key]["self_cer"]
    df["best_match_sentence_id_confirm"] = per_model[confirm_key]["best_match_sentence_id"]
    df["best_match_cer_confirm"] = per_model[confirm_key]["best_match_cer"]

    margin = float(t3["offdiag_cer_margin"])
    df["flag_offdiag"] = (
        (df["best_match_sentence_id_primary"] != df["sentence_id"])
        & ((df["self_cer_primary"] - df["best_match_cer_primary"]) > margin)
    )
    p = float(t3["highcer_percentile"])
    p99_primary = float(np.percentile(df["self_cer_primary"], p))
    p99_confirm = float(np.percentile(df["self_cer_confirm"], p))
    df["flag_highcer_primary"] = df["self_cer_primary"] > p99_primary
    df["flag_highcer_confirm"] = df["self_cer_confirm"] > p99_confirm
    df["flag_highcer"] = df["flag_highcer_primary"] | df["flag_highcer_confirm"]
    df["flag_agree"] = (
        (df["best_match_sentence_id_primary"] == df["best_match_sentence_id_confirm"])
        & (df["best_match_sentence_id_primary"] != df["sentence_id"])
    )
    df["flag_any"] = df["flag_offdiag"] | df["flag_highcer"] | df["flag_agree"]

    # -- attach filepath + duration for the human worklist --
    mani = build_manifest(cfg.data_root, cfg.bpac_subdir, cfg.sentences_csv,
                           probe_durations=True, verbose=False)
    df = df.merge(mani[["utterance_id", "filepath", "duration_s"]], on="utterance_id", how="left")

    save_table(df, out / "utterance_cer.csv", index=False)

    # -- cell aggregation --
    cell = (
        df.groupby(["region", "sentence_id"])
        .agg(n_speakers=("utterance_id", "size"),
             n_flagged=("flag_any", "sum"),
             n_offdiag=("flag_offdiag", "sum"),
             n_highcer=("flag_highcer", "sum"),
             n_agree=("flag_agree", "sum"),
             mean_self_cer_primary=("self_cer_primary", "mean"))
        .reset_index()
    )
    cell["pct_flagged"] = (cell["n_flagged"] / cell["n_speakers"] * 100).round(2)
    cell["cell_flagged"] = cell["n_flagged"] > 0
    save_table(cell, out / "cell_flags.csv", index=False)

    # -- cross-validate against C0's duration-based flags --
    dur_path = cfg.duration_suspect_cells
    agreement = None
    if dur_path and os.path.exists(dur_path):
        dur = pd.read_csv(dur_path)
        dur_cells = dur[["region", "sentence_id", "suspect"]].rename(columns={"suspect": "duration_flagged"})
        merged = cell[["region", "sentence_id", "cell_flagged"]].merge(
            dur_cells, on=["region", "sentence_id"], how="outer"
        )
        merged["cell_flagged"] = merged["cell_flagged"].fillna(False)
        merged["duration_flagged"] = merged["duration_flagged"].fillna(False)
        agreement = pd.crosstab(merged["cell_flagged"], merged["duration_flagged"],
                                 rownames=["asr_flagged"], colnames=["duration_flagged"])
        save_table(agreement, out / "detector_agreement.csv", index=True)
        both = int(((merged["cell_flagged"]) & (merged["duration_flagged"])).sum())
        either = int(((merged["cell_flagged"]) | (merged["duration_flagged"])).sum())
        agreement_rate = both / either if either else float("nan")
    else:
        print(f"[T3] WARNING duration_suspect_cells not found at {dur_path} -- skipping cross-validation")
        agreement_rate = None

    # -- spotcheck worklist: T3-flagged utterances first (highest-severity /
    # highest-agreement first -- these are actual evidence of a mismatch),
    # THEN known-suspect-sentence coverage fills any remaining budget.
    # (Previously sorted known-suspect-sentence first, which -- because that
    # set alone can exceed spotcheck_worklist_top_n -- silently crowded out
    # most genuinely-flagged utterances from ever being reviewed; found via
    # full-corpus audit 2026-07-29, see PROVENANCE.md.)
    known = set(int(x) for x in t3.get("known_suspect_sentences", []))
    df["severity"] = (df["self_cer_primary"] - df["best_match_cer_primary"]).clip(lower=0)
    df["is_known_suspect_sentence"] = df["sentence_id"].isin(known)
    worklist = df[df["flag_any"] | df["is_known_suspect_sentence"]].copy()
    worklist["best_match_text_primary"] = worklist["best_match_sentence_id_primary"].map(sentences)
    worklist = worklist.sort_values(
        ["flag_any", "flag_agree", "severity", "is_known_suspect_sentence"],
        ascending=[False, False, False, False],
    )
    top_n = int(t3.get("spotcheck_worklist_top_n", 300))
    worklist_cols = [
        "utterance_id", "filepath", "region", "speaker_uid", "sentence_id", "reference",
        "hyp_primary", "hyp_confirm", "best_match_sentence_id_primary", "best_match_text_primary",
        "best_match_cer_primary", "self_cer_primary", "duration_s",
        "flag_offdiag", "flag_highcer", "flag_agree", "is_known_suspect_sentence",
    ]
    save_table(worklist[worklist_cols].head(top_n), out / "spotcheck_worklist.csv", index=False)

    # -- figures --
    fig, ax = plt.subplots(figsize=(8, 5))
    for region, sub in df.groupby("region"):
        sub["self_cer_primary"].plot(kind="density", ax=ax, label=region)
    ax.set_xlim(0, min(2.0, df["self_cer_primary"].quantile(0.99) * 1.5))
    ax.set_xlabel(f"self CER ({primary_key})")
    ax.set_title("T3 -- self-CER distribution by region")
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, fig_dir / "cer_distribution_by_region.png")
    fig.savefig(fig_dir / "cer_distribution_by_region.pdf", bbox_inches="tight")
    plt.close(fig)

    conf = pd.crosstab(df["sentence_id"], df["best_match_sentence_id_primary"])
    conf = conf.reindex(index=candidate_ids, columns=candidate_ids, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.log1p(conf.values), cmap="viridis", aspect="auto")
    ax.set_xlabel("best-match sentence_id")
    ax.set_ylabel("intended sentence_id")
    ax.set_title("T3 -- intended vs. best-match sentence (log1p count)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_fig(fig, fig_dir / "best_match_confusion.png")
    fig.savefig(fig_dir / "best_match_confusion.pdf", bbox_inches="tight")
    plt.close(fig)

    n_flagged = int(df["flag_any"].sum())
    summary = base_summary("T3_alignment_verification", cfg.config_path, n_in, n_used)
    summary.update({
        "primary_model": primary_key,
        "confirm_model": confirm_key,
        "n_flagged": n_flagged,
        "pct_flagged": round(100 * n_flagged / n_used, 3),
        "n_cells_flagged": int(cell["cell_flagged"].sum()),
        "n_cells_total": int(len(cell)),
        "verified_alignment_rate": round(1 - n_flagged / n_used, 5),
        "agreement_with_duration_method": agreement_rate,
        "self_cer_p99_primary": p99_primary,
        "self_cer_p99_confirm": p99_confirm,
    })
    save_json(summary, out / "summary.json")

    # -- known-suspect-sentence spot table for the report --
    known_tbl = (
        df[df["sentence_id"].isin(known)]
        .groupby(["sentence_id", "region"])
        .agg(mean_self_cer=("self_cer_primary", "mean"),
             pct_offdiag=("flag_offdiag", "mean"),
             mean_best_match_cer=("best_match_cer_primary", "mean"))
        .reset_index()
    )
    save_table(known_tbl, out / "known_suspect_sentence_detail.csv", index=False)

    rep = ReportWriter(out, "T3 -- ASR-Based Parallel-Alignment Verification")
    rep.p(
        f"Every utterance's ASR hypothesis ({primary_key}, confirmed with {confirm_key}) was compared "
        "by character-error-rate against all 250 script sentences, not just its own reference. "
        f"{n_flagged}/{n_used} utterances ({summary['pct_flagged']}%) show a discrepancy consistent with "
        "misalignment (a different sentence fits the audio better than the intended one, or CER is an "
        "extreme outlier under both models). This gives a **verified alignment rate of "
        f"{summary['verified_alignment_rate']*100:.3f}%**, independent of C0's duration-based proxy."
    )
    rep.h2("Headline numbers")
    for k in ["n_utterances_in", "n_utterances_used", "n_flagged", "pct_flagged",
              "verified_alignment_rate", "n_cells_flagged", "n_cells_total",
              "agreement_with_duration_method"]:
        rep.bullet(f"`{k}` = {summary[k]}")
    rep.h2("Why this isn't just accent showing up as high CER")
    rep.p(
        "A strong regional accent also raises CER, so `flag_highcer` alone would over-flag. The decisive "
        "signal is `flag_offdiag`: it only fires when a *different* sentence fits the audio "
        f"better than the intended one by more than {margin} CER -- accent raises self CER but leaves the "
        "correct sentence as the best match; misalignment moves the best match elsewhere. `flag_agree` "
        "(both models pick the same off-diagonal sentence) is the strongest single signal and should be "
        "reviewed first in `spotcheck_worklist.csv`."
    )
    if agreement is not None:
        rep.h2("Cross-validation against the duration-based method (Part_C/C0)")
        rep.table(agreement, index=True)
    rep.h2("Known suspect sentences (from C0's duration check)")
    rep.p(
        "C0 flagged sentence 63 in rajshahi, dhaka, khulna, and rangpur simultaneously via duration alone "
        "-- when 4/8 regions deviate the same way, the corpus-wide median is the likely outlier, not the "
        "regions. The ASR-based self-CER / off-diagonal rate for these sentences, per region:"
    )
    rep.table(known_tbl, index=False)
    rep.h2("Figures")
    rep.image("figures/cer_distribution_by_region.png", "Self-CER density by region (primary model).")
    rep.image("figures/best_match_confusion.png", "Intended vs. best-match sentence id (log1p count).")
    rep.h2("For the paper")
    rep.p(
        f"S3.6 / S5.1: report the verified alignment rate ({summary['verified_alignment_rate']*100:.3f}%) "
        "as the headline reliability number, and cite `detector_agreement.csv` as evidence the ASR-based "
        "method is a stronger, independent check than the duration-based proxy alone. `spotcheck_worklist.csv` "
        "is the input to T4's human review."
    )
    rep.write()

    print(f"[T3] done. flagged={n_flagged}/{n_used} ({summary['pct_flagged']}%)  "
          f"verified_rate={summary['verified_alignment_rate']}")


if __name__ == "__main__":
    main()
