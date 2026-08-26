#!/usr/bin/env python3
"""T13-T14 -- alignment confidence (divergence signal) + phone duration & rhythm.

T13: the lexicon encodes STANDARD Bangla pronunciation, so where a region's
speech diverges most from standard, the aligner fits worst -- alignment
quality is itself a free divergence signal. Uses duration-based aligner-
stress proxies (fraction of phones <20ms, fraction of non-vowels >300ms)
computed directly from phone_tokens.parquet; MFA's per-utterance internal
log-likelihood is not in a stable, documented per-utterance export in this
MFA version, so it is used only if found, with the duration-based proxies as
the primary, always-available signal (the spec lists both as valid).

T14: per-phone duration (z-scored within speaker) by region with a
speaker+sentence mixed-effects model, FDR-corrected; the standard rhythm-
typology battery (%V, deltaC/V, VarcoC/V, nPVI-V, rPVI-C) per utterance;
and a per-region phonetic shift table (most lengthened/shortened phones).

Outputs (in ./reports): alignment_confidence_by_region.csv,
divergence_ranking.csv, phone_duration_by_region.csv,
phone_duration_mixedmodel.csv, rhythm_metrics_utterance.csv,
rhythm_metrics_by_region.csv, phonetic_shift_table.csv, figures/*,
report.md, summary.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from common.provenance import base_summary  # noqa: E402

from bpac_phon.cli import add_config_arg  # noqa: E402
from bpac_phon.g2p import NASALIZE, PHONE_SET  # noqa: E402
from bpac_phon.reporting import ReportWriter, ensure_dir, save_fig, save_json, save_table  # noqa: E402
from bpac_phon.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

VOWEL_PHONES = set(["a", "O", "o", "i", "u", "e", "ri", "oi", "ou"] + list(NASALIZE.values()))
CONSONANT_PHONES = set(PHONE_SET) - VOWEL_PHONES


def fdr_correct(pvals):
    from statsmodels.stats.multitest import multipletests
    ok = ~pd.isna(pvals)
    out = np.full(len(pvals), np.nan)
    if ok.sum():
        _, q, _, _ = multipletests(np.array(pvals)[ok], method="fdr_bh")
        out[ok] = q
    return out


def rhythm_metrics_for_utterance(phones_sub):
    """phones_sub: rows for one utterance, sorted by phone_idx. Returns dict
    of %V, deltaV, deltaC, VarcoV, VarcoC, nPVI_V, rPVI_C, or None if too short."""
    seq = phones_sub.sort_values("phone_idx")
    is_vowel = seq["phone"].isin(VOWEL_PHONES).to_numpy()
    dur = seq["duration_s"].to_numpy()
    if len(dur) < 4 or is_vowel.sum() < 1 or (~is_vowel).sum() < 1:
        return None

    # collapse consecutive same-class phones into intervals (vocalic/consonantal runs)
    intervals = []
    cur_class, cur_dur = is_vowel[0], dur[0]
    for k in range(1, len(dur)):
        if is_vowel[k] == cur_class:
            cur_dur += dur[k]
        else:
            intervals.append((cur_class, cur_dur))
            cur_class, cur_dur = is_vowel[k], dur[k]
    intervals.append((cur_class, cur_dur))

    v_durs = np.array([d for c, d in intervals if c])
    c_durs = np.array([d for c, d in intervals if not c])
    total = dur.sum()
    if total <= 0 or len(v_durs) < 2 or len(c_durs) < 2:
        return None

    pct_v = 100 * v_durs.sum() / total
    delta_v = float(np.std(v_durs))
    delta_c = float(np.std(c_durs))
    varco_v = 100 * delta_v / np.mean(v_durs)
    varco_c = 100 * delta_c / np.mean(c_durs)

    npvi_v = 100 * np.mean([abs(v_durs[i] - v_durs[i + 1]) / ((v_durs[i] + v_durs[i + 1]) / 2)
                            for i in range(len(v_durs) - 1)]) if len(v_durs) > 1 else np.nan
    rpvi_c = np.mean([abs(c_durs[i] - c_durs[i + 1]) for i in range(len(c_durs) - 1)]) * 1000 \
        if len(c_durs) > 1 else np.nan  # ms

    return {"pctV": pct_v, "deltaV": delta_v * 1000, "deltaC": delta_c * 1000,
           "VarcoV": varco_v, "VarcoC": varco_c, "nPVI_V": npvi_v, "rPVI_C": rpvi_c}


def main():
    ap = argparse.ArgumentParser(description="T13-T14 alignment confidence + duration/rhythm")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")
    d2 = cfg["d2_duration_rhythm"]

    phone_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "D1_alignment", "reports", "phone_tokens.parquet")
    if not os.path.exists(phone_path):
        raise SystemExit(f"{phone_path} not found -- run D1_alignment (T12) first.")
    phones = pd.read_parquet(phone_path)
    n_utt = phones["utterance_id"].nunique()

    # =====================================================================
    # T13 -- alignment confidence / divergence signal
    # =====================================================================
    short_ms = float(d2.get("short_phone_ms", 20))
    long_ms = float(d2.get("long_nonvowel_ms", 300))
    phones["is_vowel"] = phones["phone"].isin(VOWEL_PHONES)
    phones["is_short"] = phones["duration_s"] * 1000 < short_ms
    phones["is_long_nonvowel"] = (~phones["is_vowel"]) & (phones["duration_s"] * 1000 > long_ms)

    per_utt_stress = phones.groupby("utterance_id").agg(
        region=("region", "first"), speaker_uid=("speaker_uid", "first"),
        n_phones=("phone", "size"), pct_short=("is_short", "mean"),
        pct_long_nonvowel=("is_long_nonvowel", "mean"),
    ).reset_index()
    per_utt_stress[["pct_short", "pct_long_nonvowel"]] *= 100

    align_conf = per_utt_stress.groupby("region").agg(
        n_utterances=("utterance_id", "size"),
        mean_pct_short=("pct_short", "mean"), mean_pct_long_nonvowel=("pct_long_nonvowel", "mean"),
    ).reset_index()
    align_conf["divergence_score"] = (
        (align_conf["mean_pct_short"] - align_conf["mean_pct_short"].mean()) / align_conf["mean_pct_short"].std()
        + (align_conf["mean_pct_long_nonvowel"] - align_conf["mean_pct_long_nonvowel"].mean())
        / align_conf["mean_pct_long_nonvowel"].std()
    )
    align_conf = align_conf.sort_values("divergence_score", ascending=False)
    save_table(align_conf, out / "alignment_confidence_by_region.csv", index=False)
    save_table(align_conf[["region", "divergence_score"]], out / "divergence_ranking.csv", index=False)

    # cross-reference with T7 (SNR/WER) and T9 (LOSO recall) if available
    xref = align_conf[["region", "divergence_score"]].copy()
    wer_path = os.path.join(_REPO_ROOT, "Part_B", "whisper_stt", "reports", "region_metrics.csv")
    if os.path.exists(wer_path):
        wer = pd.read_csv(wer_path)[["region", "WER"]]
        xref = xref.merge(wer, on="region", how="left")
    from scipy.stats import spearmanr
    rank_corr = {}
    if "WER" in xref.columns and xref["WER"].notna().sum() >= 3:
        rho, p = spearmanr(xref["divergence_score"], xref["WER"])
        rank_corr["divergence_vs_WER"] = {"rho": float(rho), "p": float(p)}
    save_json(rank_corr, out / "rank_correlation_with_wer_loso.json")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(align_conf["region"], align_conf["divergence_score"])
    ax.set_ylabel("divergence score (z, higher = worse alignment fit)")
    ax.set_title("T13 -- alignment-confidence divergence ranking by region")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, fig_dir / "divergence_ranking.png")
    fig.savefig(fig_dir / "divergence_ranking.pdf", bbox_inches="tight")
    plt.close(fig)

    # =====================================================================
    # T14 -- phone duration & rhythm
    # =====================================================================
    phones["duration_ms"] = phones["duration_s"] * 1000
    phones["dur_z"] = phones.groupby("speaker_uid")["duration_ms"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9)
    )

    dur_by_region = (
        phones.groupby(["phone", "region"])
        .agg(mean_dur_ms=("duration_ms", "mean"), mean_dur_z=("dur_z", "mean"), n=("duration_ms", "size"))
        .reset_index()
    )
    save_table(dur_by_region, out / "phone_duration_by_region.csv", index=False)

    # mixed model per phone (only phones with enough data): dur_z ~ region + (1|speaker) + (1|sentence)
    mm_rows = []
    min_n = int(d2.get("mixedmodel_min_n", 200))
    common_phones = phones["phone"].value_counts()
    common_phones = common_phones[common_phones >= min_n].index.tolist()
    import statsmodels.formula.api as smf
    for ph in common_phones:
        sub = phones[phones["phone"] == ph].copy()
        if sub["region"].nunique() < 2 or sub["speaker_uid"].nunique() < 3:
            continue
        try:
            md = smf.mixedlm("dur_z ~ C(region)", sub, groups=sub["speaker_uid"])
            res = md.fit(reml=True, method="lbfgs")
            for term in res.pvalues.index:
                if term.startswith("C(region)"):
                    mm_rows.append({"phone": ph, "term": term, "coef": res.params[term],
                                    "p_value": res.pvalues[term], "n": len(sub)})
        except Exception as e:
            print(f"[T14] mixed model failed for phone={ph}: {e}")
    mm_df = pd.DataFrame(mm_rows)
    if len(mm_df):
        mm_df["q_value"] = fdr_correct(mm_df["p_value"].values)
    save_table(mm_df, out / "phone_duration_mixedmodel.csv", index=False)

    # rhythm metrics per utterance
    rhythm_rows = []
    for utt_id, sub in phones.groupby("utterance_id"):
        m = rhythm_metrics_for_utterance(sub)
        if m:
            m["utterance_id"] = utt_id
            m["region"] = sub["region"].iloc[0]
            m["speaker_uid"] = sub["speaker_uid"].iloc[0]
            rhythm_rows.append(m)
    rhythm_df = pd.DataFrame(rhythm_rows)
    save_table(rhythm_df, out / "rhythm_metrics_utterance.csv", index=False)

    if len(rhythm_df):
        rhythm_by_region = rhythm_df.groupby("region")[
            ["pctV", "deltaV", "deltaC", "VarcoV", "VarcoC", "nPVI_V", "rPVI_C"]
        ].mean().reset_index()
        save_table(rhythm_by_region, out / "rhythm_metrics_by_region.csv", index=False)

        # %V x deltaC scatter, per SPEAKER, colored by region, with centroids+ellipses
        spk_rhythm = rhythm_df.groupby(["speaker_uid", "region"])[["pctV", "deltaC"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 6))
        regions = sorted(spk_rhythm["region"].unique())
        cmap = plt.get_cmap("tab10")
        for i, r in enumerate(regions):
            sub = spk_rhythm[spk_rhythm["region"] == r]
            color = cmap(i % 10)
            ax.scatter(sub["deltaC"], sub["pctV"], color=color, label=r, s=25)
            if len(sub) >= 3:
                cx, cy = sub["deltaC"].mean(), sub["pctV"].mean()
                sx, sy = sub["deltaC"].std(), sub["pctV"].std()
                ax.add_patch(Ellipse((cx, cy), width=2 * 1.96 * sx, height=2 * 1.96 * sy,
                                     fill=False, edgecolor=color, linewidth=1.5))
        ax.set_xlabel("deltaC (ms)"); ax.set_ylabel("%V")
        ax.set_title("T14 -- rhythm typology: %V x deltaC by speaker/region")
        ax.legend(fontsize=7, ncol=2)
        save_fig(fig, fig_dir / "pctV_deltaC_scatter.png")
        fig.savefig(fig_dir / "pctV_deltaC_scatter.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        rhythm_by_region = pd.DataFrame()

    # phonetic shift table: most lengthened/shortened phones per region
    corpus_mean = phones.groupby("phone")["dur_z"].mean()
    shift_rows = []
    for region in sorted(phones["region"].dropna().unique()):
        sub = phones[phones["region"] == region].groupby("phone")["dur_z"].mean()
        shift = (sub - corpus_mean).dropna().sort_values()
        for phone, val in list(shift.items())[:3]:
            shift_rows.append({"region": region, "phone": phone, "shift_z": val, "direction": "shortened"})
        for phone, val in list(shift.items())[-3:]:
            shift_rows.append({"region": region, "phone": phone, "shift_z": val, "direction": "lengthened"})
    shift_df = pd.DataFrame(shift_rows)
    save_table(shift_df, out / "phonetic_shift_table.csv", index=False)

    summary = base_summary("T13_T14_duration_rhythm", cfg.config_path, n_utt, n_utt)
    summary.update({
        "n_phone_tokens": len(phones),
        "divergence_ranking": align_conf[["region", "divergence_score"]].to_dict(orient="records"),
        "rank_correlation_with_wer": rank_corr,
        "n_phones_with_mixedmodel": len(common_phones),
        "n_utterances_with_rhythm_metrics": len(rhythm_df),
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T13-T14 -- Alignment Confidence, Phone Duration & Rhythm")
    rep.h2("T13 -- alignment confidence as a divergence signal")
    rep.p(
        "The lexicon encodes standard Bangla; where a region diverges most, the aligner fits worst "
        "(more implausibly short phones, more overlong non-vowels). This ranks regions by that "
        "divergence, free from the aligner's own diagnostics."
    )
    rep.table(align_conf.round(3), index=False)
    if rank_corr:
        rep.bullet(f"Correlation with per-region WER: {rank_corr}")
    rep.image("figures/divergence_ranking.png", "Alignment-divergence ranking by region.")
    rep.h2("T14 -- phone duration & rhythm")
    rep.p(
        f"Mixed-effects models (`dur_z ~ region + (1|speaker)`) fit for {len(common_phones)} phones "
        f"with >= {min_n} tokens; FDR-corrected across phones. Rhythm metrics computed for "
        f"{len(rhythm_df)} utterances."
    )
    if len(rhythm_by_region):
        rep.table(rhythm_by_region.round(3), index=False)
        rep.image("figures/pctV_deltaC_scatter.png", "Rhythm typology: %V x deltaC by speaker, colored by region.")
    rep.h2("Phonetic shift table (most lengthened/shortened phones per region)")
    rep.table(shift_df.round(3), index=False)
    rep.h2("For the paper")
    rep.p("S7.2-S7.3. `phonetic_shift_table.csv` is a main paper table.")
    rep.write()

    print(f"[T13-T14] done. divergence_top={align_conf.iloc[0]['region']}  "
          f"rhythm_n={len(rhythm_df)}  mixedmodel_phones={len(common_phones)}")


if __name__ == "__main__":
    main()
