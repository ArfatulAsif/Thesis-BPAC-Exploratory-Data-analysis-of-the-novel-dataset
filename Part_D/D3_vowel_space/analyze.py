#!/usr/bin/env python3
"""T15 -- Vowel space (F1/F2/F3, Lobanov-normalized).

Extracts formants (Praat/parselmouth Burg algorithm) at the 20/50/80% points
of every vowel token from T12's forced alignment, filters implausible
tracking, Lobanov-normalizes per speaker (removes vocal-tract-length/
anatomical differences, preserving dialect differences -- necessary because
S6 already showed speaker identity dominates the raw acoustic geometry),
and reports per-region vowel quadrilaterals, vowel-space area, and Pillai
scores for near/merged vowel pairs.

Outputs (in ./reports): vowel_tokens.parquet, vowel_means_by_region.csv,
vowel_space_area.csv, pillai_scores.csv, vowel_mixedmodel.csv, figures/*,
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
from bpac_phon.reporting import ReportWriter, ensure_dir, maybe_tqdm, save_fig, save_json, save_table  # noqa: E402
from bpac_phon.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

VOWEL_PHONES = ["a", "O", "o", "i", "u", "e", "oi", "ou"]
ADJACENT_EXCLUDE = {"r", "rr", "l"}


def extract_formants(wav_path, t0, t1, ceiling_hz, sample_points=(0.2, 0.5, 0.8), window_ms=25):
    import parselmouth
    snd = parselmouth.Sound(str(wav_path))
    dur = t1 - t0
    seg = snd.extract_part(from_time=max(0, t0 - 0.02), to_time=min(snd.duration, t1 + 0.02), preserve_times=True)
    formant = seg.to_formant_burg(time_step=None, max_number_of_formants=5.0,
                                  maximum_formant=ceiling_hz, window_length=window_ms / 1000.0,
                                  pre_emphasis_from=50.0)
    out = {}
    for frac in sample_points:
        t = t0 + frac * dur
        row = {}
        for fnum in (1, 2, 3):
            try:
                v = formant.get_value_at_time(fnum, t)
            except Exception:
                v = np.nan
            row[f"F{fnum}"] = float(v) if v == v else np.nan  # NaN check
        out[frac] = row
    return out


def lobanov_normalize(df, formant_cols=("F1", "F2", "F3")):
    df = df.copy()
    for col in formant_cols:
        grp = df.groupby("speaker_uid")[col]
        mean = grp.transform("mean")
        sd = grp.transform("std")
        df[col + "_z"] = (df[col] - mean) / sd.replace(0, np.nan)
    return df


def convex_hull_area(points):
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return np.nan
    try:
        return float(ConvexHull(points).volume)  # 'volume' = area in 2D
    except Exception:
        return np.nan


def pillai_score(df, vowel_a, vowel_b, cols=("F1_z", "F2_z")):
    """MANOVA Pillai trace for two vowel categories on (F1_z, F2_z);
    higher = more distinct, near 0 = merged."""
    from statsmodels.multivariate.manova import MANOVA
    sub = df[df["phone"].isin([vowel_a, vowel_b])].dropna(subset=list(cols))
    if sub["phone"].nunique() < 2 or len(sub) < 10:
        return np.nan
    try:
        m = MANOVA.from_formula(f"{cols[0]} + {cols[1]} ~ phone", data=sub)
        r = m.mv_test()
        pillai = r.results["phone"]["stat"].loc["Pillai's trace", "Value"]
        return float(pillai)
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser(description="T15 vowel space")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--max_tokens_per_vowel_region", type=int, default=None,
                     help="Optional cap for a quick pass.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")
    d3 = cfg["d3_vowel_space"]

    phone_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "D1_alignment", "reports", "phone_tokens.parquet")
    if not os.path.exists(phone_path):
        raise SystemExit(f"{phone_path} not found -- run D1_alignment (T12) first.")
    phones = pd.read_parquet(phone_path)

    clean_mani = None
    from bpac_phon.manifest import build_manifest
    clean_mani = build_manifest(cfg.data_root, cfg.bpac_subdir, None, verbose=False)
    filepath_map = dict(zip(clean_mani["utterance_id"], clean_mani["filepath"]))

    vtok = phones[phones["phone"].isin(VOWEL_PHONES)].copy()
    min_ms = float(d3.get("min_vowel_ms", 50))
    vtok = vtok[vtok["duration_s"] * 1000 >= min_ms]
    vtok = vtok[~vtok["prev_phone"].isin(ADJACENT_EXCLUDE) & ~vtok["next_phone"].isin(ADJACENT_EXCLUDE)]
    if args.max_tokens_per_vowel_region:
        vtok = (vtok.groupby(["region", "phone"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.max_tokens_per_vowel_region), random_state=42)))
    n_candidate = len(vtok)

    ceilings = d3.get("formant_ceiling_hz", {"male": 5000, "female": 5500})
    sample_points = tuple(d3.get("sample_points", [0.2, 0.5, 0.8]))
    f1_lo, f1_hi = d3.get("f1_valid_hz", [200, 1200])
    f2_lo, f2_hi = d3.get("f2_valid_hz", [500, 3000])

    rows = []
    for r in maybe_tqdm(vtok.itertuples(index=False), desc="T15 formants", total=len(vtok)):
        wav = filepath_map.get(r.utterance_id)
        if wav is None:
            continue
        ceiling = ceilings.get(r.gender, 5500)
        try:
            samples = extract_formants(wav, r.start_s, r.end_s, ceiling, sample_points=sample_points)
        except Exception:
            continue
        mid = sample_points[len(sample_points) // 2]
        f = samples.get(mid, {})
        rows.append({
            "utterance_id": r.utterance_id, "region": r.region, "gender": r.gender,
            "speaker_uid": r.speaker_uid, "phone": r.phone, "duration_s": r.duration_s,
            "F1": f.get("F1"), "F2": f.get("F2"), "F3": f.get("F3"),
            **{f"{k}_frac{int(frac*100)}": v for frac, vals in samples.items() for k, v in vals.items()},
        })
    vdf = pd.DataFrame(rows)
    n_before_filter = len(vdf)
    vdf = vdf.dropna(subset=["F1", "F2"])
    vdf = vdf[vdf["F1"].between(f1_lo, f1_hi) & vdf["F2"].between(f2_lo, f2_hi)]
    n_after_filter = len(vdf)
    retention_pct = round(100 * n_after_filter / n_candidate, 2) if n_candidate else None

    vdf = lobanov_normalize(vdf)
    vdf.to_parquet(out / "vowel_tokens.parquet", index=False)

    retention_by_region = (
        pd.DataFrame({"stage_candidate": vtok.groupby("region").size()}).join(
            vdf.groupby("region").size().rename("stage_retained"), how="left"
        ).fillna(0)
    )
    retention_by_region["retention_pct"] = (
        100 * retention_by_region["stage_retained"] / retention_by_region["stage_candidate"]
    ).round(2)
    save_table(retention_by_region.reset_index().rename(columns={"index": "region"}),
              out / "retention_by_region.csv", index=False)

    means = (
        vdf.groupby(["region", "phone"])[["F1_z", "F2_z"]]
        .agg(["mean", "std", "count"])
    )
    means.columns = ["_".join(c) for c in means.columns]
    means = means.reset_index()
    save_table(means, out / "vowel_means_by_region.csv", index=False)

    area_rows = []
    for region, sub in vdf.groupby("region"):
        centroids = sub.groupby("phone")[["F2_z", "F1_z"]].mean().dropna()
        area_rows.append({"region": region, "vowel_space_area": convex_hull_area(centroids.values),
                          "n_vowel_types": len(centroids)})
    area_df = pd.DataFrame(area_rows).sort_values("vowel_space_area", ascending=False)
    save_table(area_df, out / "vowel_space_area.csv", index=False)

    # Pillai scores for commonly-discussed near pairs (Bangla e/E-ish and o/O)
    pillai_pairs = [("e", "i"), ("o", "u"), ("O", "o"), ("a", "O")]
    pillai_rows = []
    for region, sub in vdf.groupby("region"):
        for va, vb in pillai_pairs:
            pillai_rows.append({"region": region, "pair": f"{va}-{vb}",
                                "pillai": pillai_score(sub, va, vb)})
    pillai_df = pd.DataFrame(pillai_rows)
    save_table(pillai_df, out / "pillai_scores.csv", index=False)

    # mixed model per vowel: F1_z ~ region + (1|speaker)
    import statsmodels.formula.api as smf
    mm_rows = []
    for vowel in VOWEL_PHONES:
        sub = vdf[vdf["phone"] == vowel].dropna(subset=["F1_z", "F2_z"])
        if sub["region"].nunique() < 2 or sub["speaker_uid"].nunique() < 3 or len(sub) < 20:
            continue
        for target in ["F1_z", "F2_z"]:
            try:
                md = smf.mixedlm(f"{target} ~ C(region)", sub, groups=sub["speaker_uid"])
                res = md.fit(reml=True, method="lbfgs")
                for term in res.pvalues.index:
                    if term.startswith("C(region)"):
                        mm_rows.append({"vowel": vowel, "target": target, "term": term,
                                        "coef": res.params[term], "p_value": res.pvalues[term]})
            except Exception as e:
                print(f"[T15] mixed model failed for vowel={vowel} target={target}: {e}")
    mm_df = pd.DataFrame(mm_rows)
    if len(mm_df):
        from statsmodels.stats.multitest import multipletests
        _, q, _, _ = multipletests(mm_df["p_value"], method="fdr_bh")
        mm_df["q_value"] = q
    save_table(mm_df, out / "vowel_mixedmodel.csv", index=False)

    # figures: overlaid quadrilateral (region means) + small-multiple ellipses
    fig, ax = plt.subplots(figsize=(7, 7))
    regions = sorted(vdf["region"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    for i, region in enumerate(regions):
        sub_means = means[means["region"] == region].set_index("phone")
        present = [v for v in VOWEL_PHONES if v in sub_means.index]
        if len(present) < 3:
            continue
        xs = [sub_means.loc[v, "F2_z_mean"] for v in present]
        ys = [sub_means.loc[v, "F1_z_mean"] for v in present]
        color = cmap(i % 10)
        ax.plot(xs + [xs[0]], ys + [ys[0]], marker="o", color=color, label=region, alpha=0.8)
        for v, x, y in zip(present, xs, ys):
            ax.annotate(v, (x, y), fontsize=6, color=color)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("F2 (Lobanov z, reversed)"); ax.set_ylabel("F1 (Lobanov z, reversed)")
    ax.set_title("T15 -- vowel quadrilateral overlay, 8 regions")
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, fig_dir / "vowel_quadrilateral_overlay.png")
    fig.savefig(fig_dir / "vowel_quadrilateral_overlay.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    for ax, region in zip(axes.flat, regions):
        sub = vdf[vdf["region"] == region]
        for vi, vowel in enumerate(VOWEL_PHONES):
            vsub = sub[sub["phone"] == vowel].dropna(subset=["F1_z", "F2_z"])
            if len(vsub) < 3:
                continue
            color = plt.get_cmap("tab10")(vi % 10)
            ax.scatter(vsub["F2_z"], vsub["F1_z"], s=3, color=color, alpha=0.3)
            cx, cy = vsub["F2_z"].mean(), vsub["F1_z"].mean()
            sx, sy = vsub["F2_z"].std(), vsub["F1_z"].std()
            if sx == sx and sy == sy:
                ax.add_patch(Ellipse((cx, cy), width=2 * 1.96 * sx, height=2 * 1.96 * sy,
                                     fill=False, edgecolor=color, linewidth=1.2))
            ax.annotate(vowel, (cx, cy), fontsize=7, color=color)
        ax.set_title(region, fontsize=9)
        ax.invert_xaxis(); ax.invert_yaxis()
    fig.suptitle("T15 -- vowel ellipses by region (small multiples)")
    save_fig(fig, fig_dir / "vowel_ellipses_by_region.png")
    fig.savefig(fig_dir / "vowel_ellipses_by_region.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = base_summary("T15_vowel_space", cfg.config_path, n_candidate, n_after_filter, count_unit="vowel_tokens")
    summary.update({
        "n_before_formant_extraction": n_candidate,
        "n_after_range_filter": n_after_filter,
        "retention_pct": retention_pct,
        "vowel_space_area_by_region": area_df.to_dict(orient="records"),
        "n_pillai_pairs": len(pillai_pairs),
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T15 -- Vowel Space (Lobanov-Normalized F1/F2)")
    rep.p(
        f"{n_after_filter}/{n_candidate} vowel tokens retained ({retention_pct}%) after duration/"
        "adjacency/formant-range filtering. Lobanov-normalized per speaker to remove vocal-tract-"
        "length differences while preserving dialect differences."
    )
    rep.h2("Retention by region")
    rep.table(retention_by_region.reset_index().round(2), index=False)
    rep.h2("Vowel space area by region")
    rep.table(area_df.round(3), index=False)
    rep.h2("Pillai scores (near-vowel-pair distinctness; near 0 = merged)")
    rep.table(pillai_df.round(4), index=False)
    rep.h2("Figures")
    rep.image("figures/vowel_quadrilateral_overlay.png", "8-region vowel quadrilateral overlay -- main paper figure.")
    rep.image("figures/vowel_ellipses_by_region.png", "Vowel ellipses, small multiples by region.")
    rep.h2("For the paper")
    rep.p("S7.4. The overlaid quadrilateral is a main paper figure.")
    rep.write()

    print(f"[T15] done. tokens={n_after_filter}/{n_candidate} ({retention_pct}%)  "
          f"regions_with_data={len(regions)}")


if __name__ == "__main__":
    main()
