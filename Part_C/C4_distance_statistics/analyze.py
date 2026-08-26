#!/usr/bin/env python3
"""T10 -- statistics for the region accent-distance matrix.

C1's `ssl_controlled` distance matrix (per-sentence-centered, speaker-pair-
averaged WavLM) already reflects the T5-verified/re-frozen corpus -- data_clean/
did not change (T5's re-freeze wrote 0 new files; the verified_manifest.csv
excluded nothing), so its cached embeddings and distance matrix are current
and are NOT recomputed here. This adds the statistics needed to say whether
C1's near-1.0 pairwise distances (e.g. Chattogram-Sylhet 0.983 vs.
Rangpur-Sylhet 1.056) are a reliable ordering or noise:
  * bootstrap CI per pairwise distance (resample speakers within region)
  * Mantel test against geographic distance between the 8 regions
  * Ward hierarchical clustering + cophenetic correlation

Outputs (in ./reports): distance_ci.csv, mantel_geographic.json,
dendrogram_embedding.{png,pdf}, distance_heatmap.{png,pdf}, report.md, summary.json.
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.provenance import base_summary  # noqa: E402

from bpac_accent.cli import add_config_arg  # noqa: E402
from bpac_accent.distances import region_matrix_pooled  # noqa: E402
from bpac_accent.embeddings import extract_embeddings  # noqa: E402
from bpac_accent.reporting import ReportWriter, ensure_dir, maybe_tqdm, save_fig, save_json, save_table  # noqa: E402
from bpac_accent.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

# Approximate district-HQ coordinates (lat, lon) for the 8 BPAC regions --
# public geography, used only to build a geographic distance matrix for the
# Mantel test (T10).
REGION_COORDS = {
    "mymensingh": (24.7471, 90.4203),
    "faridpur": (23.6070, 89.8429),
    "dhaka": (23.8103, 90.4125),
    "chattogram": (22.3569, 91.7832),
    "khulna": (22.8456, 89.5403),
    "rajshahi": (24.3745, 88.6042),
    "rangpur": (25.7439, 89.2752),
    "sylhet": (24.8949, 91.8687),
}


def haversine_km(p1, p2):
    lat1, lon1, lat2, lon2 = map(np.radians, [p1[0], p1[1], p2[0], p2[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def geographic_distance_matrix(regions):
    n = len(regions)
    mat = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d = haversine_km(REGION_COORDS[regions[i]], REGION_COORDS[regions[j]])
        mat[i, j] = mat[j, i] = d
    return mat


def mantel_test(mat_a, mat_b, n_perm=10000, seed=42, method="pearson"):
    """Permutation test for correlation between two distance matrices'
    upper-triangular values, permuting the row/column labeling of mat_b."""
    n = mat_a.shape[0]
    iu = np.triu_indices(n, k=1)
    a_vec = mat_a[iu]
    valid = ~np.isnan(a_vec)
    rng = np.random.default_rng(seed)

    def corr(v1, v2):
        if method == "spearman":
            from scipy.stats import spearmanr
            return spearmanr(v1, v2).correlation
        return float(np.corrcoef(v1, v2)[0, 1])

    b_vec0 = mat_b[iu]
    observed = corr(a_vec[valid], b_vec0[valid])

    perm_corrs = np.empty(n_perm)
    idx = np.arange(n)
    for k in range(n_perm):
        p = rng.permutation(idx)
        b_perm = mat_b[np.ix_(p, p)][iu]
        perm_corrs[k] = corr(a_vec[valid], b_perm[valid])
    p_value = float(np.mean(np.abs(perm_corrs) >= abs(observed)))
    return {"observed_r": float(observed), "n_permutations": n_perm, "p_value": p_value,
           "perm_mean": float(np.nanmean(perm_corrs)), "perm_sd": float(np.nanstd(perm_corrs))}


def cophenetic_corr(D, Z):
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import squareform
    coph, coph_dists = cophenet(Z, squareform(D, checks=False))
    return float(coph)


def bootstrap_speaker_resample(meta, seed, rng):
    """Resample speakers WITH replacement within each region; return the row
    index array selecting (with repeats folded to distinct rows) the
    utterances belonging to the resampled speaker multiset."""
    keep_speakers = []
    for region, grp in meta.groupby("region"):
        spks = grp["speaker_uid"].unique()
        boot = rng.choice(spks, size=len(spks), replace=True)
        keep_speakers.extend(np.unique(boot).tolist())  # unique speakers used >=1 time
    return meta[meta["speaker_uid"].isin(set(keep_speakers))].index.to_numpy()


def main():
    ap = argparse.ArgumentParser(description="T10 distance-matrix statistics")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--n_bootstrap", type=int, default=200)
    ap.add_argument("--n_mantel_perm", type=int, default=10000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")
    c1 = cfg["c1_distance"]

    c1_dir = out.parent.parent / "C1_Region_accent_distance_matrix" / "reports"
    dm_path = c1_dir / "distance_matrix.csv"
    if not dm_path.exists():
        raise SystemExit(f"{dm_path} not found -- run C1 first.")
    dm = pd.read_csv(dm_path, index_col=0)
    regions = list(dm.index)
    mat = dm.values

    meta, X = extract_embeddings(cfg, "ssl")
    n_in = len(meta)
    metric = c1.get("metric", "cosine")
    standardize = c1.get("standardize", True)
    psc = c1.get("per_sentence_center", True)

    # -- bootstrap CI on each pairwise distance (resample speakers within region) --
    rng = np.random.default_rng(42)
    n_regions = len(regions)
    boot_stack = np.full((args.n_bootstrap, n_regions, n_regions), np.nan)
    for b in maybe_tqdm(range(args.n_bootstrap), desc="bootstrap distance CI", total=args.n_bootstrap):
        idx = bootstrap_speaker_resample(meta, 42, rng)
        meta_b = meta.loc[idx].reset_index(drop=True)
        X_b = X[idx]
        try:
            r_b, mat_b = region_matrix_pooled(meta_b, X_b, metric=metric, standardize=standardize,
                                              per_sentence_center=psc)
            order = [r_b.index(r) if r in r_b else None for r in regions]
            for i in range(n_regions):
                for j in range(n_regions):
                    if order[i] is not None and order[j] is not None:
                        boot_stack[b, i, j] = mat_b[order[i], order[j]]
        except Exception:
            continue

    ci_rows = []
    for i, j in itertools.combinations(range(n_regions), 2):
        vals = boot_stack[:, i, j]
        vals = vals[~np.isnan(vals)]
        if len(vals) < 10:
            continue
        ci_rows.append({
            "region_a": regions[i], "region_b": regions[j], "observed": mat[i, j],
            "boot_mean": float(np.mean(vals)), "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)), "n_boot_valid": len(vals),
        })
    ci_df = pd.DataFrame(ci_rows).sort_values("observed")
    save_table(ci_df, out / "distance_ci.csv", index=False)

    # -- Mantel test vs geography --
    geo_mat = geographic_distance_matrix(regions)
    D = (mat + mat.T) / 2.0
    np.fill_diagonal(D, 0.0)
    mantel_pearson = mantel_test(D, geo_mat, n_perm=args.n_mantel_perm, method="pearson")
    mantel_spearman = mantel_test(D, geo_mat, n_perm=args.n_mantel_perm, method="spearman")
    save_json({"pearson": mantel_pearson, "spearman": mantel_spearman}, out / "mantel_geographic.json")

    # -- Ward clustering + cophenetic correlation --
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    D_full = np.nan_to_num(D, nan=np.nanmax(D))
    Z = linkage(squareform(D_full, checks=False), method="ward")
    coph = cophenetic_corr(D_full, Z)

    fig, ax = plt.subplots(figsize=(1.2 * len(regions) + 3, 5))
    dendrogram(Z, labels=regions, ax=ax, leaf_rotation=40)
    ax.set_title(f"T10 -- Ward dendrogram (cophenetic r={coph:.3f})")
    ax.set_ylabel("distance")
    save_fig(fig, fig_dir / "dendrogram_embedding.png")
    fig.savefig(fig_dir / "dendrogram_embedding.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.1 * len(regions) + 2, 1.0 * len(regions) + 1.5))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=40, ha="right")
    ax.set_yticks(range(len(regions))); ax.set_yticklabels(regions)
    for i in range(len(regions)):
        for j in range(len(regions)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_title("T10 -- region accent distance (verified corpus)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, fig_dir / "distance_heatmap.png")
    fig.savefig(fig_dir / "distance_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # is the closest/farthest ordering reliable, per the bootstrap CIs?
    closest = ci_df.iloc[0]
    farthest = ci_df.iloc[-1]
    ci_overlap = not (closest["ci_high"] < farthest["ci_low"])

    summary = base_summary("T10_distance_statistics", cfg.config_path, n_in, n_in)
    summary.update({
        "closest_pair": [closest["region_a"], closest["region_b"], round(float(closest["observed"]), 4),
                         round(float(closest["ci_low"]), 4), round(float(closest["ci_high"]), 4)],
        "farthest_pair": [farthest["region_a"], farthest["region_b"], round(float(farthest["observed"]), 4),
                          round(float(farthest["ci_low"]), 4), round(float(farthest["ci_high"]), 4)],
        "closest_farthest_ci_overlap": ci_overlap,
        "mantel_pearson": mantel_pearson, "mantel_spearman": mantel_spearman,
        "cophenetic_correlation": coph,
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T10 -- Region Accent-Distance Matrix: Statistics")
    rep.p(
        "C1's `ssl_controlled` distance matrix already reflects the T5-verified corpus (data_clean/ "
        "did not change during re-freeze). This adds bootstrap CIs, a Mantel test against geographic "
        "distance, and Ward clustering + cophenetic correlation, to say whether the near-1.0 pairwise "
        "distances in C1 (e.g. closest 0.983, farthest 1.056) are a reliable ordering or noise."
    )
    rep.h2("Is the closest/farthest ordering reliable?")
    rep.p(
        f"Closest pair **{closest['region_a']}-{closest['region_b']}** = {closest['observed']:.4f} "
        f"(95% CI [{closest['ci_low']:.4f}, {closest['ci_high']:.4f}]); farthest pair "
        f"**{farthest['region_a']}-{farthest['region_b']}** = {farthest['observed']:.4f} "
        f"(95% CI [{farthest['ci_low']:.4f}, {farthest['ci_high']:.4f}]). "
        + ("**CIs overlap -- the ordering is NOT statistically reliable; do not claim a specific "
           "closest/farthest pair, only that overall distances are small and noisy.**" if ci_overlap
           else "CIs do not overlap -- the ordering is statistically reliable.")
    )
    rep.h2("Mantel test: does acoustic distance track geography?")
    rep.bullet(f"Pearson r={mantel_pearson['observed_r']:.3f}, p={mantel_pearson['p_value']:.4f} "
              f"({mantel_pearson['n_permutations']} permutations)")
    rep.bullet(f"Spearman r={mantel_spearman['observed_r']:.3f}, p={mantel_spearman['p_value']:.4f}")
    rep.p(
        "A significant positive correlation would mean acoustic distance tracks geography (a clean, "
        "interpretable result); a null result is also reportable and expected given how weak the "
        "unsupervised distances are overall (see S6.2's honest-negative-result framing)."
    )
    rep.h2("Ward clustering")
    rep.bullet(f"Cophenetic correlation = **{coph:.4f}** (how well the dendrogram represents the "
              "full distance matrix; closer to 1 = more faithful).")
    rep.image("figures/dendrogram_embedding.png", "Ward dendrogram of region accent distances.")
    rep.image("figures/distance_heatmap.png", "Region x region accent distance heatmap.")
    rep.h2("Bootstrap CIs, all pairs")
    rep.table(ci_df.round(4), index=False)
    rep.h2("For the paper")
    rep.p(
        "S6.2 and the left panel of T19's headline figure. If CIs overlap, state plainly that the "
        "unsupervised distance matrix does not reliably order regions -- this motivates T9's "
        "discriminative LOSO test and Part D's phonetic distance matrix (T18) as the more decisive "
        "analyses."
    )
    rep.write()

    print(f"[T10] done. closest={list(summary['closest_pair'])}  farthest={list(summary['farthest_pair'])}  "
          f"ci_overlap={ci_overlap}  mantel_p={mantel_pearson['p_value']}  cophenetic={coph:.4f}")


if __name__ == "__main__":
    main()
