#!/usr/bin/env python3
"""T18 -- fused phonetic distance matrix + dendrogram.

Fuses T14 (phone duration + rhythm), T15 (vowel formants), T16 (sibilant
CoG + VOT), and T17 (F0 features) into one region x feature matrix, purely
from interpretable phonetic measurements -- as opposed to S6's matrix built
from opaque neural embeddings. Two independent routes to the same object is
the setup for T19's convergence test.

Outputs (in ./reports): region_feature_matrix.csv,
phonetic_distance_matrix.csv, phonetic_distance_ci.csv, branch_support.csv,
feature_block_sensitivity.csv, pca_loadings.csv, figures/*, report.md, summary.json.
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

from bpac_phon.cli import add_config_arg  # noqa: E402
from bpac_phon.reporting import ReportWriter, ensure_dir, save_fig, save_json, save_table  # noqa: E402
from bpac_phon.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PART_D = os.path.dirname(HERE)


def _read_csv_if_exists(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def build_feature_blocks(regions):
    """Returns dict[block_name] -> DataFrame(index=region, columns=features),
    reading whatever upstream D-task outputs exist. Missing blocks are
    skipped (documented via feature_block_sensitivity.csv / the report)."""
    blocks = {}

    # -- duration/rhythm (T14) --
    dur_path = os.path.join(PART_D, "D2_duration_rhythm", "reports", "phone_duration_by_region.csv")
    dur = _read_csv_if_exists(dur_path)
    if dur is not None:
        piv = dur.pivot_table(index="region", columns="phone", values="mean_dur_z", aggfunc="mean")
        blocks["phone_duration"] = piv.reindex(regions)
    rhythm_path = os.path.join(PART_D, "D2_duration_rhythm", "reports", "rhythm_metrics_by_region.csv")
    rhythm = _read_csv_if_exists(rhythm_path)
    if rhythm is not None:
        blocks["rhythm"] = rhythm.set_index("region").reindex(regions)

    # -- vowel space (T15) --
    vowel_path = os.path.join(PART_D, "D3_vowel_space", "reports", "vowel_means_by_region.csv")
    vowel = _read_csv_if_exists(vowel_path)
    if vowel is not None:
        f1 = vowel.pivot_table(index="region", columns="phone", values="F1_z_mean", aggfunc="mean")
        f2 = vowel.pivot_table(index="region", columns="phone", values="F2_z_mean", aggfunc="mean")
        f1.columns = [f"F1_{c}" for c in f1.columns]
        f2.columns = [f"F2_{c}" for c in f2.columns]
        blocks["vowel_formants"] = pd.concat([f1, f2], axis=1).reindex(regions)
    area_path = os.path.join(PART_D, "D3_vowel_space", "reports", "vowel_space_area.csv")
    area = _read_csv_if_exists(area_path)
    if area is not None:
        blocks["vowel_space_area"] = area.set_index("region")[["vowel_space_area"]].reindex(regions)
    pillai_path = os.path.join(PART_D, "D3_vowel_space", "reports", "pillai_scores.csv")
    pillai = _read_csv_if_exists(pillai_path)
    if pillai is not None:
        piv = pillai.pivot_table(index="region", columns="pair", values="pillai", aggfunc="mean")
        blocks["pillai"] = piv.reindex(regions)

    # -- sibilants + VOT (T16) --
    sib_path = os.path.join(PART_D, "D4_consonants", "reports", "sibilant_cog_by_region.csv")
    sib = _read_csv_if_exists(sib_path)
    if sib is not None:
        blocks["sibilant_cog"] = sib.set_index("region")[["cog_z_mean"]].reindex(regions)
    vot_path = os.path.join(PART_D, "D4_consonants", "reports", "vot_by_region_category.csv")
    vot = _read_csv_if_exists(vot_path)
    if vot is not None:
        piv = vot.pivot_table(index="region", columns="phone", values="mean_vot_ms", aggfunc="mean")
        blocks["vot"] = piv.reindex(regions)

    # -- F0 (T17) --
    f0_path = os.path.join(PART_D, "D5_intonation", "reports", "f0_summary_features_by_region.csv")
    f0 = _read_csv_if_exists(f0_path)
    if f0 is not None:
        blocks["f0"] = f0.set_index("region").reindex(regions)

    return blocks


def standardize_blocks(blocks):
    std_blocks = {}
    for name, df in blocks.items():
        df = df.astype(float)
        std = (df - df.mean(axis=0)) / (df.std(axis=0) + 1e-9)
        std_blocks[name] = std
    return std_blocks


def cophenetic_corr(D, Z):
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import squareform
    coph, _ = cophenet(Z, squareform(D, checks=False))
    return float(coph)


def main():
    ap = argparse.ArgumentParser(description="T18 fused phonetic distance matrix")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--n_bootstrap", type=int, default=200)
    ap.add_argument("--n_branch_support", type=int, default=1000)
    ap.add_argument("--exclude_blocks", nargs="*", default=[],
                     help="Feature block names to drop before fusing (e.g. 'vot' once T16's "
                          "VOT is policy-excluded from reported findings for lack of manual "
                          "verification -- see Part_D/D4_consonants report).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")
    d6 = cfg["d6_phonetic_distance"]

    regions = ["mymensingh", "faridpur", "dhaka", "chattogram", "khulna", "rajshahi", "rangpur", "sylhet"]
    blocks = build_feature_blocks(regions)
    if not blocks:
        raise SystemExit("No upstream D-task feature blocks found -- run T14-T17 first.")
    for b in args.exclude_blocks:
        blocks.pop(b, None)
        print(f"[T18] excluded feature block '{b}' (policy exclusion, not missing data)")
    std_blocks = standardize_blocks(blocks)

    feat_matrix = pd.concat(std_blocks.values(), axis=1)
    feat_matrix = feat_matrix.dropna(axis=1, how="all")
    save_table(feat_matrix.reset_index().rename(columns={"index": "region"}),
              out / "region_feature_matrix.csv", index=False)

    X = feat_matrix.fillna(feat_matrix.mean()).values
    from scipy.spatial.distance import pdist, squareform
    D_euc = squareform(pdist(X, metric="euclidean"))
    D_cos = squareform(pdist(X, metric="cosine"))
    dist_df = pd.DataFrame(D_euc, index=regions, columns=regions)
    save_table(dist_df, out / "phonetic_distance_matrix.csv")
    save_table(pd.DataFrame(D_cos, index=regions, columns=regions), out / "phonetic_distance_matrix_cosine.csv")

    # -- bootstrap CI (resample FEATURES within each block, proxy for measurement uncertainty
    #    since T18 works on already-aggregated per-region values, not raw tokens) --
    rng = np.random.default_rng(42)
    n = len(regions)
    boot_stack = np.full((args.n_bootstrap, n, n), np.nan)
    cols = feat_matrix.columns.tolist()
    for b in range(args.n_bootstrap):
        if len(cols) < 2:
            break
        sel = rng.choice(cols, size=len(cols), replace=True)
        Xb = feat_matrix[sel].fillna(feat_matrix[sel].mean()).values
        boot_stack[b] = squareform(pdist(Xb, metric="euclidean"))
    ci_rows = []
    for i, j in itertools.combinations(range(n), 2):
        vals = boot_stack[:, i, j]
        vals = vals[~np.isnan(vals)]
        if len(vals) < 10:
            continue
        ci_rows.append({"region_a": regions[i], "region_b": regions[j], "observed": D_euc[i, j],
                        "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5))})
    save_table(pd.DataFrame(ci_rows), out / "phonetic_distance_ci.csv", index=False)

    # -- Ward clustering + cophenetic + branch support (bootstrap over features) --
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(squareform(D_euc, checks=False), method="ward")
    coph = cophenetic_corr(D_euc, Z)

    from scipy.cluster.hierarchy import fcluster
    orig_splits = set()
    def get_splits(Z, n_leaves):
        from scipy.cluster.hierarchy import to_tree
        tree = to_tree(Z)
        splits = []
        def recurse(node):
            if node.is_leaf():
                return {node.id}
            left = recurse(node.left)
            right = recurse(node.right)
            splits.append(frozenset(left | right))
            return left | right
        recurse(tree)
        return splits
    orig_splits = set(get_splits(Z, n))
    support_counts = {s: 0 for s in orig_splits}
    n_support = int(args.n_branch_support)
    for _ in range(n_support):
        sel = rng.choice(cols, size=len(cols), replace=True) if len(cols) else cols
        Xb = feat_matrix[sel].fillna(feat_matrix[sel].mean()).values if len(cols) else X
        Db = squareform(pdist(Xb, metric="euclidean"))
        Zb = linkage(squareform(Db, checks=False), method="ward")
        b_splits = set(get_splits(Zb, n))
        for s in orig_splits:
            if s in b_splits:
                support_counts[s] += 1
    branch_support = pd.DataFrame([
        {"split": ",".join(regions[i] for i in sorted(s)), "support_pct": 100 * support_counts[s] / n_support}
        for s in orig_splits
    ]).sort_values("support_pct", ascending=False)
    save_table(branch_support, out / "branch_support.csv", index=False)

    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 5))
    dendrogram(Z, labels=regions, ax=ax, leaf_rotation=40)
    ax.set_title(f"T18 -- phonetic-feature Ward dendrogram (cophenetic r={coph:.3f})")
    save_fig(fig, fig_dir / "dendrogram_phonetic.png")
    fig.savefig(fig_dir / "dendrogram_phonetic.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.0 * n + 1.5))
    im = ax.imshow(D_euc, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(regions, rotation=40, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(regions)
    ax.set_title("T18 -- fused phonetic distance heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, fig_dir / "phonetic_distance_heatmap.png")
    fig.savefig(fig_dir / "phonetic_distance_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # -- feature-block sensitivity: drop each block, recompute distance, correlate with full --
    sens_rows = []
    full_vec = squareform(D_euc, checks=True)
    for name in std_blocks:
        remaining = {k: v for k, v in std_blocks.items() if k != name}
        if not remaining:
            continue
        Xr = pd.concat(remaining.values(), axis=1)
        Xr = Xr.fillna(Xr.mean()).values
        Dr = squareform(pdist(Xr, metric="euclidean"))
        rho = np.corrcoef(squareform(Dr, checks=True), full_vec)[0, 1]
        sens_rows.append({"dropped_block": name, "corr_with_full_matrix": float(rho)})
    save_table(pd.DataFrame(sens_rows), out / "feature_block_sensitivity.csv", index=False)

    # -- PCA loadings --
    from sklearn.decomposition import PCA
    Xc = feat_matrix.fillna(feat_matrix.mean())
    pca = PCA(n_components=min(3, Xc.shape[1], Xc.shape[0] - 1))
    pca.fit(Xc.values)
    loadings = pd.DataFrame(pca.components_.T, index=Xc.columns,
                            columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    loadings["PC1_abs"] = loadings["PC1"].abs()
    loadings = loadings.sort_values("PC1_abs", ascending=False).drop(columns="PC1_abs")
    save_table(loadings.reset_index().rename(columns={"index": "feature"}), out / "pca_loadings.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    scores = pca.transform(Xc.values)
    ax.scatter(scores[:, 0], scores[:, 1])
    for i, r in enumerate(regions):
        ax.annotate(r, (scores[i, 0], scores[i, 1]))
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)" if pca.n_components_ > 1 else "PC2")
    ax.set_title("T18 -- PCA of the phonetic feature matrix")
    save_fig(fig, fig_dir / "pca_biplot.png")
    fig.savefig(fig_dir / "pca_biplot.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = base_summary("T18_phonetic_distance", cfg.config_path, len(regions), len(regions), count_unit="regions")
    summary.update({
        "feature_blocks_used": list(blocks.keys()),
        "feature_blocks_excluded": list(args.exclude_blocks),
        "n_features": feat_matrix.shape[1],
        "cophenetic_correlation": coph,
        "top_branch_support": branch_support.head(3).to_dict(orient="records") if len(branch_support) else [],
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T18 -- Fused Phonetic Distance Matrix")
    rep.p(
        f"Fused {len(blocks)} feature blocks ({', '.join(blocks.keys())}) -> {feat_matrix.shape[1]} "
        f"standardized features across 8 regions. Cophenetic correlation of the Ward dendrogram: "
        f"**{coph:.4f}**."
    )
    rep.h2("Feature blocks used")
    for name, df in blocks.items():
        rep.bullet(f"`{name}`: {df.shape[1]} feature(s)")
    rep.h2("Branch support (bootstrap over features)")
    rep.table(branch_support.round(1), index=False)
    rep.h2("Feature-block sensitivity")
    rep.table(pd.DataFrame(sens_rows).round(4), index=False)
    rep.h2("PCA")
    rep.p(f"PC1 explains {100*pca.explained_variance_ratio_[0]:.1f}% of variance. Top-loading features:")
    rep.table(loadings.head(10).round(3).reset_index().rename(columns={"index": "feature"}), index=False)
    rep.image("figures/dendrogram_phonetic.png", "Ward dendrogram of the fused phonetic distance matrix.")
    rep.image("figures/pca_biplot.png", "PCA of the region x feature matrix.")
    rep.h2("For the paper")
    rep.p("S7.7.")
    rep.write()

    print(f"[T18] done. blocks={list(blocks.keys())}  cophenetic={coph:.4f}  "
          f"pc1_var={100*pca.explained_variance_ratio_[0]:.1f}%")


if __name__ == "__main__":
    main()
