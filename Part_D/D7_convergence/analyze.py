#!/usr/bin/env python3
"""T19 -- convergence test: phonetic (T18) vs. neural-embedding (Part_C/C1)
distance matrices. The headline figure/statistic of the paper.

Two distance matrices built by completely independent methods (opaque
self-supervised embeddings vs. hand-measured phonetics, no shared code path
or assumptions) -- if they recover the same regional structure, that is
strong mutual validation: the neural model is picking up real phonetic
differences, and the phonetic measurements capture what the model sees.

Outputs (in ./reports): mantel_results.json, procrustes.json,
per_region_agreement.csv, geographic_agreement.json, figures/*, report.md,
summary.json.
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

from common.provenance import base_summary  # noqa: E402

from bpac_phon.cli import add_config_arg  # noqa: E402
from bpac_phon.reporting import ReportWriter, ensure_dir, save_fig, save_json, save_table  # noqa: E402
from bpac_phon.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

sys.path.insert(0, os.path.join(_REPO_ROOT, "Part_C", "C4_distance_statistics"))
from analyze import geographic_distance_matrix, mantel_test  # noqa: E402


def align_matrix(df, regions):
    return df.reindex(index=regions, columns=regions)


def main():
    ap = argparse.ArgumentParser(description="T19 convergence test")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--n_permutations", type=int, default=10000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")
    d7 = cfg["d7_convergence"]

    emb_path = os.path.join(_REPO_ROOT, "Part_C", "C1_Region_accent_distance_matrix", "reports", "distance_matrix.csv")
    phon_path = os.path.join(_REPO_ROOT, "Part_D", "D6_phonetic_distance", "reports", "phonetic_distance_matrix.csv")
    if not os.path.exists(emb_path) or not os.path.exists(phon_path):
        raise SystemExit(f"Need both {emb_path} and {phon_path} -- run Part_C/C1 and Part_D/D6 first.")
    emb_df = pd.read_csv(emb_path, index_col=0)
    phon_df = pd.read_csv(phon_path, index_col=0)

    regions = sorted(set(emb_df.index) & set(phon_df.index))
    emb = align_matrix(emb_df, regions).values.astype(float)
    phon = align_matrix(phon_df, regions).values.astype(float)
    np.fill_diagonal(emb, 0.0)
    np.fill_diagonal(phon, 0.0)

    n_perm = int(d7.get("n_permutations", args.n_permutations))
    mantel_pearson = mantel_test(emb, phon, n_perm=n_perm, method="pearson")
    mantel_spearman = mantel_test(emb, phon, n_perm=n_perm, method="spearman")
    save_json({"pearson": mantel_pearson, "spearman": mantel_spearman}, out / "mantel_results.json")

    # -- Procrustes: MDS both to 2D, align, report residual --
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    emb_2d = mds.fit_transform(np.nan_to_num(emb, nan=np.nanmax(emb)))
    phon_2d = mds.fit_transform(np.nan_to_num(phon, nan=np.nanmax(phon)))
    from scipy.spatial import procrustes
    mtx1, mtx2, disparity = procrustes(emb_2d, phon_2d)
    save_json({"disparity": float(disparity)}, out / "procrustes.json")

    # -- per-region agreement: correlate each region's distance PROFILE (row) between matrices --
    rows = []
    for i, r in enumerate(regions):
        rho = np.corrcoef(emb[i], phon[i])[0, 1]
        rows.append({"region": r, "row_correlation": float(rho)})
    per_region = pd.DataFrame(rows).sort_values("row_correlation", ascending=False)
    save_table(per_region, out / "per_region_agreement.csv", index=False)

    # -- geographic three-way check --
    geo = geographic_distance_matrix(regions)
    geo_emb = mantel_test(geo, emb, n_perm=n_perm, method="pearson")
    geo_phon = mantel_test(geo, phon, n_perm=n_perm, method="pearson")
    save_json({"geo_vs_embedding": geo_emb, "geo_vs_phonetic": geo_phon}, out / "geographic_agreement.json")

    # -- tanglegram --
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    Z_emb = linkage(squareform(emb, checks=False), method="average")
    Z_phon = linkage(squareform(phon, checks=False), method="ward")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    dendrogram(Z_emb, labels=regions, ax=axes[0], leaf_rotation=40, orientation="left")
    axes[0].set_title("Neural embedding (Part_C/C1)")
    dendrogram(Z_phon, labels=regions, ax=axes[1], leaf_rotation=40)
    axes[1].set_title("Phonetic (Part_D/D6)")
    fig.suptitle(f"T19 -- dendrograms side by side "
                f"(Mantel r={mantel_pearson['observed_r']:.3f}, p={mantel_pearson['p_value']:.4f})")
    save_fig(fig, fig_dir / "dendrograms_side_by_side.png")
    fig.savefig(fig_dir / "dendrograms_side_by_side.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(mtx1[:, 0], mtx1[:, 1], color="C0", label="embedding (Procrustes-aligned)")
    ax.scatter(mtx2[:, 0], mtx2[:, 1], color="C1", label="phonetic (Procrustes-aligned)")
    for i, r in enumerate(regions):
        ax.plot([mtx1[i, 0], mtx2[i, 0]], [mtx1[i, 1], mtx2[i, 1]], color="gray", linewidth=0.7)
        ax.annotate(r, mtx1[i], fontsize=7)
    ax.set_title(f"T19 -- MDS + Procrustes overlay (disparity={disparity:.4f})")
    ax.legend()
    save_fig(fig, fig_dir / "mds_procrustes_overlay.png")
    fig.savefig(fig_dir / "mds_procrustes_overlay.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = base_summary("T19_convergence", cfg.config_path, len(regions), len(regions), count_unit="regions")
    summary.update({
        "mantel_pearson": mantel_pearson, "mantel_spearman": mantel_spearman,
        "procrustes_disparity": float(disparity),
        "per_region_agreement": per_region.to_dict(orient="records"),
        "geo_vs_embedding": geo_emb, "geo_vs_phonetic": geo_phon,
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T19 -- Convergence Test: Phonetic vs. Neural-Embedding Distance")
    rep.p(
        "Two independently-derived 8x8 region distance matrices -- one from opaque self-supervised "
        "embeddings (Part_C/C1), one from hand-measured phonetics (Part_D/D6) -- compared directly. "
        "This is the headline statistic of the paper."
    )
    rep.h2("Mantel test")
    rep.bullet(f"Pearson r={mantel_pearson['observed_r']:.3f}, p={mantel_pearson['p_value']:.4f} "
              f"({mantel_pearson['n_permutations']} permutations)")
    rep.bullet(f"Spearman r={mantel_spearman['observed_r']:.3f}, p={mantel_spearman['p_value']:.4f}")
    rep.h2("Procrustes")
    rep.bullet(f"Disparity (MDS 2D alignment residual): **{disparity:.4f}** (lower = better geometric agreement)")
    rep.h2("Per-region agreement")
    rep.table(per_region.round(3), index=False)
    rep.h2("Three-way check against geography (T10)")
    rep.bullet(f"geography vs. embedding: r={geo_emb['observed_r']:.3f}, p={geo_emb['p_value']:.4f}")
    rep.bullet(f"geography vs. phonetic: r={geo_phon['observed_r']:.3f}, p={geo_phon['p_value']:.4f}")
    rep.h2("Figures")
    rep.image("figures/dendrograms_side_by_side.png", "Neural vs. phonetic dendrograms side by side.")
    rep.image("figures/mds_procrustes_overlay.png", "MDS + Procrustes overlay -- MAIN PAPER FIGURE.")
    rep.h2("For the paper")
    rep.p("S7.8 and the abstract. This is the headline figure.")
    rep.write()

    print(f"[T19] done. mantel_pearson_r={mantel_pearson['observed_r']:.3f} p={mantel_pearson['p_value']:.4f}  "
          f"procrustes_disparity={disparity:.4f}")


if __name__ == "__main__":
    main()
