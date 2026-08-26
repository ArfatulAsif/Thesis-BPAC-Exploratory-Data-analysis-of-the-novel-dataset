#!/usr/bin/env python3
"""C1 (v2) — Region accent-distance matrix on the CLEAN corpus.

Methods (config c1_distance.method):
  ssl_controlled  : middle-layer WavLM, per-sentence centering + speaker-pair
                    averaging (parallel-aware; recommended)
  ssl_pooled      : pooled WavLM without per-sentence centering (baseline)
  loso_confusion  : leave-one-speaker-out region-classifier confusion -> distance
  mfcc_dtw        : classic DTW on MFCCs (subsampled)

Outputs (in ./reports): distance_matrix.csv, distance_from_standard.csv,
distance_heatmap.png, dendrogram.png, summary.json, report.md
(+ loso_confusion.csv and loso accuracy when method=loso_confusion).

Run:  python C1_Region_accent_distance_matrix/analyze.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bpac_accent.cli import add_config_arg  # noqa: E402
from bpac_accent.distances import (confusion_to_distance, distance_from_standard,  # noqa: E402
                                   loso_region_confusion, region_matrix_mfcc_dtw,
                                   region_matrix_pooled)
from bpac_accent.embeddings import extract_embeddings  # noqa: E402
from bpac_accent.reporting import ReportWriter, ensure_dir, save_fig, save_json, save_table  # noqa: E402
from bpac_accent.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def _heatmap(mat, regions, title, path):
    fig, ax = plt.subplots(figsize=(1.1 * len(regions) + 2, 1.0 * len(regions) + 1.5))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=40, ha="right")
    ax.set_yticks(range(len(regions))); ax.set_yticklabels(regions)
    for i in range(len(regions)):
        for j in range(len(regions)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, path)


def main():
    ap = argparse.ArgumentParser(description="C1 region accent-distance matrix (clean corpus)")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    c1 = cfg["c1_distance"]
    method = c1.get("method", "ssl_controlled")
    metric = c1.get("metric", "cosine")
    standard = c1.get("standard_region", "dhaka")
    loso = None

    if method == "mfcc_dtw":
        from bpac_accent.manifest import build_manifest
        meta = build_manifest(cfg.data_root, cfg.bpac_subdir, cfg.sentences_csv,
                              verbose=cfg.verbose,
                              verified_manifest_path=getattr(cfg, "verified_manifest", None)
                              ).dropna(subset=["sentence"]).reset_index(drop=True)
        regions, mat = region_matrix_mfcc_dtw(meta, cfg)
        feat_desc = f"MFCC({c1.get('mfcc_n',13)}) + DTW (subsampled)"
    else:
        meta, X = extract_embeddings(cfg, "ssl")
        if method == "loso_confusion":
            regions, conf, acc, chance = loso_region_confusion(
                meta, X, classifier=c1.get("loso_classifier", "lda"),
                standardize=c1.get("standardize", True))
            mat = confusion_to_distance(regions, conf)
            loso = {"accuracy": acc, "chance": chance}
            save_table(pd.DataFrame(conf, index=regions, columns=regions).round(3),
                       out / "loso_confusion.csv")
            feat_desc = (f"{cfg.ssl_id} (layer={cfg.ssl_layer}) LOSO "
                         f"{c1.get('loso_classifier','lda')} confusion -> distance")
        else:
            psc = (method == "ssl_controlled") and c1.get("per_sentence_center", True)
            regions, mat = region_matrix_pooled(meta, X, metric=metric,
                                                standardize=c1.get("standardize", True),
                                                per_sentence_center=psc)
            feat_desc = (f"{cfg.ssl_id} (layer={cfg.ssl_layer}) pooled ({cfg.ssl_pooling}), "
                         f"{metric}" + (", per-sentence-centered" if psc else ""))

    mat_df = pd.DataFrame(mat, index=regions, columns=regions)
    save_table(mat_df, out / "distance_matrix.csv")
    ranking = distance_from_standard(regions, mat, standard)
    save_table(ranking, out / "distance_from_standard.csv", index=False)
    _heatmap(mat, regions, f"C1 region accent-distance ({method})", out / "distance_heatmap.png")

    dendro_ok = False
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        D = (mat + mat.T) / 2.0
        np.fill_diagonal(D, 0.0)
        D = np.nan_to_num(D, nan=np.nanmax(D))
        Z = linkage(squareform(D, checks=False), method="average")
        fig, ax = plt.subplots(figsize=(1.2 * len(regions) + 3, 5))
        dendrogram(Z, labels=regions, ax=ax, leaf_rotation=40)
        ax.set_title(f"C1 accent dendrogram ({method})"); ax.set_ylabel("distance")
        save_fig(fig, out / "dendrogram.png")
        dendro_ok = True
    except Exception as e:
        print(f"[C1] dendrogram skipped: {e}")

    pairs = [(regions[i], regions[j], float(mat[i, j]))
             for i in range(len(regions)) for j in range(i + 1, len(regions))
             if not np.isnan(mat[i, j])]
    pairs.sort(key=lambda t: t[2])
    summary = {
        "method": method, "features": feat_desc, "data_root": str(cfg.data_root),
        "n_regions": len(regions), "standard_region": standard,
        "closest_pair": list(pairs[0][:2]) + [round(pairs[0][2], 4)] if pairs else None,
        "farthest_pair": list(pairs[-1][:2]) + [round(pairs[-1][2], 4)] if pairs else None,
        "loso": loso,
    }
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "C1 (v2) — Region Accent-Distance Matrix")
    rep.p(f"**Features:** {feat_desc}.")
    rep.p(f"**Corpus:** clean tree at `{cfg.data_root}` (uniform 16 kHz mono).")
    if loso:
        rep.h2("Leave-one-speaker-out region accuracy")
        rep.bullet(f"Accuracy on **unseen speakers**: **{loso['accuracy']:.3f}** "
                   f"(chance {loso['chance']:.3f}). Above chance ⇒ accents are "
                   f"distinguishable across speakers (not speaker memorization).")
    rep.h2("Headline")
    if summary["closest_pair"]:
        rep.bullet(f"Closest: **{summary['closest_pair'][0]} ↔ {summary['closest_pair'][1]}** "
                   f"({summary['closest_pair'][2]}).")
        rep.bullet(f"Farthest: **{summary['farthest_pair'][0]} ↔ {summary['farthest_pair'][1]}** "
                   f"({summary['farthest_pair'][2]}).")
    rep.h2("Distance matrix")
    rep.table(mat_df.round(3))
    rep.h2(f"Distance from standard ({standard})")
    rep.table(ranking.round(4), index=False)
    rep.image("distance_heatmap.png", "Region x region accent distance.")
    if dendro_ok:
        rep.image("dendrogram.png", "Hierarchical clustering of accents.")
    rep.h2("Note")
    rep.bullet("Now that audio format is uniform, a surviving cluster is dialectal "
               "rather than a recording-pipeline artifact.")
    rep.write()

    print(f"[C1] method={method} closest={summary['closest_pair']} "
          f"farthest={summary['farthest_pair']}"
          + (f" loso_acc={loso['accuracy']:.3f}" if loso else ""))


if __name__ == "__main__":
    main()
