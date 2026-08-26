#!/usr/bin/env python3
"""C2 (v2) — Speaker-vs-accent separability on the CLEAN corpus.

Beyond the basic silhouette, this adds the analyses that actually settle the
confound:
  * speaker-RESIDUALIZED content space (subtract each speaker's centroid) — is
    there region structure left once speaker timbre is removed?
  * leave-one-speaker-out (LOSO) region classification — can region be predicted
    on UNSEEN speakers? (accuracy >> chance ⇒ accent generalizes, not memorized)
  * per-sentence region silhouette — controls content via the parallel design
  * paired t-SNE/UMAP projections (by region, by speaker) for the content space
    and the speaker-residualized content space.

Run:  python C2_Speaker_vs_accent_embedding_separability/analyze.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bpac_accent.cli import add_config_arg  # noqa: E402
from bpac_accent.embeddings import extract_embeddings  # noqa: E402
from bpac_accent.reporting import ReportWriter, ensure_dir, save_fig, save_json, save_table  # noqa: E402
from bpac_accent.separability import (loso_region_classification,  # noqa: E402
                                      per_sentence_region_silhouette,
                                      residualize_by_speaker, silhouette, standardize)
from bpac_accent.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def _subsample(meta, X, per_speaker, seed):
    if not per_speaker:
        return meta.reset_index(drop=True), X
    rng = np.random.default_rng(seed)
    keep = []
    m = meta.reset_index(drop=True)
    for spk, grp in m.groupby("speaker_uid"):
        idx = grp.index.to_numpy()
        if len(idx) > per_speaker:
            idx = rng.choice(idx, size=per_speaker, replace=False)
        keep.extend(idx.tolist())
    keep = sorted(keep)
    return m.loc[keep].reset_index(drop=True), X[keep]


def _project(X, proj, cfg):
    seed = cfg["c2_separability"].get("random_state", 42)
    if proj == "tsne":
        from sklearn.manifold import TSNE
        perp = min(cfg["c2_separability"].get("tsne_perplexity", 30), max(5, (len(X) - 1) // 3))
        return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(X)
    if proj == "umap":
        import umap
        return umap.UMAP(n_neighbors=cfg["c2_separability"].get("umap_neighbors", 15),
                         min_dist=cfg["c2_separability"].get("umap_min_dist", 0.1),
                         random_state=seed).fit_transform(X)
    raise ValueError(proj)


def _scatter(emb2d, labels, title, path, legend_max=20):
    cats = sorted(pd.Series(labels).unique())
    cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
    color = {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in cats:
        m = np.array(labels) == c
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=10, color=color[c], label=str(c), alpha=0.7)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    if len(cats) <= legend_max:
        ax.legend(fontsize=6, markerscale=1.5, ncol=2, loc="best")
    save_fig(fig, path)


def main():
    ap = argparse.ArgumentParser(description="C2 speaker-vs-accent separability (clean corpus)")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    c2 = cfg["c2_separability"]
    proj_mode = c2.get("projection", "both")
    std = c2.get("standardize", True)
    seed = c2.get("random_state", 42)

    spk_meta, spk_X = extract_embeddings(cfg, "speaker")
    ssl_meta, ssl_X = extract_embeddings(cfg, "ssl")

    # speaker-residualized content space
    ssl_resid = residualize_by_speaker(ssl_meta, ssl_X) if c2.get("residualize_speaker", True) else None

    # ---- silhouette table ----
    def sil_row(name, meta, X):
        Xs = standardize(X) if std else X
        return {"embedding_space": name,
                "silhouette_by_region": round(silhouette(Xs, meta["region"].values), 4),
                "silhouette_by_speaker": round(silhouette(Xs, meta["speaker_uid"].values), 4),
                "n": len(meta)}
    sil_rows = [sil_row("speaker_ECAPA", spk_meta, spk_X),
                sil_row("content_WavLM", ssl_meta, ssl_X)]
    if ssl_resid is not None:
        sil_rows.append(sil_row("content_WavLM_speaker_residualized", ssl_meta, ssl_resid))
    sil = pd.DataFrame(sil_rows)
    save_table(sil, out / "silhouette_scores.csv", index=False)

    # ---- per-sentence region silhouette (controls content) ----
    ps_val, ps_n = (per_sentence_region_silhouette(ssl_meta, ssl_X, standardize_x=std)
                    if c2.get("per_sentence_silhouette", True) else (float("nan"), 0))

    # ---- LOSO region classification (content space) ----
    loso = loso_region_classification(ssl_meta, ssl_X,
                                      classifier=c2.get("loso_region_classifier", "lda"),
                                      standardize_x=std)
    save_table(loso["confusion"], out / "loso_region_confusion.csv")

    # ---- projections (speaker + content spaces; residualized space is omitted
    # because, with speakers nested in regions, it is structure-free by design) ----
    spaces = {"speaker_ECAPA": (spk_meta, spk_X), "content_WavLM": (ssl_meta, ssl_X)}
    projs = ["tsne", "umap"] if proj_mode == "both" else [proj_mode]
    made = []
    for name, (meta, X) in spaces.items():
        m2, X2 = _subsample(meta, X, c2.get("sample_per_speaker"), seed)
        Xs = standardize(X2) if std else X2
        for proj in projs:
            try:
                emb2d = _project(Xs, proj, cfg)
            except Exception as e:
                print(f"[C2] {proj} on {name} skipped: {e}")
                continue
            p1 = out / f"{name}_{proj}_by_region.png"
            p2 = out / f"{name}_{proj}_by_speaker.png"
            _scatter(emb2d, m2["region"].values, f"{name} — {proj.upper()} by region", p1)
            _scatter(emb2d, m2["speaker_uid"].values, f"{name} — {proj.upper()} by speaker", p2)
            made.append((name, proj, p1.name, p2.name))

    # ---- verdict (now evidence-based, using LOSO + residualized silhouette) ----
    resid_region_sil = (sil[sil["embedding_space"] == "content_WavLM_speaker_residualized"]
                        ["silhouette_by_region"].iloc[0] if ssl_resid is not None else float("nan"))
    acc, chance = loso["accuracy"], loso["chance"]
    lift = (acc / chance) if (chance and not np.isnan(acc)) else float("nan")
    if np.isnan(acc):
        verdict = "Could not run LOSO classification (too few speakers/regions)."
    elif acc >= 2.0 * chance:
        verdict = (f"Region is predictable on UNSEEN speakers (LOSO accuracy {acc:.3f} "
                   f"vs chance {chance:.3f}, ×{lift:.1f}). Accent generalizes across "
                   f"speakers ⇒ the accent signal is GENUINE, not speaker identity.")
    elif acc >= 1.3 * chance:
        verdict = (f"Region is somewhat predictable on unseen speakers (LOSO {acc:.3f} "
                   f"vs chance {chance:.3f}). A real but modest accent signal; report "
                   f"it with the speaker-count caveat.")
    else:
        verdict = (f"Region is barely predictable on unseen speakers (LOSO {acc:.3f} "
                   f"vs chance {chance:.3f}). Even on clean audio the accent axis is "
                   f"weak relative to speaker identity — state this confound honestly.")

    summary = {
        "ssl_model": cfg.ssl_id, "ssl_layer": cfg.ssl_layer, "speaker_model": cfg.speaker_source,
        "data_root": str(cfg.data_root),
        "silhouette": sil.set_index("embedding_space").to_dict(orient="index"),
        "per_sentence_region_silhouette": round(ps_val, 4), "per_sentence_n": ps_n,
        "loso_region_accuracy": round(acc, 4) if not np.isnan(acc) else None,
        "loso_chance": round(chance, 4), "loso_macro_f1": round(loso["macro_f1"], 4)
        if not np.isnan(loso["macro_f1"]) else None,
        "loso_per_region_recall": loso["per_region_recall"],
        "residualized_region_silhouette": round(resid_region_sil, 4)
        if not np.isnan(resid_region_sil) else None,
        "verdict": verdict,
    }
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "C2 (v2) — Speaker-vs-Accent Separability")
    rep.p(f"**Speaker:** `{cfg.speaker_source}` (ECAPA). **Content:** `{cfg.ssl_id}` "
          f"(WavLM, layer={cfg.ssl_layer}). **Corpus:** clean `{cfg.data_root}`.")
    rep.h2("Leave-one-speaker-out region classification (the confound test)")
    rep.bullet(f"Accuracy on **unseen speakers**: **{acc:.3f}** vs chance "
               f"**{chance:.3f}** (×{lift:.1f}), macro-F1 "
               f"{summary['loso_macro_f1']}. High ⇒ accent generalizes across "
               f"speakers and is not speaker memorization.")
    rep.bullet(f"Per-region recall (unseen speakers): {loso['per_region_recall']}.")
    rep.h2("Verdict")
    rep.p(verdict)
    rep.h2("Silhouette scores")
    rep.p("`*_residualized` removes each speaker's centroid. **Caveat:** because "
          "every speaker belongs to one region (speakers are *nested* in regions), "
          "removing the speaker mean also removes the region mean — so a near-zero "
          "residualized region score is **expected by design and does not mean "
          "accent is absent**. It quantifies how completely speaker and accent are "
          "confounded in this corpus. The LOSO test above is the valid measure of "
          "accent, because predicting region for an *unseen* speaker requires a "
          "speaker-independent accent signal.")
    rep.table(sil, index=False)
    rep.bullet(f"Per-sentence region silhouette (controls content; still carries the "
               f"speaker confound): **{summary['per_sentence_region_silhouette']}** "
               f"over {ps_n} sentences.")
    rep.h2("Confusion (LOSO, row-normalized P(pred|true))")
    rep.table(loso["confusion"])
    rep.h2("Projections")
    rep.p("Coloured by region and by speaker, for content and "
          "speaker-residualized spaces.")
    for name, proj, r_png, s_png in made:
        rep.h3(f"{name} — {proj.upper()}")
        rep.image(r_png, f"{name} {proj.upper()} by region.")
        rep.image(s_png, f"{name} {proj.upper()} by speaker.")
    rep.write()

    print(f"[C2] LOSO region acc={acc:.3f} (chance {chance:.3f}) "
          f"resid_region_sil={summary['residualized_region_silhouette']} "
          f"per_sentence_sil={summary['per_sentence_region_silhouette']}")
    print(f"[C2] {verdict}")


if __name__ == "__main__":
    main()
