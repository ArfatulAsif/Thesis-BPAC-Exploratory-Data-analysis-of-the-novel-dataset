#!/usr/bin/env python3
"""T9 -- within-gender / within-pipeline LOSO region classification.

C2's pooled LOSO result (region predictable on unseen speakers, ~2x chance)
is the headline finding -- but `pipeline` is perfectly confounded with
`gender` (lab=male, app=female, invariant 3), so the obvious objection is
that the classifier might be detecting recording channel, not accent.
Re-running LOSO WITHIN each gender holds the channel constant (a within-male
run is automatically a within-lab run), so any surviving accuracy above
chance cannot be channel.

Compute note: LDA refit on full 1024-dim WavLM embeddings takes >1 min per
fold (>50 min per full LOSO run) -- too slow to repeat for 3 subsets plus a
1000x permutation test. All LOSO runs here use PCA(50)-reduced embeddings
instead (validated against C2's cached full-dim pooled number: 0.2511 full-dim
vs the PCA(50) pooled number computed here -- both ~2x chance, same
conclusion). The permutation test and bootstrap CI additionally cap
utterances/speaker for tractability; see config for exact numbers and the
resulting compute-budget-driven reduction from the spec's suggested 1000
permutations.

Outputs (in ./reports): loso_accuracy_by_subset.csv,
loso_confusion_{all,male,female}.csv, permutation_test.json,
silhouette_by_subset.csv, ecapa_control.json, figures/*, report.md, summary.json.
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

from bpac_accent.cli import add_config_arg  # noqa: E402
from bpac_accent.embeddings import extract_embeddings  # noqa: E402
from bpac_accent.reporting import (ReportWriter, ensure_dir, maybe_tqdm, save_fig,  # noqa: E402
                                   save_json, save_table)
from bpac_accent.separability import (bootstrap_ci_loso, loso_region_classification_pca,  # noqa: E402
                                      permutation_test_loso, region_chance_levels,
                                      region_classification_from_speaker_embedding, silhouette,
                                      standardize)
from bpac_accent.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def cap_per_speaker(meta, X, cap, seed):
    if not cap:
        return meta.reset_index(drop=True), X
    rng = np.random.default_rng(seed)
    keep = []
    m = meta.reset_index(drop=True)
    for _, grp in m.groupby("speaker_uid"):
        idx = grp.index.to_numpy()
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.extend(idx.tolist())
    keep = sorted(keep)
    return m.loc[keep].reset_index(drop=True), X[keep]


def subset_by_gender(meta, X, gender):
    if gender == "all":
        return meta.reset_index(drop=True), X
    mask = (meta["gender"] == gender).to_numpy()
    return meta[mask].reset_index(drop=True), X[mask]


def main():
    ap = argparse.ArgumentParser(description="T9 within-gender/within-pipeline LOSO")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")

    c3 = cfg.get("c3_loso_within_gender", {}) if isinstance(cfg, dict) else cfg["c3_loso_within_gender"]
    subsets = c3.get("loso_subsets", ["all", "male", "female"])
    pca_components = int(c3.get("pca_components", 50))
    perm_pca_components = int(c3.get("permutation_pca_components", 20))
    n_permutations = int(c3.get("n_permutations", 100))
    n_boot = int(c3.get("n_bootstrap", 100))
    cap_for_perm = int(c3.get("permutation_cap_per_speaker", 60))
    seed = int(c3.get("random_state", 42))
    classifier = c3.get("loso_classifier", "lda")

    ssl_meta, ssl_X = extract_embeddings(cfg, "ssl")
    spk_meta, spk_X = extract_embeddings(cfg, "speaker")
    n_in = len(ssl_meta)

    # ---- headline: LOSO accuracy per subset (PCA-50, validated vs C2's full-dim pooled number) ----
    rows = []
    confusions = {}
    for subset in subsets:
        meta_s, X_s = subset_by_gender(ssl_meta, ssl_X, subset)
        chance = region_chance_levels(meta_s)
        res = loso_region_classification_pca(meta_s, X_s, classifier=classifier, n_components=pca_components, seed=seed)
        confusions[subset] = res["confusion"]
        save_table(res["confusion"], out / f"loso_confusion_{subset}.csv")
        rows.append({
            "subset": subset, "n_utterances": len(meta_s), "n_speakers": meta_s["speaker_uid"].nunique(),
            "n_regions": meta_s["region"].nunique(), "accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
            "chance_uniform": chance["uniform"], "chance_majority_class": chance["majority_class"],
            "chance_stratified_random": chance["stratified_random"],
            "lift_vs_stratified": res["accuracy"] / chance["stratified_random"] if chance["stratified_random"] else np.nan,
        })
        print(f"[T9] subset={subset}: acc={res['accuracy']:.4f}  "
              f"stratified_chance={chance['stratified_random']:.4f}  "
              f"lift={rows[-1]['lift_vs_stratified']:.2f}x")
    loso_df = pd.DataFrame(rows)
    save_table(loso_df, out / "loso_accuracy_by_subset.csv", index=False)

    # ---- permutation test (pooled "all" subset -- the primary confound-defense number) ----
    meta_all, X_all = subset_by_gender(ssl_meta, ssl_X, "all")
    meta_capped, X_capped = cap_per_speaker(meta_all, X_all, cap_for_perm, seed)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Xz = StandardScaler().fit_transform(X_capped)
    Xp = PCA(n_components=perm_pca_components, random_state=seed).fit_transform(Xz)
    perm = permutation_test_loso(meta_capped, Xp, classifier=classifier, n_permutations=n_permutations,
                                 seed=seed, progress=maybe_tqdm)
    save_json(perm, out / "permutation_test.json")
    print(f"[T9] permutation test: observed={perm['observed_accuracy']:.4f}  "
          f"null_mean={perm['perm_accuracy_mean']:.4f}  p={perm['p_value']:.4f}")

    # ---- bootstrap CI (pooled "all" subset) ----
    boot = bootstrap_ci_loso(meta_capped, Xp, classifier=classifier, n_boot=n_boot, seed=seed, progress=maybe_tqdm)
    save_json(boot, out / "bootstrap_ci.json")
    print(f"[T9] bootstrap 95% CI: [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")

    # ---- silhouette by subset ----
    sil_rows = []
    for subset in subsets:
        meta_s, X_s = subset_by_gender(ssl_meta, ssl_X, subset)
        Xs = standardize(X_s)
        sil_rows.append({
            "subset": subset,
            "silhouette_by_region": round(silhouette(Xs, meta_s["region"].values), 4),
            "silhouette_by_speaker": round(silhouette(Xs, meta_s["speaker_uid"].values), 4),
        })
    sil_df = pd.DataFrame(sil_rows)
    save_table(sil_df, out / "silhouette_by_subset.csv", index=False)

    # ---- ECAPA (speaker embedding) control: does the SPEAKER space predict region too? ----
    ecapa_chance = region_chance_levels(spk_meta)
    ecapa_res = loso_region_classification_pca(spk_meta, spk_X, classifier=classifier, n_components=pca_components, seed=seed)
    ecapa_control = {
        "accuracy": ecapa_res["accuracy"], "macro_f1": ecapa_res["macro_f1"],
        "chance_stratified_random": ecapa_chance["stratified_random"],
        "lift": ecapa_res["accuracy"] / ecapa_chance["stratified_random"] if ecapa_chance["stratified_random"] else np.nan,
        "content_pooled_accuracy_for_comparison": float(loso_df.loc[loso_df["subset"] == "all", "accuracy"].iloc[0]),
    }
    save_json(ecapa_control, out / "ecapa_control.json")
    print(f"[T9] ECAPA (speaker) control: acc={ecapa_control['accuracy']:.4f} vs "
          f"content acc={ecapa_control['content_pooled_accuracy_for_comparison']:.4f}")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(loso_df))
    ax.bar(x, loso_df["accuracy"], color="C0", label="LOSO accuracy")
    ax.plot(x, loso_df["chance_stratified_random"], "o--", color="red", label="stratified-random chance")
    ax.set_xticks(x)
    ax.set_xticklabels(loso_df["subset"])
    ax.set_ylabel("region classification accuracy")
    ax.set_title("T9 -- LOSO accuracy by subset (PCA-50, +/- bootstrap CI on 'all')")
    if boot["n_boot_valid"]:
        # bootstrap CI is drawn from a different (speaker-resampled) computation than the
        # point estimate, so it need not straddle it -- draw as horizontal CI lines instead
        # of an errorbar (which requires non-negative offsets from the point).
        ax.axhline(boot["ci_low"], color="black", linestyle=":", linewidth=1, label="bootstrap 95% CI ('all')")
        ax.axhline(boot["ci_high"], color="black", linestyle=":", linewidth=1)
    ax.legend()
    save_fig(fig, fig_dir / "loso_accuracy_with_ci.png")
    fig.savefig(fig_dir / "loso_accuracy_with_ci.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(perm["perm_accuracies"], bins=20, color="lightgray", edgecolor="black", label="permuted null")
    ax.axvline(perm["observed_accuracy"], color="red", linewidth=2, label="observed")
    ax.set_xlabel("LOSO accuracy")
    ax.set_title(f"T9 -- permutation null distribution (p={perm['p_value']:.4f})")
    ax.legend()
    save_fig(fig, fig_dir / "permutation_null_distribution.png")
    fig.savefig(fig_dir / "permutation_null_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- interpretation ----
    acc_all = loso_df.loc[loso_df["subset"] == "all", "accuracy"].iloc[0]
    acc_male = loso_df.loc[loso_df["subset"] == "male", "accuracy"].iloc[0] if "male" in subsets else None
    acc_female = loso_df.loc[loso_df["subset"] == "female", "accuracy"].iloc[0] if "female" in subsets else None
    chance_all = loso_df.loc[loso_df["subset"] == "all", "chance_stratified_random"].iloc[0]
    within_gender_holds = all(
        (a is not None) and (a >= 1.3 * loso_df.loc[loso_df["subset"] == s, "chance_stratified_random"].iloc[0])
        for s, a in [("male", acc_male), ("female", acc_female)] if a is not None
    )
    if within_gender_holds:
        interpretation = (
            "Accuracy holds within gender (both male-only/lab-only and female-only/app-only LOSO "
            "runs stay above chance) -> channel is not driving the pooled result; the accent "
            "signal is real, though modest -- consistent with subtle-but-genuine regional variation "
            "in read Bangla speech."
        )
    elif (acc_male is not None and acc_male < 1.3 * loso_df.loc[loso_df["subset"] == "male", "chance_stratified_random"].iloc[0]) and \
         (acc_female is not None and acc_female < 1.3 * loso_df.loc[loso_df["subset"] == "female", "chance_stratified_random"].iloc[0]):
        interpretation = (
            "Accuracy collapses to chance within both gender subsets -> the pooled result was "
            "likely channel-driven (pipeline/gender confound). The paper's claim must narrow to "
            "'region separability under matched channel conditions is not established' -- report "
            "this honestly rather than the pooled headline number."
        )
    elif (acc_male is not None and acc_female is not None
          and acc_male >= 1.5 * loso_df.loc[loso_df["subset"] == "male", "chance_stratified_random"].iloc[0]
          and acc_female < 1.0 * loso_df.loc[loso_df["subset"] == "female", "chance_stratified_random"].iloc[0]):
        interpretation = (
            f"Asymmetric result: within male/lab speakers accuracy is STRONG ({acc_male:.3f} vs chance "
            f"{loso_df.loc[loso_df['subset']=='male','chance_stratified_random'].iloc[0]:.3f}) -- region is "
            f"genuinely predictable on unseen lab-recorded speakers, so the accent signal in that half of "
            f"the corpus is real, not channel. Within female/app speakers accuracy is AT OR BELOW chance "
            f"({acc_female:.3f} vs {loso_df.loc[loso_df['subset']=='female','chance_stratified_random'].iloc[0]:.3f}) "
            "-- either the app-recording channel carries less separable accent signal (compression, mic "
            "variability), or region is simply harder to detect from these speakers/regions with current "
            "methods. Do not report the pooled number as if it applies uniformly to the whole corpus -- "
            "report both halves and flag the female/app gap as an open question, not a null result to hide."
        )
    else:
        interpretation = (
            "Accuracy drops but stays above chance in at least one gender subset -> part of the "
            "pooled signal was channel, part is real accent. Report the reduced within-gender "
            "figure as the honest headline, not the pooled number."
        )

    summary = base_summary("T9_loso_within_gender", cfg.config_path, n_in, n_in)
    summary.update({
        "loso_by_subset": loso_df.to_dict(orient="records"),
        "permutation_p_value": perm["p_value"],
        "bootstrap_ci_all": [boot["ci_low"], boot["ci_high"]],
        "ecapa_control": ecapa_control,
        "interpretation": interpretation,
        "pca_components_headline": pca_components,
        "permutation_pca_components": perm_pca_components,
        "n_permutations": n_permutations,
        "compute_note": "n_permutations reduced from the spec's suggested 1000 to "
                        f"{n_permutations} for tractability (LDA refit cost); see module docstring.",
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T9 -- Within-Gender / Within-Pipeline LOSO")
    rep.p(
        "Because `pipeline` is perfectly confounded with `gender` (lab=male, app=female), the "
        "pooled LOSO result could in principle be detecting recording channel rather than accent. "
        "This reruns LOSO region classification separately within each gender, holding channel "
        "constant, plus a speaker-level permutation test, bootstrap CI, and an ECAPA "
        "(speaker-embedding) control."
    )
    rep.h2("Headline: LOSO accuracy by subset")
    rep.table(loso_df.round(4), index=False)
    rep.h2("Interpretation")
    rep.p(interpretation)
    rep.h2("Permutation test (pooled, speaker-level label shuffle)")
    rep.p(
        f"Observed accuracy **{perm['observed_accuracy']:.4f}** vs. permuted-null mean "
        f"{perm['perm_accuracy_mean']:.4f} (sd {perm['perm_accuracy_sd']:.4f}) over "
        f"{perm['n_permutations']} permutations -> **p={perm['p_value']:.4f}**. "
        f"(Reduced from the spec's suggested 1000 permutations for compute tractability -- "
        f"LDA refit is the bottleneck; see module docstring.)"
    )
    rep.h2("Bootstrap 95% CI (resampling speakers)")
    rep.p(f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] over {boot['n_boot_valid']} valid resamples.")
    rep.h2("ECAPA (speaker-embedding) control")
    rep.p(
        f"If the speaker embedding (trained for speaker verification, not content) predicts region "
        f"about as well as the content embedding, the signal is speaker identity, not accent. "
        f"ECAPA accuracy **{ecapa_control['accuracy']:.4f}** (lift {ecapa_control['lift']:.2f}x) vs. "
        f"content accuracy **{ecapa_control['content_pooled_accuracy_for_comparison']:.4f}**."
    )
    rep.h2("Silhouette by subset")
    rep.table(sil_df, index=False)
    rep.h2("Figures")
    rep.image("figures/loso_accuracy_with_ci.png", "LOSO accuracy by subset vs. stratified-random chance.")
    rep.image("figures/permutation_null_distribution.png", "Permutation null distribution vs. observed accuracy.")
    rep.h2("For the paper")
    rep.p(
        "S6.3-S6.4: this is the single most important defensive result in the paper. Lead with the "
        "within-gender result, not the pooled one, and report the permutation p-value alongside the "
        "accuracy so the claim is statistically defensible, not just 'twice chance'."
    )
    rep.write()

    print(f"[T9] done. interpretation: {interpretation}")


if __name__ == "__main__":
    main()
