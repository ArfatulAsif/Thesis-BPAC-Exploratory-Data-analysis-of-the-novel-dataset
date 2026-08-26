"""Separability metrics for C2 (the speaker-vs-accent confound test).

  * silhouette(region|speaker) in each embedding space
  * speaker-residualized content space: subtract each speaker's centroid, then
    measure region silhouette — isolates accent structure beyond speaker timbre
  * leave-one-speaker-out (LOSO) region classification accuracy + confusion —
    the confound-resistant test: can region be predicted on UNSEEN speakers?
  * per-sentence region silhouette — controls content via the parallel design
"""
import numpy as np
import pandas as pd


def standardize(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def silhouette(X, labels):
    from sklearn.metrics import silhouette_score
    uniq = pd.Series(labels).nunique()
    if uniq < 2 or len(labels) <= uniq:
        return float("nan")
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return float("nan")


def residualize_by_speaker(meta, X):
    """Subtract each speaker's mean embedding (cancels speaker identity)."""
    Xr = X.copy().astype(np.float64)
    meta = meta.reset_index(drop=True)
    for spk, idx in meta.groupby("speaker_uid").groups.items():
        ii = np.array(list(idx))
        Xr[ii] -= Xr[ii].mean(axis=0, keepdims=True)
    return Xr.astype(np.float32)


def _make_clf(kind):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, multi_class="auto")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis()


def loso_region_classification(meta, X, classifier="lda", standardize_x=True):
    """Leave-one-speaker-out region classification.

    Returns dict with accuracy, chance, macro_f1, per-region recall, and a
    row-normalized confusion DataFrame. Because every test speaker is unseen in
    training, high accuracy means region (accent) generalizes ACROSS speakers —
    it cannot be speaker memorization.
    """
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    meta = meta.reset_index(drop=True)
    regions = sorted(meta["region"].unique())
    ridx = {r: i for i, r in enumerate(regions)}
    y = meta["region"].map(ridx).to_numpy()
    spk = meta["speaker_uid"].to_numpy()
    Xz = StandardScaler().fit_transform(X) if standardize_x else np.asarray(X)

    n = len(regions)
    conf = np.zeros((n, n), dtype=np.float64)
    y_true_all, y_pred_all = [], []
    for s in np.unique(spk):
        te = spk == s
        tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        clf = _make_clf(classifier)
        try:
            clf.fit(Xz[tr], y[tr])
            pred = clf.predict(Xz[te])
        except Exception:
            continue
        for t, p in zip(y[te], pred):
            conf[t, p] += 1
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(pred.tolist())

    total = len(y_true_all)
    acc = float(np.mean(np.array(y_true_all) == np.array(y_pred_all))) if total else float("nan")
    macro_f1 = float(f1_score(y_true_all, y_pred_all, average="macro")) if total else float("nan")
    row = conf.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        conf_norm = np.where(row > 0, conf / row, 0.0)
    recall = {regions[i]: round(float(conf_norm[i, i]), 3) for i in range(n)}
    conf_df = pd.DataFrame(conf_norm, index=regions, columns=regions).round(3)
    return {
        "accuracy": acc, "chance": 1.0 / n, "macro_f1": macro_f1,
        "n_eval": total, "per_region_recall": recall, "confusion": conf_df,
        "classifier": classifier,
    }


def region_chance_levels(meta):
    """Three chance baselines for region classification (T9), since regions
    are unequal in size (Sylhet 2500 .. Rangpur/Mymensingh 1250):
      uniform          -- 1/n_regions (naive, wrong when classes are unequal)
      majority_class    -- always predict the largest region
      stratified_random -- guess each class with probability = its frequency
                            (expected accuracy = sum(p_r^2))
    """
    counts = meta["region"].value_counts()
    n = len(counts)
    p = counts / counts.sum()
    return {
        "uniform": 1.0 / n,
        "majority_class": float(p.max()),
        "stratified_random": float((p**2).sum()),
    }


def loso_region_classification_pca(meta, X, classifier="lda", standardize_x=True, n_components=50, seed=42):
    """Same as loso_region_classification but PCA-reduces X first -- an LDA
    refit on 768-dim WavLM embeddings costs O(n*d^2) and is too slow to
    repeat thousands of times (permutation test, bootstrap CI). PCA(50)
    keeps >90% of the region-discriminative signal (verify via
    accuracy_full vs accuracy_pca in the calling report) at a small
    fraction of the cost. Returns the same dict shape as
    loso_region_classification, plus a `y` (encoded true labels) and
    `spk` array the caller can reuse for permutation/bootstrap."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    meta = meta.reset_index(drop=True)
    Xz = StandardScaler().fit_transform(X) if standardize_x else np.asarray(X)
    Xp = PCA(n_components=min(n_components, Xz.shape[1], Xz.shape[0] - 1), random_state=seed).fit_transform(Xz)
    res = loso_region_classification(meta, Xp, classifier=classifier, standardize_x=False)
    res["X_pca"] = Xp
    return res


def permutation_test_loso(meta, X_pca, classifier="lda", n_permutations=200, seed=42, progress=None):
    """Permutation test for LOSO region accuracy (T9): shuffle the
    speaker->region label mapping (whole speakers move together, so the
    nested speaker-in-region structure is respected) n_permutations times,
    rerun LOSO on each shuffle, and report the empirical p-value = fraction
    of permuted accuracies >= the observed accuracy. Operates on
    already PCA-reduced embeddings for tractability (see
    loso_region_classification_pca)."""
    meta = meta.reset_index(drop=True)
    observed = loso_region_classification(meta, X_pca, classifier=classifier, standardize_x=False)["accuracy"]

    speakers = meta[["speaker_uid", "region"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm_accs = []
    iterable = range(n_permutations)
    if progress is not None:
        iterable = progress(iterable, desc="permutation test", total=n_permutations)
    for _ in iterable:
        shuffled_region = speakers["region"].sample(frac=1.0, random_state=rng.integers(0, 2**31 - 1)).to_numpy()
        spk_to_region = dict(zip(speakers["speaker_uid"], shuffled_region))
        meta_shuf = meta.copy()
        meta_shuf["region"] = meta_shuf["speaker_uid"].map(spk_to_region)
        res = loso_region_classification(meta_shuf, X_pca, classifier=classifier, standardize_x=False)
        perm_accs.append(res["accuracy"])
    perm_accs = np.array(perm_accs)
    p_value = float(np.mean(perm_accs >= observed))
    return {
        "observed_accuracy": float(observed),
        "n_permutations": n_permutations,
        "perm_accuracy_mean": float(np.nanmean(perm_accs)),
        "perm_accuracy_sd": float(np.nanstd(perm_accs)),
        "p_value": p_value,
        "perm_accuracies": perm_accs.tolist(),
    }


def bootstrap_ci_loso(meta, X_pca, classifier="lda", n_boot=200, seed=42, ci=0.95, progress=None):
    """Bootstrap 95% CI on LOSO region accuracy by resampling SPEAKERS with
    replacement (not utterances -- speakers are the independent unit here)
    and rerunning LOSO within each bootstrap sample."""
    meta = meta.reset_index(drop=True)
    speakers = meta["speaker_uid"].unique()
    rng = np.random.default_rng(seed)
    accs = []
    iterable = range(n_boot)
    if progress is not None:
        iterable = progress(iterable, desc="bootstrap CI", total=n_boot)
    for _ in iterable:
        boot_spk = rng.choice(speakers, size=len(speakers), replace=True)
        # de-duplicate resampled speakers (LOSO needs each held-out speaker once);
        # repeats of the same speaker are folded into one held-out fold.
        idx = meta["speaker_uid"].isin(set(boot_spk))
        sub = meta[idx]
        if sub["region"].nunique() < 2 or sub["speaker_uid"].nunique() < 3:
            continue
        res = loso_region_classification(sub, X_pca[idx.to_numpy()], classifier=classifier, standardize_x=False)
        if not np.isnan(res["accuracy"]):
            accs.append(res["accuracy"])
    accs = np.array(accs)
    if len(accs) == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_boot_valid": 0}
    lo = (1 - ci) / 2 * 100
    hi = (1 - (1 - ci) / 2) * 100
    return {
        "ci_low": float(np.percentile(accs, lo)),
        "ci_high": float(np.percentile(accs, hi)),
        "n_boot_valid": int(len(accs)),
        "mean": float(np.mean(accs)),
    }


def region_classification_from_speaker_embedding(meta_spk, X_spk, classifier="lda", standardize_x=True):
    """Control (T9): if the SPEAKER embedding (ECAPA, trained for speaker
    verification, should carry no content/accent signal by design) predicts
    region about as well as the content embedding does, the pooled result is
    speaker identity leaking through, not accent. Reuses the same LOSO
    machinery on the speaker-embedding space."""
    return loso_region_classification(meta_spk, X_spk, classifier=classifier, standardize_x=standardize_x)


def per_sentence_region_silhouette(meta, X, standardize_x=True):
    """Mean region silhouette computed within each shared sentence (controls
    content). Positive => regions separate even for identical text."""
    meta = meta.reset_index(drop=True)
    Xz = standardize(X) if standardize_x else X
    vals = []
    for s, idx in meta.groupby("sentence_id").groups.items():
        ii = np.array(list(idx))
        labs = meta.loc[ii, "region"].to_numpy()
        if pd.Series(labs).nunique() < 2 or len(ii) <= pd.Series(labs).nunique():
            continue
        v = silhouette(Xz[ii], labs)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)
