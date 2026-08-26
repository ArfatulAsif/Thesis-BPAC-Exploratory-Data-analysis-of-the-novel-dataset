"""Region accent-distance matrices.

Methods (config c1_distance.method):
  * ssl_controlled : pooled SSL embeddings with per-sentence centering (removes
                     content) and speaker-pair averaging (removes speaker); the
                     recommended, parallel-aware method.
  * ssl_pooled     : same but without per-sentence centering (legacy baseline).
  * loso_confusion : leave-one-speaker-out region classifier; distance derived
                     from how often regions are confused on UNSEEN speakers — the
                     most confound-resistant "are accents distinguishable" measure.
  * mfcc_dtw       : classic DTW on MFCCs (subsampled).

All return (regions, matrix) with matrix[i,j] = mean accent distance i<->j
(symmetric; diagonal = within-region spread for the pooled methods).
"""
import itertools

import numpy as np
import pandas as pd


def _standardize(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def _per_sentence_center(meta, X):
    """Subtract, from each row, the mean embedding of its sentence (across all
    speakers/regions). Removes the content axis, leaving accent + speaker."""
    Xc = X.copy().astype(np.float64)
    meta = meta.reset_index(drop=True)
    for s, idx in meta.groupby("sentence_id").groups.items():
        ii = np.array(list(idx))
        Xc[ii] -= Xc[ii].mean(axis=0, keepdims=True)
    return Xc.astype(np.float32)


def _pair_mean(A, B, metric):
    from scipy.spatial.distance import cdist
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    return float(cdist(A, B, metric=metric).mean())


def region_matrix_pooled(meta, X, metric="cosine", standardize=True, per_sentence_center=False):
    if per_sentence_center:
        X = _per_sentence_center(meta, X)
    if standardize:
        X = _standardize(X)
    meta = meta.reset_index(drop=True)
    regions = sorted(meta["region"].unique())
    idx = {r: i for i, r in enumerate(regions)}
    n = len(regions)
    acc = np.zeros((n, n)); cnt = np.zeros((n, n))
    groups = {}
    for i, row in enumerate(meta.itertuples(index=False)):
        groups.setdefault((row.sentence_id, row.region), []).append(i)
    for s in sorted(meta["sentence_id"].unique()):
        present = [r for r in regions if (s, r) in groups]
        for a, b in itertools.combinations_with_replacement(present, 2):
            Ia, Ib = groups[(s, a)], groups[(s, b)]
            if a == b:
                from scipy.spatial.distance import pdist
                dd = pdist(X[Ia], metric=metric)
                d = float(dd.mean()) if dd.size else np.nan
            else:
                d = _pair_mean(X[Ia], X[Ib], metric)
            if not np.isnan(d):
                i, j = idx[a], idx[b]
                acc[i, j] += d; cnt[i, j] += 1
                if i != j:
                    acc[j, i] += d; cnt[j, i] += 1
    with np.errstate(invalid="ignore"):
        mat = np.where(cnt > 0, acc / cnt, np.nan)
    return regions, mat


def _make_clf(kind):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, multi_class="auto")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis()


def loso_region_confusion(meta, X, classifier="lda", standardize=True):
    """Leave-one-speaker-out region classification. Returns (regions, conf_matrix,
    accuracy, chance). conf_matrix is row-normalized P(pred | true)."""
    from sklearn.preprocessing import StandardScaler

    meta = meta.reset_index(drop=True)
    regions = sorted(meta["region"].unique())
    ridx = {r: i for i, r in enumerate(regions)}
    y = meta["region"].map(ridx).to_numpy()
    spk = meta["speaker_uid"].to_numpy()
    Xz = StandardScaler().fit_transform(X) if standardize else np.asarray(X)

    n = len(regions)
    conf = np.zeros((n, n), dtype=np.float64)
    correct = total = 0
    for s in np.unique(spk):
        te = spk == s
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = _make_clf(classifier)
        try:
            clf.fit(Xz[tr], y[tr])
            pred = clf.predict(Xz[te])
        except Exception:
            continue
        for t, p in zip(y[te], pred):
            conf[t, p] += 1
            total += 1
            correct += int(t == p)
    acc = correct / total if total else float("nan")
    chance = 1.0 / n
    row_sums = conf.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        conf_norm = np.where(row_sums > 0, conf / row_sums, 0.0)
    return regions, conf_norm, acc, chance


def confusion_to_distance(regions, conf_norm):
    """Turn a row-normalized confusion matrix into a symmetric distance:
    often-confused regions are close. d(i,j) = 1 - (P(i->j)+P(j->i))/2."""
    n = len(regions)
    sim = (conf_norm + conf_norm.T) / 2.0
    mat = 1.0 - sim
    np.fill_diagonal(mat, 0.0)
    return mat


def region_matrix_mfcc_dtw(meta, cfg):
    import librosa
    rng = np.random.default_rng(0)
    regions = sorted(meta["region"].unique())
    idx = {r: i for i, r in enumerate(regions)}
    n = len(regions)
    acc = np.zeros((n, n)); cnt = np.zeros((n, n))
    cd = cfg["c1_distance"]
    max_sent = int(cd.get("dtw_max_sentences", 60))
    max_spk = int(cd.get("dtw_max_speakers_per_region", 4))
    n_mfcc = int(cd.get("mfcc_n", 13))
    sr = cfg.target_sr; top_db = cfg.top_db

    sent_ids = sorted(meta["sentence_id"].unique())
    if len(sent_ids) > max_sent:
        sent_ids = list(rng.choice(sent_ids, size=max_sent, replace=False))
    region_speakers = {r: sorted(meta[meta["region"] == r]["speaker_uid"].unique()) for r in regions}
    for r in regions:
        if len(region_speakers[r]) > max_spk:
            region_speakers[r] = list(rng.choice(region_speakers[r], size=max_spk, replace=False))

    def mfcc(path):
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        if y.size:
            yt, _ = librosa.effects.trim(y, top_db=top_db)
            if yt.size:
                y = yt
        if y.size < max(400, int(0.2 * sr)):
            return None
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T

    def dtw_dist(a, b):
        D, _ = librosa.sequence.dtw(X=a.T, Y=b.T, metric="euclidean")
        return float(D[-1, -1] / (a.shape[0] + b.shape[0]))

    fpath = dict(zip(meta["utterance_id"], meta["filepath"]))
    mcache = {}

    def gm(uid):
        if uid not in mcache:
            mcache[uid] = mfcc(fpath[uid])
        return mcache[uid]

    sel = meta[meta["sentence_id"].isin(sent_ids)]
    for s in sent_ids:
        sub = sel[sel["sentence_id"] == s]
        for a, b in itertools.combinations_with_replacement(regions, 2):
            A_ids = sub[(sub["region"] == a) & (sub["speaker_uid"].isin(region_speakers[a]))]["utterance_id"].tolist()
            B_ids = sub[(sub["region"] == b) & (sub["speaker_uid"].isin(region_speakers[b]))]["utterance_id"].tolist()
            ds = []
            for ua in A_ids:
                for ub in B_ids:
                    if a == b and ua >= ub:
                        continue
                    ma, mb = gm(ua), gm(ub)
                    if ma is None or mb is None:
                        continue
                    ds.append(dtw_dist(ma, mb))
            if ds:
                i, j = idx[a], idx[b]
                acc[i, j] += np.mean(ds); cnt[i, j] += 1
                if i != j:
                    acc[j, i] += np.mean(ds); cnt[j, i] += 1
    with np.errstate(invalid="ignore"):
        mat = np.where(cnt > 0, acc / cnt, np.nan)
    return regions, mat


def distance_from_standard(regions, mat, standard_region):
    if standard_region not in regions:
        return pd.DataFrame(columns=["region", "distance_from_standard"])
    si = regions.index(standard_region)
    rows = [{"region": r, "distance_from_standard": float(mat[si, regions.index(r)])}
            for r in regions if r != standard_region]
    return pd.DataFrame(rows).sort_values("distance_from_standard", ascending=False).reset_index(drop=True)
