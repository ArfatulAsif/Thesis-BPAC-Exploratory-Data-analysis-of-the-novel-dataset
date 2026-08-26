"""WER / CER scoring (jiwer) with identical normalization on ref and hyp,
plus helpers that compute overall and per-region metrics for a hypothesis
column, optionally excluding empty-audio (dead) utterances.
"""
import math

import pandas as pd

from .textnorm import normalize_for_scoring


def _coerce(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def score(refs, hyps, normalize=True, remove_punctuation=True) -> dict:
    """Corpus-level WER and CER over aligned lists of references/hypotheses."""
    import jiwer

    refs2, hyps2 = [], []
    n_skipped, n_empty_hyp = 0, 0
    for r, h in zip(refs, hyps):
        r = normalize_for_scoring(_coerce(r), remove_punctuation) if normalize else _coerce(r)
        h = normalize_for_scoring(_coerce(h), remove_punctuation) if normalize else _coerce(h)
        if not r.strip():
            n_skipped += 1
            continue
        if not h.strip():
            n_empty_hyp += 1
            h = " "
        refs2.append(r)
        hyps2.append(h)
    if not refs2:
        return {"wer": float("nan"), "cer": float("nan"), "n": 0,
                "n_skipped": n_skipped, "n_empty_hyp": n_empty_hyp}
    return {"wer": float(jiwer.wer(refs2, hyps2)), "cer": float(jiwer.cer(refs2, hyps2)),
            "n": len(refs2), "n_skipped": n_skipped, "n_empty_hyp": n_empty_hyp}


def _prep(pred, hyp_col, cfg):
    """Drop rows not transcribed; optionally drop empty-audio rows."""
    df = pred[pred[hyp_col].notna()].copy()
    n_transcribed = len(df)
    n_excluded = 0
    if getattr(cfg, "exclude_empty_audio", True) and "is_empty_audio" in df.columns:
        mask = df["is_empty_audio"].fillna(False).astype(bool)
        n_excluded = int(mask.sum())
        df = df[~mask]
    return df, n_transcribed, n_excluded


def overall_metrics(pred, hyp_col, cfg) -> dict:
    df, n_transcribed, n_excluded = _prep(pred, hyp_col, cfg)
    res = score(df["reference"].tolist(), df[hyp_col].tolist(),
                cfg.normalize, cfg.remove_punctuation)
    res["n_transcribed"] = n_transcribed
    res["n_excluded_empty"] = n_excluded
    return res


def per_region_metrics(pred, hyp_col, cfg) -> pd.DataFrame:
    df, _, _ = _prep(pred, hyp_col, cfg)
    rows = []
    for r in sorted(df["region"].unique()):
        sub = df[df["region"] == r]
        res = score(sub["reference"].tolist(), sub[hyp_col].tolist(),
                    cfg.normalize, cfg.remove_punctuation)
        rows.append({"region": r, "WER": round(res["wer"], 4),
                     "CER": round(res["cer"], 4), "n": res["n"]})
    return pd.DataFrame(rows)
