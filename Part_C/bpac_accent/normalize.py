"""C0 — corpus normalization and data diagnostics.

Produces a CLEAN copy of the corpus:
  * every clip resampled to a single sample rate (16 kHz) and downmixed to mono,
  * empty / too-short / silent (dead) files dropped,
  * optional peak normalization,
so the recording-format confound (44.1k vs 48k, stereo vs mono) is removed before
embeddings are computed.

Also runs targeted diagnostics:
  * per-region empty/dead breakdown (conforming/scored vs non-conforming),
  * a duration-based parallel-ALIGNMENT check: for each shared sentence the clip
    durations should be similar across regions; regions whose clips deviate
    systematically are flagged as alignment-suspect (a proxy for sound_NNN.wav
    not matching sentence id N), with extra focus on Rajshahi/Dhaka.

Reads raw audio with soundfile first (robust to mixed formats), falling back to
librosa only if needed.
"""
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import build_manifest
from .reporting import ensure_dir, maybe_tqdm, save_json, save_table

SOUND_RE = re.compile(r"^sound_(?P<num>\d+)\.wav$", re.IGNORECASE)


def _read_audio(path):
    """Return (y_mono_float32, sr) or (None, None) on failure."""
    try:
        import soundfile as sf
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, sr
    except Exception:
        pass
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return np.asarray(y, dtype=np.float32), sr
    except Exception:
        return None, None


def _resample(y, sr, target_sr):
    if sr == target_sr or y is None or y.size == 0:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr).astype(np.float32)


def normalize_corpus(cfg, manifest_override=None):
    """manifest_override: if given (T5, verified_manifest.csv with excluded=False
    rows already filtered out and sentence_id already corrected by T4), used
    in place of a fresh raw directory walk -- this is what makes data_clean/
    agree with the manifest exactly."""
    nz = cfg["normalize"]
    target_sr = int(nz.get("target_sr", 16000))
    peak_norm = bool(nz.get("peak_normalize", True))
    peak_lin = float(10.0 ** (nz.get("peak_dbfs", -1.0) / 20.0))
    min_ms = float(nz.get("min_ms", 200))
    eps = float(nz.get("silence_peak_eps", 1e-4))
    overwrite = bool(nz.get("overwrite", False))

    raw_root = Path(cfg.raw_data_root)
    clean_root = Path(cfg.data_root)
    raw_bpac = raw_root / cfg.bpac_subdir
    clean_bpac = clean_root / cfg.bpac_subdir
    if not raw_bpac.is_dir():
        raise SystemExit(f"raw corpus not found: {raw_bpac}")
    ensure_dir(clean_bpac)

    # copy the sentence CSV into the clean tree
    raw_csv = raw_root / cfg["corpus"]["sentences_filename"]
    if raw_csv.exists():
        ensure_dir(clean_root)
        shutil.copy2(raw_csv, clean_root / cfg["corpus"]["sentences_filename"])

    if manifest_override is not None:
        mani = manifest_override
    else:
        # enumerate raw files (manifest gives region/speaker/sentence + parallel link)
        mani = build_manifest(raw_root, cfg.bpac_subdir, raw_csv if raw_csv.exists() else None,
                              verbose=cfg.verbose)
    min_samples = int((min_ms / 1000.0) * target_sr)

    rows = []
    for r in maybe_tqdm(list(mani.itertuples(index=False)), desc="normalize"):
        src = Path(r.filepath)
        rel = src.relative_to(raw_bpac)
        dst = clean_bpac / rel
        y, sr = _read_audio(src)
        if y is None:
            rows.append({"filepath": str(src), "region": r.region, "sentence_id": r.sentence_id,
                         "speaker_uid": r.speaker_uid, "status": "load_failed",
                         "in_dur_s": np.nan, "out_dur_s": np.nan})
            continue
        in_dur = y.size / float(sr or 1)
        y = _resample(y, sr, target_sr)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        is_dead = (y.size == 0) or (y.size < min_samples) or (peak <= eps)
        if is_dead:
            rows.append({"filepath": str(src), "region": r.region, "sentence_id": r.sentence_id,
                         "speaker_uid": r.speaker_uid, "status": "dead_skipped",
                         "in_dur_s": round(in_dur, 3), "out_dur_s": round(y.size / target_sr, 3)})
            continue
        if peak_norm and peak > 0:
            y = (y * (peak_lin / peak)).astype(np.float32)
        ensure_dir(dst.parent)
        if dst.exists() and not overwrite:
            status = "exists_skipped"
        else:
            import soundfile as sf
            sf.write(str(dst), y, target_sr, subtype="PCM_16")
            status = "written"
        rows.append({"filepath": str(src), "region": r.region, "sentence_id": r.sentence_id,
                     "speaker_uid": r.speaker_uid, "status": status,
                     "in_dur_s": round(in_dur, 3), "out_dur_s": round(y.size / target_sr, 3)})

    df = pd.DataFrame(rows)
    return df, mani


def compute_checksums(clean_bpac, progress=None):
    """sha256 of every clip in the clean tree -- the reproducibility anchor
    (T5): anyone who re-normalizes from the same raw+verified_manifest.csv
    should get byte-identical files."""
    import hashlib

    files = sorted(Path(clean_bpac).rglob("*.wav"))
    if progress is not None:
        files = progress(files, desc="checksums")
    rows = []
    for f in files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        rows.append({"filepath": str(f), "sha256": h, "size_bytes": f.stat().st_size})
    return pd.DataFrame(rows)


def per_region_summary(df):
    def agg(g):
        return pd.Series({
            "n_input": len(g),
            "n_written": int((g["status"] == "written").sum()),
            "n_exists_skipped": int((g["status"] == "exists_skipped").sum()),
            "n_dead_skipped": int((g["status"] == "dead_skipped").sum()),
            "n_load_failed": int((g["status"] == "load_failed").sum()),
            "median_in_dur_s": g["in_dur_s"].median(),
        })
    return df.groupby("region").apply(agg).reset_index().sort_values("n_dead_skipped",
                                                                     ascending=False)


def alignment_check(df, tol=0.40, focus=None):
    """Per-(region,sentence) mean duration vs the cross-region median for that
    sentence. Region/sentence cells deviating by > tol (fraction) are flagged as
    alignment-suspect (sound_NNN may not match sentence N)."""
    live = df[df["status"].isin(["written", "exists_skipped"])].copy()
    if live.empty:
        return pd.DataFrame(), pd.DataFrame()
    # mean duration per (region, sentence)
    cell = (live.groupby(["sentence_id", "region"])["in_dur_s"].mean()
            .reset_index().rename(columns={"in_dur_s": "region_dur"}))
    med = (live.groupby("sentence_id")["in_dur_s"].median()
           .reset_index().rename(columns={"in_dur_s": "sentence_median"}))
    cell = cell.merge(med, on="sentence_id")
    cell["rel_dev"] = (cell["region_dur"] - cell["sentence_median"]).abs() / (cell["sentence_median"] + 1e-6)
    cell["suspect"] = cell["rel_dev"] > tol

    by_region = (cell.groupby("region")
                 .agg(n_sentences=("sentence_id", "nunique"),
                      n_suspect=("suspect", "sum"),
                      mean_rel_dev=("rel_dev", "mean"))
                 .reset_index())
    by_region["pct_suspect"] = (100.0 * by_region["n_suspect"] / by_region["n_sentences"]).round(1)
    by_region = by_region.sort_values("pct_suspect", ascending=False)
    if focus:
        by_region["focus"] = by_region["region"].isin([f.lower() for f in focus])
    return by_region, cell
