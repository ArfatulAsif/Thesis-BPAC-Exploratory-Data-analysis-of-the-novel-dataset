"""Build a flat manifest DataFrame from the BPAC directory tree.

One row per utterance with the canonical global speaker id, the linked
sentence text, and the recording pipeline (lab/app, derived from gender).

Prefers `verified_manifest.csv` (Phase 1, T4) once it exists -- that is the
canonical, alignment-verified manifest every Part D task must read. Falls
back to a plain directory walk via `common.manifest_core` so this package is
usable before Phase 1 lands.

Thin wrapper around `common.manifest_core` (BPAC Paper 1, T2) -- see that
module for the shared implementation used by all Parts.
"""
import sys
from pathlib import Path

import pandas as pd

SENTENCES_FILENAME = "best_diverse_sentences.csv"

# bpac_phon/manifest.py -> Part_D/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.manifest_core import (  # noqa: E402
    GENDER_TO_PIPELINE,
    PIPELINE_GENDER_CONFOUND,
    SOUND_RE,
    SPEAKER_RE,
    build_manifest_core,
    parse_speaker,
)
from common.manifest_core import load_sentences as _load_sentences_core  # noqa: E402

from .textnorm import nfc  # noqa: E402

__all__ = [
    "SENTENCES_FILENAME", "SPEAKER_RE", "SOUND_RE", "PIPELINE_GENDER_CONFOUND",
    "GENDER_TO_PIPELINE", "parse_speaker", "load_sentences", "build_manifest",
    "load_verified_manifest",
]


def load_sentences(sentences_csv) -> dict:
    """Load id -> NFC-normalized sentence from the transcript CSV."""
    return _load_sentences_core(sentences_csv, nfc)


def load_verified_manifest(verified_manifest_path, verbose=True) -> pd.DataFrame:
    """Load the Phase 1 (T4) canonical manifest, excluding rows marked excluded=True."""
    path = Path(verified_manifest_path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "excluded" in df.columns:
        n_before = len(df)
        df = df[~df["excluded"].astype(bool)].reset_index(drop=True)
        if verbose:
            print(f"[manifest] verified_manifest: {n_before} rows -> {len(df)} after excluding flagged rows")
    return df


def build_manifest(
    data_root="../data_clean",
    bpac_subdir="BPAC",
    sentences_csv=None,
    verified_manifest_path=None,
    probe_durations=False,
    probe_format=False,
    verbose=True,
):
    """Return the per-utterance DataFrame Part D analyses should read.

    If `verified_manifest_path` points at an existing file, that file is the
    source of truth (Phase 1, T4). Otherwise falls back to a directory walk
    of `data_root` (pre-Phase-1 usable state).

    Columns: utterance_id, filepath, region, region_raw, gender, pipeline,
    speaker_id, speaker_name, speaker_uid, sentence_id, sentence
    (+ duration_s if probe_durations, + recording_sr_raw/channels_raw if probe_format).
    """
    if verified_manifest_path is not None and Path(verified_manifest_path).exists():
        return load_verified_manifest(verified_manifest_path, verbose=verbose)

    if verbose:
        print("[manifest] verified_manifest.csv not found -- falling back to a plain "
              "directory walk (pre-Phase-1 state; re-run once T4 lands).")

    data_root = Path(data_root)
    if sentences_csv is None:
        sentences_csv = data_root / SENTENCES_FILENAME

    duration_fn, tqdm_fn = None, None
    if probe_durations or probe_format:
        from .reporting import maybe_tqdm

        tqdm_fn = maybe_tqdm
    if probe_durations:
        from .audio import duration_seconds

        duration_fn = duration_seconds

    return build_manifest_core(
        data_root,
        bpac_subdir,
        sentences_csv,
        nfc,
        probe_durations=probe_durations,
        probe_format=probe_format,
        duration_fn=duration_fn,
        tqdm_fn=tqdm_fn,
        verbose=verbose,
    )
