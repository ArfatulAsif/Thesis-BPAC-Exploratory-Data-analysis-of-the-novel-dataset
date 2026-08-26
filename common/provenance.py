"""Provenance helpers so every summary.json is traceable (spec invariant 6).

Every `summary.json` in the repo must include:
    task_id, generated_at, config_hash, n_utterances_in, n_utterances_used
"""
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def config_hash(config_path) -> str:
    """Short sha256 of a config.yml's raw bytes, so a summary.json can be
    matched back to the exact config that produced it."""
    data = Path(config_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def base_summary(task_id, config_path, n_utterances_in, n_utterances_used, count_unit="utterances"):
    """count_unit documents what n_utterances_in/used actually count when a
    task's natural unit isn't the utterance (e.g. T11 counts word types,
    T13-T17 count phone/vowel/sibilant/VOT tokens, T18-T19 count regions).
    common/consistency_check.py (T8) only checks n_in against the corpus's
    canonical utterance counts when count_unit == 'utterances' -- otherwise
    it just records the numbers for traceability, per invariant 6."""
    return {
        "task_id": task_id,
        "generated_at": now_iso(),
        "config_hash": config_hash(config_path),
        "n_utterances_in": int(n_utterances_in),
        "n_utterances_used": int(n_utterances_used),
        "count_unit": count_unit,
    }
