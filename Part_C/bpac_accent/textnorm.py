"""Text normalization for STT scoring.

Identical normalization is applied to BOTH reference and hypothesis before
computing WER/CER: Unicode NFC, optional punctuation/danda removal, whitespace
collapse. This avoids inflating error rates due to encoding/punctuation
mismatches (critical for Bangla).
"""
import re
import unicodedata

# Bangla danda/double-danda + common ASCII/Unicode punctuation.
_PUNCT_RE = re.compile(
    r"[।॥…!?,.;:\"'`“”‘’()\[\]{}<>/\\|@#%^&*_+=~\-—–·•]+"
)
_WS_RE = re.compile(r"\s+")


def nfc(s) -> str:
    return unicodedata.normalize("NFC", str(s))


def normalize_for_scoring(s, remove_punctuation=True) -> str:
    """NFC -> (optional) strip punctuation/danda -> collapse whitespace -> strip."""
    if s is None:
        return ""
    s = nfc(s)
    if remove_punctuation:
        s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s
