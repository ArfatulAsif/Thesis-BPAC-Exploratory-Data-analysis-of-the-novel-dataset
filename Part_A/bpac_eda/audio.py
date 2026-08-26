"""Audio helpers: duration, speech-only duration, and F0 estimation.

Duration uses the file header only (fast). F0 prefers Praat via parselmouth and
falls back to librosa.pyin if parselmouth is unavailable.
"""
import math

import numpy as np

from .config import ANALYSIS_SR, DEFAULT_TOP_DB, PITCH_RANGES


def duration_seconds(path) -> float:
    """Raw file duration in seconds (reads header only)."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / float(info.samplerate)
    except Exception:
        try:
            import librosa

            return float(librosa.get_duration(path=str(path)))
        except Exception:
            return float("nan")


def audio_specs(path) -> dict:
    """Sample rate / channels / subtype for an integrity probe."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "format": info.format,
        }
    except Exception:
        return {"sample_rate": None, "channels": None, "subtype": None, "format": None}


def load_mono(path, sr=ANALYSIS_SR):
    import librosa

    y, sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr


def speech_duration_seconds(path, sr=ANALYSIS_SR, top_db=DEFAULT_TOP_DB) -> float:
    """Duration after trimming leading/trailing silence (librosa.effects.trim)."""
    try:
        import librosa

        y, sr = load_mono(path, sr)
        if y.size == 0:
            return 0.0
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        return len(yt) / float(sr)
    except Exception:
        return float("nan")


_VAD_MODEL = None


def load_vad_model():
    """Load (and cache) the Silero VAD model. T6: replaces librosa.effects.trim
    energy thresholding, which explodes WPM on quiet recordings (near-zero
    speech duration -> huge words/min)."""
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from silero_vad import load_silero_vad

        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def speech_duration_vad(path, model=None, sr=ANALYSIS_SR, threshold=0.5) -> dict:
    """VAD-based speech duration. Returns speech_dur_s, total_dur_s, speech_ratio."""
    from silero_vad import get_speech_timestamps

    model = model or load_vad_model()
    y, file_sr = load_mono(path, sr=sr)
    if y.size == 0:
        return {"speech_dur_s": 0.0, "total_dur_s": 0.0, "speech_ratio": 0.0}
    import torch

    total_dur_s = y.size / float(sr)
    ts = get_speech_timestamps(torch.from_numpy(y), model, threshold=threshold,
                               sampling_rate=sr, return_seconds=True)
    speech_dur_s = float(sum(seg["end"] - seg["start"] for seg in ts))
    return {
        "speech_dur_s": speech_dur_s,
        "total_dur_s": total_dur_s,
        "speech_ratio": speech_dur_s / total_dur_s if total_dur_s > 0 else 0.0,
    }


def mean_f0(path, gender="male", sr=ANALYSIS_SR, top_db=DEFAULT_TOP_DB, pitch_ranges=None) -> dict:
    """Return F0 stats over voiced frames: mean/sd/min/max/voiced_fraction (Hz)."""
    ranges = pitch_ranges or PITCH_RANGES
    floor, ceiling = ranges.get(gender, (75, 500))
    nan = float("nan")
    # --- Preferred: Praat (parselmouth) ---
    try:
        import parselmouth

        snd = parselmouth.Sound(str(path))
        try:
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
        except Exception:
            pitch = snd.to_pitch_ac(pitch_floor=floor, pitch_ceiling=ceiling)
        freqs = pitch.selected_array["frequency"]
        voiced = freqs[freqs > 0]
        total = len(freqs)
        if voiced.size == 0:
            return {"f0_mean": nan, "f0_sd": nan, "f0_min": nan, "f0_max": nan,
                    "voiced_fraction": 0.0, "f0_method": "parselmouth"}
        return {
            "f0_mean": float(np.mean(voiced)),
            "f0_sd": float(np.std(voiced)),
            "f0_min": float(np.min(voiced)),
            "f0_max": float(np.max(voiced)),
            "voiced_fraction": float(voiced.size / total) if total else nan,
            "f0_method": "parselmouth",
        }
    except Exception:
        pass
    # --- Fallback: librosa.pyin ---
    try:
        import librosa

        y, sr = load_mono(path, sr)
        if y.size == 0:
            return {"f0_mean": nan, "f0_sd": nan, "f0_min": nan, "f0_max": nan,
                    "voiced_fraction": 0.0, "f0_method": "pyin"}
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        f0, _, _ = librosa.pyin(yt, fmin=floor, fmax=ceiling, sr=sr)
        voiced = f0[~np.isnan(f0)]
        total = np.count_nonzero(~np.isnan(f0)) + np.count_nonzero(np.isnan(f0))
        if voiced.size == 0:
            return {"f0_mean": nan, "f0_sd": nan, "f0_min": nan, "f0_max": nan,
                    "voiced_fraction": 0.0, "f0_method": "pyin"}
        return {
            "f0_mean": float(np.mean(voiced)),
            "f0_sd": float(np.std(voiced)),
            "f0_min": float(np.min(voiced)),
            "f0_max": float(np.max(voiced)),
            "voiced_fraction": float(voiced.size / total) if total else nan,
            "f0_method": "pyin",
        }
    except Exception:
        return {"f0_mean": nan, "f0_sd": nan, "f0_min": nan, "f0_max": nan,
                "voiced_fraction": nan, "f0_method": "none"}
