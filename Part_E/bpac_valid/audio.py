"""Minimal audio helpers shared by D0-D7. Task-specific DSP (formants, VOT,
spectral moments, F0 contours) lands in each D-task's own module as T13-T17
are implemented.
"""


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
