"""T7 -- WADA-style SNR estimator + noise floor.

`check_audio_quality.py`'s existing `snr_db` is an AD-HOC, UNNAMED estimator:
90th-percentile frame RMS (dB) minus 10th-percentile frame RMS (dB) over
25 ms frames. It is not a published method and is not directly comparable
across corpora -- see T7 in the BPAC Paper 1 spec.

This implements the core statistic behind WADA-SNR (Kim & Stern, "Robust
Signal-to-Noise Ratio Estimation Based on Waveform Amplitude Distribution
Analysis", Interspeech 2008): the log-amplitude distribution of a
noisy-speech signal is modeled as Gamma-distributed with a shape parameter
alpha that shrinks as SNR rises (speech's amplitude distribution is more
peaked/heavy-tailed than noise's). alpha is a closed-form function of the
signal via the digamma equation below -- no memorized constants.

IMPORTANT DEVIATION from the original paper: Kim & Stern's alpha -> SNR(dB)
mapping is an empirically-fit lookup table calibrated on their own
speech+noise corpus, which is not republished in a form that can be
reproduced from first principles here. Rather than risk silently
transcribing that table incorrectly, this module CALIBRATES THE SAME
alpha -> SNR(dB) mapping LOCALLY: synthetic mixtures are built from a sample
of this corpus's own clips + additive white Gaussian noise at known,
controlled SNR levels, and a monotonic (isotonic) regression is fit from
alpha to the known SNR. This is the same underlying statistic and method
family, transparently and reproducibly calibrated in-repo -- report this as
"WADA-style SNR, locally calibrated", not literally Kim & Stern's numbers.
"""
import numpy as np
from scipy.special import digamma
from scipy.optimize import brentq

EPS = 1e-10


def gamma_shape_alpha(x: np.ndarray) -> float:
    """alpha solving digamma(alpha) - ln(alpha) = -G, where
    G = ln(E|x|) - E[ln|x|] (the WADA statistic)."""
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x))
    if peak <= 0:
        return np.nan
    x = x / peak
    ax = np.abs(x)
    ax = np.maximum(ax, EPS)
    G = np.log(np.mean(ax)) - np.mean(np.log(ax))
    if not np.isfinite(G) or G <= 0:
        return np.nan

    def f(a):
        return np.log(a) - digamma(a) - G

    lo, hi = 1e-3, 1e4
    try:
        if f(lo) * f(hi) > 0:
            return np.nan
        return float(brentq(f, lo, hi, maxiter=200))
    except Exception:
        return np.nan


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` (tiled/cropped to len(clean)) so the mixture has the given SNR."""
    n = len(clean)
    reps = int(np.ceil(n / len(noise)))
    noise = np.tile(noise, reps)[:n]
    p_clean = np.mean(clean.astype(np.float64) ** 2) + EPS
    p_noise = np.mean(noise.astype(np.float64) ** 2) + EPS
    target_p_noise = p_clean / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_p_noise / p_noise)
    return (clean + scale * noise).astype(np.float32)


def calibrate(clean_clips, snr_grid=(-5, 0, 5, 10, 15, 20, 25, 30, 35, 40), seed=42):
    """Fit alpha -> SNR(dB) from synthetic mixtures of this corpus's own
    clips + white Gaussian noise at known SNR. Returns a fitted
    sklearn.isotonic.IsotonicRegression plus the (alpha, snr) pairs used,
    for a report figure showing the fit quality."""
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(seed)
    alphas, targets = [], []
    for clip in clean_clips:
        clip = np.asarray(clip, dtype=np.float32)
        if clip.size < 100:
            continue
        noise = rng.normal(0, 1, size=clip.size).astype(np.float32)
        for snr_db in snr_grid:
            mix = mix_at_snr(clip, noise, snr_db)
            a = gamma_shape_alpha(mix)
            if np.isfinite(a):
                alphas.append(a)
                targets.append(snr_db)
    alphas, targets = np.array(alphas), np.array(targets)
    order = np.argsort(-alphas)  # alpha decreases as SNR increases -> sort descending alpha
    iso = IsotonicRegression(out_of_bounds="clip", increasing=False)
    iso.fit(alphas, targets)
    fit_r2 = np.corrcoef(iso.predict(alphas), targets)[0, 1] ** 2
    return iso, {"alphas": alphas.tolist(), "targets": targets.tolist(), "fit_r2": float(fit_r2)}


def wada_snr(x: np.ndarray, calibration) -> float:
    a = gamma_shape_alpha(x)
    if not np.isfinite(a):
        return np.nan
    return float(calibration.predict([a])[0])


def noise_floor_dbfs(x: np.ndarray, sr: int, frame_ms=25, floor_pct=5) -> float:
    """Median energy (dBFS) of the lowest floor_pct% of 25ms frames."""
    x = np.asarray(x, dtype=np.float64)
    win = max(1, int(frame_ms / 1000.0 * sr))
    n = x.size
    if n < win:
        return float("nan")
    nf = n // win
    frames = x[: nf * win].reshape(nf, win)
    fr_rms = np.sqrt(np.mean(frames**2, axis=1) + EPS)
    fr_db = 20 * np.log10(fr_rms + EPS)
    cutoff = np.percentile(fr_db, floor_pct)
    low = fr_db[fr_db <= cutoff]
    return float(np.median(low)) if low.size else float(np.median(fr_db))
