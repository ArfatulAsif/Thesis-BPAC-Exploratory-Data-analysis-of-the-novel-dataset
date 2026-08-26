#!/usr/bin/env python3
"""T16 -- Consonants: sibilant spectral moments + VOT.

This is the analysis that quantifies the observation motivating the whole
project: sibilant spectral centre-of-gravity (CoG) numerically separates
/s/-like (high CoG) from /sh/-like (lower CoG) realizations, converting the
Sylhet anecdote into a measurement with error bars.

SAMPLE-RATE EXCEPTION (invariant 2): sibilant tokens are read from RAW audio
(44.1/48 kHz, via verified_manifest.csv's filepath, which points at data/ not
data_clean/) resampled to 32 kHz -- at 16 kHz the Nyquist limit is 8 kHz,
which clips the upper /s/ energy and compresses exactly the distinction
being measured. This is the ONE place in Part D that does not read
data_clean/; see D0-D3 for the normal rule.

VOT detection (word-initial four-way stops p/ph/b/bh, t/th/d/dh, T/Th/D/Dh,
k/kh/g/gh) is semi-automatic: automatic detection runs over every token,
but should be treated as provisional until the ~40/region hand-verification
sample (vot_manual_check_worklist.csv) is actually reviewed by a person --
same human-in-the-loop policy as T4/T11.

Outputs (in ./reports): sibilant_tokens.parquet, sibilant_cog_by_region.csv,
sibilant_mixedmodel.csv, vot_tokens.csv, vot_by_region_category.csv,
vot_manual_check_worklist.csv, figures/*, report.md, summary.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.provenance import base_summary  # noqa: E402

from bpac_phon.cli import add_config_arg  # noqa: E402
from bpac_phon.reporting import ReportWriter, ensure_dir, maybe_tqdm, save_fig, save_json, save_table  # noqa: E402
from bpac_phon.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

STOP_SERIES = {
    "p": "bilabial", "ph": "bilabial", "b": "bilabial", "bh": "bilabial",
    "t": "dental", "th": "dental", "d": "dental", "dh": "dental",
    "T": "retroflex", "Th": "retroflex", "D": "retroflex", "Dh": "retroflex",
    "k": "velar", "kh": "velar", "g": "velar", "gh": "velar",
}
VOICED_STOPS = {"b", "bh", "d", "dh", "D", "Dh", "g", "gh"}


def load_raw_audio_32k(path, t0, t1, pad_s=0.05):
    import soundfile as sf
    import librosa
    info = sf.info(str(path))
    start_frame = max(0, int((t0 - pad_s) * info.samplerate))
    stop_frame = min(info.frames, int((t1 + pad_s) * info.samplerate))
    y, sr = sf.read(str(path), start=start_frame, stop=stop_frame, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 32000:
        y = librosa.resample(y, orig_sr=sr, target_sr=32000)
        sr = 32000
    seg_t0 = max(0, t0 - pad_s)
    return y, sr, seg_t0


def spectral_moments(y, sr, band=(500, 8000)):
    """First four spectral moments (CoG, SD, skewness, kurtosis) of the
    middle 50% of the signal, Hamming-windowed + pre-emphasized, over `band`."""
    n = len(y)
    if n < 32:
        return {k: np.nan for k in ("cog", "sd", "skew", "kurt")}
    lo, hi = int(n * 0.25), int(n * 0.75)
    seg = y[lo:hi]
    if len(seg) < 16:
        seg = y
    pre = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])
    win = pre * np.hamming(len(pre))
    spec = np.abs(np.fft.rfft(win)) ** 2
    freqs = np.fft.rfftfreq(len(win), d=1.0 / sr)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    f, p = freqs[band_mask], spec[band_mask]
    p_sum = p.sum()
    if p_sum <= 0:
        return {k: np.nan for k in ("cog", "sd", "skew", "kurt")}
    p_norm = p / p_sum
    cog = float(np.sum(f * p_norm))
    var = float(np.sum(((f - cog) ** 2) * p_norm))
    sd = float(np.sqrt(var))
    skew = float(np.sum(((f - cog) ** 3) * p_norm) / (sd ** 3 + 1e-9))
    kurt = float(np.sum(((f - cog) ** 4) * p_norm) / (sd ** 4 + 1e-9) - 3)
    return {"cog": cog, "sd": sd, "skew": skew, "kurt": kurt}


def detect_vot(y, sr, burst_search_end_frac=0.6):
    """Semi-automatic VOT: burst = argmax of the high-frequency (>2kHz)
    energy derivative in the first burst_search_end_frac of the segment;
    voicing onset = first frame after the burst where autocorrelation-based
    periodicity exceeds a threshold (proxy for F0 onset). Returns VOT in ms
    (negative = prevoicing) or NaN if detection fails."""
    import librosa
    n = len(y)
    if n < 64:
        return np.nan, np.nan, np.nan
    hop = max(1, int(0.002 * sr))  # 2ms hop
    win = max(1, int(0.010 * sr))  # 10ms window
    S = np.abs(librosa.stft(y, n_fft=max(256, win), hop_length=hop, win_length=win))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=max(256, win))
    hf_mask = freqs > 2000
    hf_energy = S[hf_mask].sum(axis=0)
    search_end = int(len(hf_energy) * burst_search_end_frac)
    if search_end < 2:
        return np.nan, np.nan, np.nan
    deriv = np.diff(hf_energy[:search_end])
    if len(deriv) == 0:
        return np.nan, np.nan, np.nan
    burst_frame = int(np.argmax(deriv)) + 1
    burst_t = burst_frame * hop / sr

    # voicing onset: first frame after burst with strong autocorrelation periodicity
    f0_frame_len = int(0.03 * sr)
    onset_t = np.nan
    for start in range(burst_frame * hop, n - f0_frame_len, hop):
        frame = y[start:start + f0_frame_len]
        frame = frame - frame.mean()
        if np.allclose(frame, 0):
            continue
        ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        ac = ac / (ac[0] + 1e-9)
        # look for a strong peak in the plausible pitch-period range (60-500 Hz)
        lo_lag, hi_lag = int(sr / 500), int(sr / 60)
        if hi_lag >= len(ac):
            continue
        peak = ac[lo_lag:hi_lag].max() if hi_lag > lo_lag else 0
        if peak > 0.35:
            onset_t = start / sr
            break
    if onset_t != onset_t:  # NaN
        return np.nan, burst_t, np.nan
    vot_ms = (onset_t - burst_t) * 1000
    return vot_ms, burst_t, onset_t


def main():
    ap = argparse.ArgumentParser(description="T16 consonants: sibilants + VOT")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")
    d4 = cfg["d4_consonants"]

    phone_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "D1_alignment", "reports", "phone_tokens.parquet")
    if not os.path.exists(phone_path):
        raise SystemExit(f"{phone_path} not found -- run D1_alignment (T12) first.")
    phones = pd.read_parquet(phone_path)

    if not os.path.exists(cfg.verified_manifest):
        raise SystemExit(f"{cfg.verified_manifest} not found -- run T4 first (need raw filepaths).")
    vm = pd.read_csv(cfg.verified_manifest, encoding="utf-8-sig")
    raw_filepath = dict(zip(vm["utterance_id"], vm["filepath"]))  # verified_manifest points at raw data/

    # =====================================================================
    # Sibilants (single canonical phone "s" -- see phoneset.md; regional
    # realization is discovered acoustically here, not baked into the lexicon)
    # =====================================================================
    sib = phones[phones["phone"] == "s"].copy()
    min_ms = float(d4.get("sibilant_min_ms", 50))
    sib = sib[sib["duration_s"] * 1000 >= min_ms]
    n_sib_candidate = len(sib)

    rows = []
    for r in maybe_tqdm(sib.itertuples(index=False), desc="T16 sibilants", total=len(sib)):
        path = raw_filepath.get(r.utterance_id)
        if path is None or not os.path.exists(path):
            continue
        try:
            y, sr, _ = load_raw_audio_32k(path, r.start_s, r.end_s, pad_s=0.0)
            mom = spectral_moments(y, sr, band=tuple(d4.get("sibilant_band_hz", [500, 8000])))
        except Exception:
            continue
        rows.append({"utterance_id": r.utterance_id, "region": r.region, "gender": r.gender,
                    "speaker_uid": r.speaker_uid, "phone": r.phone, "duration_s": r.duration_s,
                    "start_s": r.start_s, "end_s": r.end_s,
                    **mom})
    sib_df = pd.DataFrame(rows)
    if len(sib_df):
        sib_df["cog_z"] = sib_df.groupby("speaker_uid")["cog"].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-9))
    sib_df.to_parquet(out / "sibilant_tokens.parquet", index=False)

    cog_by_region = (
        sib_df.groupby("region").agg(
            n=("cog", "size"), cog_mean_hz=("cog", "mean"), cog_median_hz=("cog", "median"),
            cog_z_mean=("cog_z", "mean"), cog_z_sem=("cog_z", lambda s: s.std() / np.sqrt(len(s))),
        ).reset_index()
    ) if len(sib_df) else pd.DataFrame()
    save_table(cog_by_region, out / "sibilant_cog_by_region.csv", index=False)

    mm_df = pd.DataFrame()
    if len(sib_df) and sib_df["region"].nunique() >= 2:
        import statsmodels.formula.api as smf
        try:
            md = smf.mixedlm("cog_z ~ C(region)", sib_df.dropna(subset=["cog_z"]), groups=sib_df.dropna(subset=["cog_z"])["speaker_uid"])
            res = md.fit(reml=True, method="lbfgs")
            mm_df = pd.DataFrame({"term": res.params.index, "coef": res.params.values, "p_value": res.pvalues.values})
            mm_df = mm_df[mm_df["term"].str.startswith("C(region)")]
        except Exception as e:
            print(f"[T16] sibilant mixed model failed: {e}")
    save_table(mm_df, out / "sibilant_mixedmodel.csv", index=False)

    sylhet_shift = None
    if len(cog_by_region) and "sylhet" in cog_by_region["region"].values:
        overall_mean = sib_df["cog"].mean()
        overall_sd = sib_df["cog"].std()
        syl_mean = cog_by_region.loc[cog_by_region["region"] == "sylhet", "cog_mean_hz"].iloc[0]
        sylhet_shift = {"shift_hz": float(syl_mean - overall_mean),
                        "shift_sd_units": float((syl_mean - overall_mean) / overall_sd) if overall_sd else None}

    if len(sib_df):
        fig, ax = plt.subplots(figsize=(8, 5))
        for region, sub in sib_df.groupby("region"):
            sub["cog"].plot(kind="density", ax=ax, label=region)
        ax.set_xlabel("sibilant spectral CoG (Hz)")
        ax.set_title("T16 -- sibilant CoG density by region (32 kHz)")
        ax.legend(fontsize=7, ncol=2)
        save_fig(fig, fig_dir / "cog_density_by_region.png")
        fig.savefig(fig_dir / "cog_density_by_region.pdf", bbox_inches="tight")
        plt.close(fig)

    # =====================================================================
    # VOT (word-initial four-way stops)
    # =====================================================================
    stops = phones[phones["phone"].isin(STOP_SERIES) & (phones["position_in_word"] == 0)].copy()
    stops["place"] = stops["phone"].map(STOP_SERIES)
    stops["voicing"] = np.where(stops["phone"].isin(VOICED_STOPS), "voiced", "voiceless")
    stops["aspiration"] = np.where(stops["phone"].str.endswith("h"), "aspirated", "unaspirated")

    vot_rows = []
    for r in maybe_tqdm(stops.itertuples(index=False), desc="T16 VOT", total=len(stops)):
        path = raw_filepath.get(r.utterance_id)
        if path is None or not os.path.exists(path):
            continue
        try:
            y, sr, seg_t0 = load_raw_audio_32k(path, r.start_s, r.end_s, pad_s=0.08)
            vot_ms, burst_t, onset_t = detect_vot(y, sr)
        except Exception:
            continue
        vot_rows.append({
            "utterance_id": r.utterance_id, "region": r.region, "gender": r.gender,
            "speaker_uid": r.speaker_uid, "phone": r.phone, "place": r.place,
            "voicing": r.voicing, "aspiration": r.aspiration, "vot_ms": vot_ms,
        })
    vot_df = pd.DataFrame(vot_rows)
    save_table(vot_df, out / "vot_tokens.csv", index=False)

    vot_by_cat = (
        vot_df.dropna(subset=["vot_ms"]).groupby(["region", "phone"]).agg(
            n=("vot_ms", "size"), mean_vot_ms=("vot_ms", "mean"), sd_vot_ms=("vot_ms", "std"),
        ).reset_index()
    ) if len(vot_df) else pd.DataFrame()
    save_table(vot_by_cat, out / "vot_by_region_category.csv", index=False)

    # manual-verification worklist: ~40/region stratified sample, per spec
    check_n = int(d4.get("vot_manual_check_n", 300))
    per_region_n = max(1, check_n // max(1, vot_df["region"].nunique())) if len(vot_df) else 0
    worklist = (
        vot_df.dropna(subset=["vot_ms"])
        .groupby("region", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_region_n), random_state=42))
    ) if len(vot_df) else pd.DataFrame()
    if len(worklist):
        worklist = worklist.copy()
        worklist["manual_vot_ms"] = ""
        worklist["reviewed"] = False
        worklist["reviewer"] = ""
    save_table(worklist, out / "vot_manual_check_worklist.csv", index=False)

    summary = base_summary("T16_consonants", cfg.config_path,
                           n_sib_candidate + len(stops), len(sib_df) + len(vot_df.dropna(subset=["vot_ms"])) if len(vot_df) else len(sib_df),
                           count_unit="sibilant_plus_stop_tokens")
    summary.update({
        "n_sibilant_tokens": len(sib_df), "n_sibilant_candidates": n_sib_candidate,
        "sylhet_cog_shift": sylhet_shift,
        "n_vot_tokens_detected": int(vot_df["vot_ms"].notna().sum()) if len(vot_df) else 0,
        "n_vot_candidates": len(stops),
        "vot_manual_check_status": "NOT STARTED -- vot_manual_check_worklist.csv needs human review "
                                   f"({len(worklist)} rows). Automatic VOT is provisional until then.",
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T16 -- Consonants: Sibilant Spectral Moments + VOT")
    rep.p(
        "**Sample-rate exception**: sibilant and VOT tokens are read from RAW audio resampled to "
        "32 kHz, not data_clean/ (16 kHz) -- see module docstring / invariant 2."
    )
    rep.h2("Sibilant CoG by region")
    rep.table(cog_by_region.round(2), index=False)
    if sylhet_shift:
        rep.bullet(f"Sylhet CoG shift vs. corpus mean: **{sylhet_shift['shift_hz']:.1f} Hz** "
                  f"({sylhet_shift['shift_sd_units']:.3f} SD units).")
    rep.image("figures/cog_density_by_region.png", "Sibilant CoG density by region -- likely the most-quoted number in the paper.")
    rep.h2("VOT (word-initial four-way stops) -- PROVISIONAL, automatic only")
    rep.p(
        f"{summary['n_vot_tokens_detected']}/{summary['n_vot_candidates']} candidates got an automatic "
        f"VOT estimate. **Manual verification not yet done** -- `vot_manual_check_worklist.csv` "
        f"({len(worklist)} rows, ~{per_region_n}/region) needs a human to listen and correct "
        "`manual_vot_ms`. Report automatic-vs-manual agreement once reviewed; if poor, report VOT on "
        "the hand-checked subsample only."
    )
    rep.table(vot_by_cat.round(2).head(40), index=False)
    rep.h2("For the paper")
    rep.p("S7.5. The sibilant CoG result is likely the paper's most-quoted number.")
    rep.write()

    print(f"[T16] done. sibilants={len(sib_df)}/{n_sib_candidate}  "
          f"vot_detected={summary['n_vot_tokens_detected']}/{summary['n_vot_candidates']}  "
          f"sylhet_shift={sylhet_shift}")


if __name__ == "__main__":
    main()
