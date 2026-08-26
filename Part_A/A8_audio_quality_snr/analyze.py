#!/usr/bin/env python3
"""T7 -- Named, documented SNR estimator + noise floor + SNR<->WER correlation.

`check_audio_quality.py`'s `snr_db` is an ad-hoc, unnamed percentile-frame
heuristic (documented below, kept for comparison). This adds a WADA-style
SNR estimator (Kim & Stern 2008 statistic, locally calibrated -- see
bpac_eda/snr.py for exactly what deviates from the original paper) and a
noise-floor measure, computed per file on data_clean/, aggregated by region
and by pipeline (== gender, see invariant 3), and correlated against
Part_B's per-region WER to test whether noise explains the WER differences.

Outputs (in ./reports): per_file_quality.csv, per_region_quality.csv,
region_pipeline_quality.csv, snr_wer_correlation.csv, figures/*, report.md,
summary.json.
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
import soundfile as sf  # noqa: E402
from scipy import stats  # noqa: E402

from common.provenance import base_summary  # noqa: E402

from bpac_eda import snr as snrmod  # noqa: E402
from bpac_eda.cli import add_config_arg  # noqa: E402
from bpac_eda.manifest import build_manifest  # noqa: E402
from bpac_eda.reporting import (ReportWriter, ensure_dir, maybe_tqdm, save_fig,  # noqa: E402
                                save_json, save_table)
from bpac_eda.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

CURRENT_ESTIMATOR_DOC = (
    "check_audio_quality.py's existing `snr_db` = 90th-percentile 25ms-frame RMS(dB) "
    "minus 10th-percentile 25ms-frame RMS(dB) -- an ad-hoc percentile-loudness-ratio "
    "heuristic, not a published/named method."
)


def analyze_file_full(path, calibration):
    info = sf.info(path)
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = y.size
    out = {
        "sample_rate": info.samplerate, "channels": info.channels, "subtype": info.subtype,
        "duration_s": n / float(sr or 1),
    }
    if n == 0:
        out.update(rms_dbfs=np.nan, peak_dbfs=np.nan, clip_pct=np.nan,
                   wada_snr_db=np.nan, noise_floor_dbfs=np.nan, old_pctile_snr_db=np.nan)
        return out
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    out["rms_dbfs"] = 20 * np.log10(rms + 1e-12)
    out["peak_dbfs"] = 20 * np.log10(peak + 1e-12) if peak > 0 else -120.0
    out["clip_pct"] = 100.0 * float(np.mean(np.abs(y) >= 0.999))
    out["wada_snr_db"] = snrmod.wada_snr(y, calibration)
    out["noise_floor_dbfs"] = snrmod.noise_floor_dbfs(y, sr)
    # old ad-hoc estimator, for direct before/after comparison
    win = max(1, int(0.025 * (sr or 16000)))
    if n >= win:
        nf = n // win
        frames = y[: nf * win].reshape(nf, win)
        fr_db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1) + 1e-12) + 1e-12)
        out["old_pctile_snr_db"] = float(np.percentile(fr_db, 90) - np.percentile(fr_db, 10))
    else:
        out["old_pctile_snr_db"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description="T7 SNR estimator + quality audit")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--calibration_n", type=int, default=60,
                     help="Number of clips used to calibrate the WADA alpha->SNR mapping.")
    ap.add_argument("--wer_csv", default=os.path.join(_REPO_ROOT, "Part_B", "whisper_stt", "reports", "region_metrics.csv"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")

    mani = build_manifest(cfg.data_clean_root, cfg.bpac_subdir, None, verbose=False)
    n_in = len(mani)

    # -- calibrate WADA on a random sample of this corpus's own clips --
    rng = np.random.default_rng(42)
    cal_sample = mani.sample(min(args.calibration_n, len(mani)), random_state=42)
    clean_clips = []
    for p in cal_sample["filepath"]:
        y, sr = sf.read(p, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        clean_clips.append(y)
    calibration, cal_info = snrmod.calibrate(clean_clips)
    print(f"[T7] WADA calibration fit R^2={cal_info['fit_r2']:.3f} on {len(clean_clips)} clips")

    rows = []
    for r in maybe_tqdm(mani.itertuples(index=False), desc="T7 quality", total=len(mani)):
        stats_row = analyze_file_full(r.filepath, calibration)
        stats_row["utterance_id"] = r.utterance_id
        stats_row["region"] = r.region
        stats_row["gender"] = r.gender
        stats_row["pipeline"] = {"male": "lab", "female": "app"}[r.gender]
        rows.append(stats_row)
    df = pd.DataFrame(rows)
    save_table(df, out / "per_file_quality.csv", index=False)

    per_region = df.groupby("region").agg(
        n=("utterance_id", "size"),
        wada_snr_med=("wada_snr_db", "median"),
        old_snr_med=("old_pctile_snr_db", "median"),
        noise_floor_med=("noise_floor_dbfs", "median"),
        rms_dbfs_med=("rms_dbfs", "median"),
        clip_pct_med=("clip_pct", "median"),
    ).reset_index()
    save_table(per_region, out / "per_region_quality.csv", index=False)

    region_pipeline = df.groupby(["region", "pipeline"]).agg(
        n=("utterance_id", "size"),
        wada_snr_med=("wada_snr_db", "median"),
        noise_floor_med=("noise_floor_dbfs", "median"),
    ).reset_index()
    save_table(region_pipeline, out / "region_pipeline_quality.csv", index=False)

    # -- Kruskal-Wallis: does SNR differ by region? + effect size (epsilon-squared) --
    groups = [g["wada_snr_db"].dropna().values for _, g in df.groupby("region")]
    h_stat, p_val = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    epsilon_sq = (h_stat - k + 1) / (n_total - k) if n_total > k else float("nan")

    # -- Spearman: SNR vs per-region WER (does noise explain WER differences?) --
    wer_corr = None
    if os.path.exists(args.wer_csv):
        wer = pd.read_csv(args.wer_csv)[["region", "WER"]]
        merged = per_region.merge(wer, on="region", how="inner")
        rho, p_rho = stats.spearmanr(merged["wada_snr_med"], merged["WER"])
        wer_corr = {"n_regions": int(len(merged)), "spearman_rho": float(rho), "p_value": float(p_rho)}
        save_table(merged, out / "snr_wer_correlation.csv", index=False)
    else:
        print(f"[T7] WARNING WER csv not found: {args.wer_csv}")

    # -- figures --
    fig, ax = plt.subplots(figsize=(9, 5))
    regions = sorted(df["region"].unique())
    ax.boxplot([df[df["region"] == r]["wada_snr_db"].dropna() for r in regions],
               labels=regions, showfliers=False)
    ax.set_ylabel("WADA-style SNR (dB, locally calibrated)")
    ax.set_title("T7 -- SNR by region")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, fig_dir / "snr_by_region.png")
    fig.savefig(fig_dir / "snr_by_region.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([df[df["region"] == r]["noise_floor_dbfs"].dropna() for r in regions],
               labels=regions, showfliers=False)
    ax.set_ylabel("noise floor (dBFS, lowest-5% frames)")
    ax.set_title("T7 -- noise floor by region")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, fig_dir / "noise_floor_by_region.png")
    fig.savefig(fig_dir / "noise_floor_by_region.pdf", bbox_inches="tight")
    plt.close(fig)

    if wer_corr is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(merged["wada_snr_med"], merged["WER"])
        for _, row in merged.iterrows():
            ax.annotate(row["region"], (row["wada_snr_med"], row["WER"]), fontsize=7)
        ax.set_xlabel("median WADA-style SNR (dB)")
        ax.set_ylabel("WER (Whisper, primary model)")
        ax.set_title(f"T7 -- SNR vs WER (Spearman rho={wer_corr['spearman_rho']:.2f}, p={wer_corr['p_value']:.3f})")
        save_fig(fig, fig_dir / "snr_vs_wer_scatter.png")
        fig.savefig(fig_dir / "snr_vs_wer_scatter.pdf", bbox_inches="tight")
        plt.close(fig)

    summary = base_summary("T7_snr_quality", cfg.config_path, n_in, n_in)
    summary.update({
        "current_estimator_doc": CURRENT_ESTIMATOR_DOC,
        "wada_calibration_fit_r2": cal_info["fit_r2"],
        "kruskal_wallis_H": float(h_stat),
        "kruskal_wallis_p": float(p_val),
        "kruskal_wallis_epsilon_squared": float(epsilon_sq),
        "snr_wer_correlation": wer_corr,
        "overall_wada_snr_median": round(float(df["wada_snr_db"].median()), 2),
        "overall_noise_floor_median": round(float(df["noise_floor_dbfs"].median()), 2),
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T7 -- SNR Estimator & Quality Audit")
    rep.p(
        f"{CURRENT_ESTIMATOR_DOC} This adds a **named, documented WADA-style SNR estimator** "
        "(Kim & Stern 2008 log-amplitude-Gamma statistic, locally calibrated against synthetic "
        f"known-SNR mixtures built from this corpus's own clips -- calibration fit R^2="
        f"{cal_info['fit_r2']:.3f}; see `bpac_eda/snr.py` docstring for exactly what deviates "
        "from the original paper) plus a noise-floor measure."
    )
    rep.h2("Headline numbers")
    for k in ["overall_wada_snr_median", "overall_noise_floor_median",
              "kruskal_wallis_H", "kruskal_wallis_p", "kruskal_wallis_epsilon_squared"]:
        rep.bullet(f"`{k}` = {summary[k]}")
    rep.h2("Does SNR differ by region?")
    rep.p(
        f"Kruskal-Wallis H={h_stat:.2f}, p={p_val:.2e}, epsilon-squared={epsilon_sq:.4f} "
        "(effect size: 0=no effect, 1=region fully determines SNR rank). "
    )
    rep.h2("Region x pipeline (== gender, invariant 3 -- cannot separate the two)")
    rep.p("Lab (male) vs. app (female) recordings cannot be compared independently of gender.")
    rep.table(region_pipeline.round(2), index=False)
    if wer_corr is not None:
        rep.h2("Does noise explain the per-region WER differences?")
        rep.p(
            f"Spearman rho={wer_corr['spearman_rho']:.3f} (p={wer_corr['p_value']:.3f}) between "
            "median WADA-SNR and per-region Whisper WER, across "
            f"{wer_corr['n_regions']} regions. "
            + ("Weak/non-significant correlation -- SNR does **not** explain per-region WER "
               "differences, strengthening the case that WER differences reflect accent, not "
               "audio quality." if abs(wer_corr["spearman_rho"]) < 0.5 or wer_corr["p_value"] > 0.05
               else "Notable correlation -- audio quality is a plausible partial confound for "
               "the per-region WER differences and should be discussed, not ignored.")
        )
        rep.table(merged.round(3), index=False)
    rep.h2("Per-region summary")
    rep.table(per_region.round(2), index=False)
    rep.h2("Figures")
    rep.image("figures/snr_by_region.png", "WADA-style SNR by region.")
    rep.image("figures/noise_floor_by_region.png", "Noise floor by region.")
    if wer_corr is not None:
        rep.image("figures/snr_vs_wer_scatter.png", "SNR vs. WER scatter, the evidence for/against the noise-explains-WER claim.")
    rep.h2("For the paper")
    rep.p(
        "S5.1 quality audit: `snr_vs_wer_scatter` is the figure that rules out (or flags) noise "
        "as the WER explanation. Report the WADA-style estimator by name, with the calibration "
        "caveat above -- do not call it 'the WADA-SNR from Kim & Stern (2008)' without qualification."
    )
    rep.write()

    print(f"[T7] done. n={n_in}  wada_snr_med={summary['overall_wada_snr_median']}  "
          f"KW_p={p_val:.2e}  snr_wer_rho={wer_corr['spearman_rho'] if wer_corr else None}")


if __name__ == "__main__":
    main()
