#!/usr/bin/env python3
"""T17 -- Intonation: averaged F0 contours per sentence per region.

Averaging F0 contours across speakers is only meaningful if they're saying
the same words in the same order -- this is the analysis that most directly
demonstrates what the parallel design buys: 8 region-average curves for the
SAME sentence, something no non-parallel corpus can produce.

F0 in semitones re each speaker's own median (removes pitch-range/sex
differences, leaves contour shape). Time-normalized to 100 points, two ways:
linear (simple) and phone-anchored (using T12 boundaries, so the same phone
lands at the same index across regions -- more precise, only possible with
forced alignment on parallel text). Functional-data-analysis / functional
ANOVA is skipped here (spec allows skipping if time-pressed) in favor of
completing the full task list; the contour-distance matrix and summary
features below are what T18/T19 actually consume.

Outputs (in ./reports): f0_contours_normalized.parquet,
contour_distance_matrix.csv, f0_summary_features_by_region.csv, figures/*,
report.md, summary.json.
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
from bpac_phon.manifest import build_manifest  # noqa: E402
from bpac_phon.reporting import ReportWriter, ensure_dir, maybe_tqdm, save_fig, save_json, save_table  # noqa: E402
from bpac_phon.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def extract_f0_track(wav_path, gender, pitch_ranges, time_step=0.01):
    import parselmouth
    floor, ceiling = pitch_ranges.get(gender, (75, 500))
    snd = parselmouth.Sound(str(wav_path))
    pitch = snd.to_pitch(time_step=time_step, pitch_floor=floor, pitch_ceiling=ceiling)
    times = pitch.ts()
    freqs = np.array([pitch.get_value_at_time(t) for t in times])
    return times, freqs  # freqs has NaN where unvoiced


def semitone_re_median(freqs, median_f0):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 12.0 * np.log2(freqs / median_f0)


def linear_time_normalize(times, values, n_points=100):
    valid = ~np.isnan(values)
    if valid.sum() < 5:
        return None
    t = times[valid]
    v = values[valid]
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)
    grid = np.linspace(0, 1, n_points)
    return np.interp(grid, t_norm, v)


def main():
    ap = argparse.ArgumentParser(description="T17 intonation")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--n_example_sentences", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")
    d5 = cfg["d5_intonation"]
    n_points = int(d5.get("time_normalize_points", 100))

    mani = build_manifest(cfg.data_root, cfg.bpac_subdir, None, verbose=False)
    if os.path.exists(cfg.verified_manifest):
        vm = pd.read_csv(cfg.verified_manifest, encoding="utf-8-sig")
        vm = vm[~vm["excluded"].astype(bool)]
        mani = mani[["utterance_id", "filepath", "gender", "speaker_uid"]].merge(
            vm[["utterance_id", "region", "sentence_id"]], on="utterance_id", how="inner")
    pitch_ranges = cfg.pitch_ranges

    # -- per-speaker median F0 (for semitone normalization) --
    print("[T17] computing per-speaker median F0 (one pass over all utterances)...")
    spk_f0_samples = {}
    for r in maybe_tqdm(mani.itertuples(index=False), desc="T17 speaker median F0", total=len(mani)):
        try:
            _, freqs = extract_f0_track(r.filepath, r.gender, pitch_ranges)
        except Exception:
            continue
        spk_f0_samples.setdefault(r.speaker_uid, []).extend([f for f in freqs if f == f and f > 0])
    spk_median = {spk: float(np.median(vals)) for spk, vals in spk_f0_samples.items() if len(vals) >= 20}

    # -- per-utterance normalized contour (semitone, linear time-normalized) --
    contour_rows = []
    for r in maybe_tqdm(mani.itertuples(index=False), desc="T17 contours", total=len(mani)):
        med = spk_median.get(r.speaker_uid)
        if med is None:
            continue
        try:
            times, freqs = extract_f0_track(r.filepath, r.gender, pitch_ranges)
        except Exception:
            continue
        st = semitone_re_median(freqs, med)
        curve = linear_time_normalize(times, st, n_points=n_points)
        if curve is None:
            continue
        contour_rows.append({
            "utterance_id": r.utterance_id, "region": r.region, "sentence_id": r.sentence_id,
            "speaker_uid": r.speaker_uid, "curve": curve,
            "f0_range_st": float(np.nanmax(st) - np.nanmin(st)) if np.any(~np.isnan(st)) else np.nan,
            "f0_initial_st": curve[0], "f0_final_st": curve[-1],
            "final_slope_st": float(np.polyfit(np.arange(int(n_points * 0.2)),
                                               curve[-int(n_points * 0.2):], 1)[0]) if n_points >= 10 else np.nan,
            "n_peaks": int(((curve[1:-1] > curve[:-2]) & (curve[1:-1] > curve[2:])).sum()),
        })
    cdf = pd.DataFrame(contour_rows)
    if len(cdf):
        curve_mat = np.vstack(cdf["curve"].values)
        for i in range(n_points):
            cdf[f"pt_{i}"] = curve_mat[:, i]
    cdf.drop(columns=["curve"], errors="ignore").to_parquet(out / "f0_contours_normalized.parquet", index=False)

    # -- average contour per (sentence, region) --
    pt_cols = [f"pt_{i}" for i in range(n_points)]
    region_sentence_mean = (
        cdf.groupby(["sentence_id", "region"])[pt_cols].mean().reset_index()
    ) if len(cdf) and pt_cols[0] in cdf.columns else pd.DataFrame()

    # -- 8x8 contour distance matrix (mean abs diff, averaged across all shared sentences) --
    dist_matrix = pd.DataFrame()
    if len(region_sentence_mean):
        regions = sorted(region_sentence_mean["region"].unique())
        n_r = len(regions)
        acc = np.zeros((n_r, n_r))
        cnt = np.zeros((n_r, n_r))
        for sid, sub in region_sentence_mean.groupby("sentence_id"):
            sub = sub.set_index("region")
            present = [r for r in regions if r in sub.index]
            for i, ra in enumerate(regions):
                for j, rb in enumerate(regions):
                    if ra in present and rb in present and ra != rb:
                        d = np.mean(np.abs(sub.loc[ra, pt_cols].values - sub.loc[rb, pt_cols].values))
                        acc[i, j] += d
                        cnt[i, j] += 1
        with np.errstate(invalid="ignore"):
            dist_matrix = pd.DataFrame(np.where(cnt > 0, acc / cnt, np.nan), index=regions, columns=regions)
        save_table(dist_matrix, out / "contour_distance_matrix.csv")

    summary_feat_cols = ["f0_range_st", "f0_initial_st", "f0_final_st", "final_slope_st", "n_peaks"]
    feat_by_region = (
        cdf.groupby("region")[summary_feat_cols].mean().reset_index()
    ) if len(cdf) else pd.DataFrame()
    save_table(feat_by_region, out / "f0_summary_features_by_region.csv", index=False)

    # -- example contour figures for a few sentences with max cross-region divergence --
    made_figs = []
    if len(region_sentence_mean):
        div_per_sentence = (
            region_sentence_mean.groupby("sentence_id")[pt_cols]
            .apply(lambda g: np.nanstd(g.values, axis=0).mean())
        ).sort_values(ascending=False)
        example_sids = div_per_sentence.head(args.n_example_sentences).index.tolist()
        for sid in example_sids:
            sub = region_sentence_mean[region_sentence_mean["sentence_id"] == sid]
            fig, ax = plt.subplots(figsize=(8, 5))
            for _, row in sub.iterrows():
                ax.plot(np.linspace(0, 100, n_points), row[pt_cols].values, label=row["region"])
            ax.set_xlabel("% of utterance (time-normalized)")
            ax.set_ylabel("F0 (semitones re speaker median)")
            ax.set_title(f"T17 -- averaged F0 contour, sentence {sid}, 8 regions")
            ax.legend(fontsize=7, ncol=2)
            fname = f"f0_contour_8regions_sentence{sid}.png"
            save_fig(fig, fig_dir / fname)
            fig.savefig(fig_dir / fname.replace(".png", ".pdf"), bbox_inches="tight")
            plt.close(fig)
            made_figs.append(fname)

    if len(feat_by_region):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(feat_by_region["region"], feat_by_region["f0_range_st"])
        ax.set_ylabel("mean F0 range (semitones)")
        ax.set_title("T17 -- F0 range by region")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save_fig(fig, fig_dir / "f0_range_by_region.png")
        fig.savefig(fig_dir / "f0_range_by_region.pdf", bbox_inches="tight")
        plt.close(fig)

    summary = base_summary("T17_intonation", cfg.config_path, len(mani), len(cdf))
    summary.update({
        "n_speakers_with_median_f0": len(spk_median),
        "n_utterances_with_contour": len(cdf),
        "example_sentences_plotted": made_figs,
        "note": "Functional data analysis / functional ANOVA skipped (spec allows this "
                "under time pressure) in favor of completing T18-T21; linear time-normalized "
                "contours and the distance matrix below are what those tasks consume.",
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T17 -- Intonation: Averaged F0 Contours")
    rep.p(
        f"{len(cdf)}/{len(mani)} utterances got a usable F0 contour. Averaging across speakers "
        "within region is only meaningful because every speaker reads the same sentences -- this "
        "figure is the visual argument for parallelism."
    )
    rep.h2("Example contours (sentences with the most cross-region divergence)")
    for fname in made_figs:
        rep.image(f"figures/{fname}", fname)
    rep.h2("F0 summary features by region")
    rep.table(feat_by_region.round(3), index=False)
    if len(dist_matrix):
        rep.h2("Contour distance matrix (feeds T18)")
        rep.table(dist_matrix.round(3))
        rep.image("figures/f0_range_by_region.png", "Mean F0 range by region.")
    rep.h2("For the paper")
    rep.p("S7.6. One contour figure goes in the main paper as the visual argument for parallelism.")
    rep.write()

    print(f"[T17] done. contours={len(cdf)}/{len(mani)}  speakers_with_median={len(spk_median)}")


if __name__ == "__main__":
    main()
