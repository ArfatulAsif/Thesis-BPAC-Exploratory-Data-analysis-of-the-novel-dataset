#!/usr/bin/env python3
"""
check_audio_quality.py — per-region audio-quality diagnostic for the BPAC corpus.

Place this file in the project ROOT (the folder that contains `data/`) and run:

    python check_audio_quality.py
    python check_audio_quality.py --focus dhaka rajshahi      # extra detail on these
    python check_audio_quality.py --data-root data --workers 4

It answers two questions:
  1) Is some region's AUDIO different from the others? (sample rate, channels,
     duration, loudness/RMS, estimated SNR, clipping, silence, dead files)
  2) Could the spike be a DATA bug rather than audio? It reports per-region
     sample-rate/channel consistency and dead/near-empty files, which are the
     usual causes of a whole region scoring badly.

Outputs (written next to this script, in ./audio_quality_report/):
  - per_file_audio_stats.csv      one row per .wav  (the raw data)
  - per_region_audio_summary.csv  medians/means per region (the headline table)
  - region_quality_flags.csv      automatic "looks off vs the rest" flags
  - figures: duration / RMS_dBFS / SNR / sample-rate by region
  - report.md                     human-readable summary

Only needs: numpy, pandas, soundfile, matplotlib, tqdm (optional).
Audio is read at native sample rate; nothing is modified.
"""
import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SPEAKER_RE = re.compile(r"^speaker_(?P<gender>male|female)_(?P<id>\d+)_(?P<name>.+)$", re.IGNORECASE)
SOUND_RE = re.compile(r"^sound_(?P<num>\d+)\.wav$", re.IGNORECASE)


# ---------------------------------------------------------------- file discovery
def find_wavs(bpac_dir: Path):
    rows = []
    for region_dir in sorted(p for p in bpac_dir.iterdir() if p.is_dir()):
        region = region_dir.name.lower()
        for spk_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            m = SPEAKER_RE.match(spk_dir.name)
            if not m:
                continue
            gender, sid, name = m.group("gender").lower(), m.group("id"), m.group("name")
            speaker_uid = f"{region}_{gender}_{sid}_{name}".lower()
            for wav in sorted(spk_dir.glob("*.wav")):
                sm = SOUND_RE.match(wav.name)
                rows.append({
                    "filepath": str(wav),
                    "region": region,
                    "gender": gender,
                    "speaker_uid": speaker_uid,
                    "sentence_id": int(sm.group("num")) if sm else None,
                    "conforming_name": bool(sm),
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- audio stats
def analyze_file(path: str) -> dict:
    """Return audio-quality metrics for one wav (native sample rate, no resample)."""
    import soundfile as sf

    out = {
        "ok": False, "sample_rate": None, "channels": None, "subtype": None,
        "n_samples": 0, "duration_s": np.nan, "rms_dbfs": np.nan, "peak_dbfs": np.nan,
        "snr_db": np.nan, "clip_pct": np.nan, "silence_pct": np.nan, "is_empty": True,
        "error": "",
    }
    try:
        info = sf.info(path)
        out.update(sample_rate=info.samplerate, channels=info.channels, subtype=info.subtype)
        y, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        out["error"] = str(e)[:200]
        return out

    if y.ndim > 1:           # stereo -> mono for the measurement only
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    n = y.size
    out["n_samples"] = int(n)
    out["is_empty"] = n == 0
    if n == 0:
        out["ok"] = True
        return out

    out["duration_s"] = n / float(sr or 1)
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    out["rms_dbfs"] = 20 * np.log10(rms + 1e-12)
    out["peak_dbfs"] = 20 * np.log10(peak + 1e-12) if peak > 0 else -120.0
    # fraction of samples at/near full scale (clipping)
    out["clip_pct"] = 100.0 * float(np.mean(np.abs(y) >= 0.999))

    # Frame-based loudness for a rough SNR + silence estimate.
    win = max(1, int(0.025 * (sr or 16000)))           # 25 ms frames
    if n >= win:
        nf = n // win
        frames = y[:nf * win].reshape(nf, win)
        fr_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        fr_db = 20 * np.log10(fr_rms + 1e-12)
        # noise floor = 10th percentile frame, speech = 90th percentile frame
        noise_db = float(np.percentile(fr_db, 10))
        speech_db = float(np.percentile(fr_db, 90))
        out["snr_db"] = speech_db - noise_db
        # silence = frames within 3 dB of the noise floor
        out["silence_pct"] = 100.0 * float(np.mean(fr_db <= noise_db + 3.0))
    out["ok"] = True
    return out


# ------------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(description="Per-region audio-quality diagnostic.")
    ap.add_argument("--data-root", default="data", help="Folder containing BPAC/ (default: data)")
    ap.add_argument("--bpac-subdir", default="BPAC")
    ap.add_argument("--focus", nargs="*", default=["dhaka", "rajshahi"],
                    help="Regions to highlight in the report (default: dhaka rajshahi)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1).")
    ap.add_argument("--limit-per-speaker", type=int, default=None,
                    help="Sample N files/speaker for a quick pass (default: all).")
    ap.add_argument("--out", default=None, help="Output dir (default: ./audio_quality_report)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    bpac = Path(args.data_root) / args.bpac_subdir
    if not bpac.is_dir():
        raise SystemExit(f"BPAC folder not found: {bpac}  (run from the project root, "
                         f"or pass --data-root)")
    out = Path(args.out) if args.out else here / "audio_quality_report"
    out.mkdir(parents=True, exist_ok=True)

    files = find_wavs(bpac)
    if files.empty:
        raise SystemExit(f"No wavs found under {bpac}")
    if args.limit_per_speaker:
        files = (files.sort_values(["speaker_uid", "sentence_id"])
                 .groupby("speaker_uid", group_keys=False)
                 .head(args.limit_per_speaker).reset_index(drop=True))
    print(f"[scan] {len(files)} wav files across {files['region'].nunique()} regions")

    paths = files["filepath"].tolist()
    try:
        from tqdm import tqdm
        it = tqdm(paths, desc="analyzing")
    except Exception:
        it = paths

    if args.workers and args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            stats = list(ex.map(analyze_file, paths, chunksize=16))
    else:
        stats = [analyze_file(p) for p in it]

    df = pd.concat([files.reset_index(drop=True), pd.DataFrame(stats)], axis=1)
    df.to_csv(out / "per_file_audio_stats.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------- per-region summary table
    def agg(g):
        return pd.Series({
            "n_files": len(g),
            "n_empty": int(g["is_empty"].sum()),
            "n_load_errors": int((~g["ok"]).sum()),
            "sample_rates": "/".join(sorted({str(int(x)) for x in g["sample_rate"].dropna().unique()})),
            "channels": "/".join(sorted({str(int(x)) for x in g["channels"].dropna().unique()})),
            "dur_med_s": g["duration_s"].median(),
            "dur_p95_s": g["duration_s"].quantile(0.95),
            "dur_max_s": g["duration_s"].max(),
            "rms_dbfs_med": g["rms_dbfs"].median(),
            "peak_dbfs_med": g["peak_dbfs"].median(),
            "snr_db_med": g["snr_db"].median(),
            "clip_pct_med": g["clip_pct"].median(),
            "silence_pct_med": g["silence_pct"].median(),
        })

    summary = df.groupby("region").apply(agg).reset_index()
    summary = summary.sort_values("region").reset_index(drop=True)
    summary.to_csv(out / "per_region_audio_summary.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ automatic "off vs rest" flags
    # Compare each region to the median-of-regions using robust (MAD) z-scores.
    metrics = ["dur_med_s", "rms_dbfs_med", "snr_db_med", "silence_pct_med", "clip_pct_med"]
    flags = summary[["region"]].copy()
    for m in metrics:
        vals = summary[m].astype(float)
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med)) or 1e-9
        z = 0.6745 * (vals - med) / mad           # robust z
        flags[m + "_z"] = z.round(2)
    # a region is "suspect" if any metric is a strong outlier, or it has
    # inconsistent sample rate/channels vs the corpus, or dead files
    corpus_sr = set()
    for s in summary["sample_rates"]:
        corpus_sr.update(s.split("/"))
    flags["sr_mismatch"] = summary["sample_rates"].apply(lambda s: set(s.split("/")) != corpus_sr if len(corpus_sr) > 1 else False)
    flags["multi_sr_within_region"] = summary["sample_rates"].str.contains("/")
    flags["multi_ch_within_region"] = summary["channels"].str.contains("/")
    flags["has_empty_files"] = summary["n_empty"] > 0
    zcols = [c for c in flags.columns if c.endswith("_z")]
    flags["max_abs_z"] = flags[zcols].abs().max(axis=1).round(2)
    flags["SUSPECT"] = (
        (flags["max_abs_z"] >= 3.5)
        | flags["sr_mismatch"] | flags["multi_sr_within_region"]
        | flags["multi_ch_within_region"] | flags["has_empty_files"]
    )
    flags = flags.sort_values("max_abs_z", ascending=False).reset_index(drop=True)
    flags.to_csv(out / "region_quality_flags.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        regions = sorted(df["region"].unique())
        focus = set(r.lower() for r in args.focus)

        def _box(col, title, ylabel, fname):
            data = [df[df["region"] == r][col].dropna().values for r in regions]
            fig, ax = plt.subplots(figsize=(max(7, len(regions) * 1.1), 5))
            bp = ax.boxplot(data, labels=regions, showfliers=False, patch_artist=True)
            for i, r in enumerate(regions):
                bp["boxes"][i].set_facecolor("salmon" if r in focus else "lightsteelblue")
            ax.set_title(title); ax.set_ylabel(ylabel)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout(); fig.savefig(out / fname, dpi=150); plt.close(fig)

        _box("duration_s", "Per-region clip duration (focus = red)", "seconds", "duration_by_region.png")
        _box("rms_dbfs", "Per-region loudness (RMS dBFS)", "dBFS", "rms_by_region.png")
        _box("snr_db", "Per-region estimated SNR", "dB", "snr_by_region.png")

        # sample-rate composition bar
        srtab = (df.assign(sr=df["sample_rate"].fillna(0).astype(int).astype(str))
                 .groupby(["region", "sr"]).size().unstack(fill_value=0))
        fig, ax = plt.subplots(figsize=(max(7, len(regions) * 1.1), 5))
        srtab.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Sample-rate composition by region"); ax.set_ylabel("files")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(out / "sample_rate_by_region.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"[figures] skipped: {e}")

    # -------------------------------------------------------------------- report
    lines = ["# Audio-quality diagnostic — per region", ""]
    lines += [f"Scanned **{len(df)}** files across **{df['region'].nunique()}** regions. "
              "Audio read at native sample rate; nothing modified.", ""]
    lines += ["## Per-region summary", "", summary.round(3).to_markdown(index=False), ""]
    lines += ["## Automatic flags (is a region different from the rest?)", "",
              "`*_z` are robust z-scores vs the median region; `|z|>=3.5`, a sample-rate/"
              "channel mismatch, or dead files mark a region SUSPECT.", "",
              flags.round(2).to_markdown(index=False), ""]

    suspects = flags[flags["SUSPECT"]]["region"].tolist()
    lines += ["## Verdict", ""]
    if suspects:
        lines.append(f"**Regions that look different from the rest:** {', '.join(suspects)}.")
    else:
        lines.append("No region stands out on audio quality — sample rate, loudness, SNR, "
                     "duration and silence are comparable across regions.")
    lines.append("")
    for r in sorted(focus):
        if r in set(summary["region"]):
            row = summary[summary["region"] == r].iloc[0]
            rest = summary[summary["region"] != r]
            note = []
            if row["n_empty"] > 0:
                note.append(f"{int(row['n_empty'])} empty/dead file(s)")
            if "/" in str(row["sample_rates"]):
                note.append(f"mixed sample rates ({row['sample_rates']})")
            if "/" in str(row["channels"]):
                note.append(f"mixed channels ({row['channels']})")
            if abs(row["snr_db_med"] - rest["snr_db_med"].median()) >= 5:
                note.append(f"SNR {row['snr_db_med']:.1f} dB vs ~{rest['snr_db_med'].median():.1f} dB elsewhere")
            if abs(row["rms_dbfs_med"] - rest["rms_dbfs_med"].median()) >= 4:
                note.append(f"loudness {row['rms_dbfs_med']:.1f} dBFS vs ~{rest['rms_dbfs_med'].median():.1f} elsewhere")
            if row["dur_med_s"] >= 1.6 * rest["dur_med_s"].median():
                note.append(f"clips ~{row['dur_med_s']:.1f}s vs ~{rest['dur_med_s'].median():.1f}s elsewhere")
            verdict = "; ".join(note) if note else "looks comparable to the other regions on audio metrics"
            lines.append(f"- **{r}**: {verdict}.")
    lines += ["", "## How to read this",
              "- If a focus region is **not** flagged here, its high ASR error is probably "
              "**not** an audio-quality problem — check the reference/parallel alignment "
              "instead (does `sound_NNN.wav` actually say sentence id N?).",
              "- If it **is** flagged (different sample rate, much lower SNR, much louder/"
              "quieter, or far longer clips), that recording-condition difference is a "
              "plausible cause of the per-region error and is a real confound to address.",
              "", "Files: `per_file_audio_stats.csv`, `per_region_audio_summary.csv`, "
              "`region_quality_flags.csv`, and the PNG figures."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------- console digest
    print("\n=== per-region summary ===")
    cols = ["region", "n_files", "n_empty", "sample_rates", "channels",
            "dur_med_s", "rms_dbfs_med", "snr_db_med", "silence_pct_med"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[cols].round(2).to_string(index=False))
    print("\n=== suspect regions ===")
    print(", ".join(suspects) if suspects else "(none flagged on audio quality)")
    print(f"\nWrote report + CSVs + figures to: {out}")


if __name__ == "__main__":
    main()
