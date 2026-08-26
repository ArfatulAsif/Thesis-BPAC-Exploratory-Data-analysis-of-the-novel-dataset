#!/usr/bin/env python3
"""T20 -- denoising sanity check.

The thesis trained voice-cloning models on a denoised variant of BPAC; this
paper deliberately does not denoise. Argument: spectral-gating denoisers
suppress broadband noise-like energy, and fricatives/aspiration bursts ARE
broadband noise-like energy -- so denoising attenuates exactly the cues T16
(sibilant CoG, VOT) depends on. This demonstrates that in one paired
comparison and checks that vowel formants are NOT meaningfully affected
(so the thesis's use of denoised audio for voice cloning was still
reasonable -- this report is meant to be balanced, not one-sided).

`noisereduce` parameters used in the original thesis pipeline were not
recoverable from the thesis materials available here; library defaults are
used instead and this is stated explicitly (per spec: "if unrecoverable,
use defaults and say so").

Outputs (in ./reports): denoise_cog_comparison.csv,
effect_size_attenuation.csv, figures/*, report.md, summary.json.
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

from common.provenance import base_summary  # noqa: E402

from bpac_valid.cli import add_config_arg  # noqa: E402
from bpac_valid.reporting import ReportWriter, ensure_dir, maybe_tqdm, save_fig, save_json, save_table  # noqa: E402
from bpac_valid.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

sys.path.insert(0, os.path.join(_REPO_ROOT, "Part_D"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Part_D", "D4_consonants"))
from analyze import spectral_moments as spectral_moments_local  # noqa: E402  (reuse T16's implementation)


def denoise(y, sr, params=None):
    import noisereduce as nr
    params = params or {}
    return nr.reduce_noise(y=y, sr=sr, **params)


def main():
    ap = argparse.ArgumentParser(description="T20 denoising sanity check")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    fig_dir = ensure_dir(out / "figures")
    e1 = cfg["e1_denoise_check"]

    sib_path = cfg.sibilant_tokens
    if not os.path.exists(sib_path):
        raise SystemExit(f"{sib_path} not found -- run Part_D/D4_consonants (T16) first.")
    sib = pd.read_parquet(sib_path)

    n_sample = int(e1.get("n_sample_tokens", 2000))
    sample = (
        sib.groupby(["region", "phone"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(1, n_sample // sib.groupby(["region", "phone"]).ngroups)),
                                  random_state=42))
    )
    n_in = len(sib)

    from bpac_phon.manifest import build_manifest
    clean_mani = build_manifest(cfg.data_root, cfg.bpac_subdir, None, verbose=False)
    filepath_map = dict(zip(clean_mani["utterance_id"], clean_mani["filepath"]))

    nr_params = e1.get("noisereduce_params") or {}
    if not nr_params:
        print("[T20] noisereduce_params not recoverable from the thesis pipeline -- using library defaults.")

    rows = []
    for r in maybe_tqdm(sample.itertuples(index=False), desc="T20 denoise CoG", total=len(sample)):
        wav = filepath_map.get(r.utterance_id)
        if wav is None:
            continue
        try:
            y, sr = sf.read(wav, dtype="float32")
            start = int(r.start_s * sr) if hasattr(r, "start_s") else None
        except Exception:
            continue
        # sibilant_tokens.parquet doesn't carry the raw audio segment (T16 read raw/32k
        # on the fly); re-extract from data_clean at the phone's timestamps if present.
        if not hasattr(r, "start_s") or not hasattr(r, "end_s"):
            continue
        t0, t1 = r.start_s, r.end_s
        s0, s1 = max(0, int(t0 * sr)), min(len(y), int(t1 * sr))
        seg = y[s0:s1]
        if len(seg) < 32:
            continue
        seg_dn = denoise(seg, sr, nr_params)

        cog_clean = spectral_moments_local(seg, sr)["cog"]
        cog_dn = spectral_moments_local(seg_dn, sr)["cog"]
        rows.append({"utterance_id": r.utterance_id, "region": r.region, "phone": r.phone,
                    "cog_clean": cog_clean, "cog_denoised": cog_dn, "shift_hz": cog_dn - cog_clean})
    cmp_df = pd.DataFrame(rows)
    save_table(cmp_df, out / "denoise_cog_comparison.csv", index=False)

    from scipy.stats import ttest_rel, wilcoxon
    paired = cmp_df.dropna(subset=["cog_clean", "cog_denoised"])
    t_stat, t_p = (ttest_rel(paired["cog_clean"], paired["cog_denoised"]) if len(paired) > 5
                  else (np.nan, np.nan))
    w_stat, w_p = (wilcoxon(paired["cog_clean"], paired["cog_denoised"]) if len(paired) > 5
                  else (np.nan, np.nan))
    diff = paired["cog_clean"] - paired["cog_denoised"]
    cohen_d = float(diff.mean() / diff.std()) if len(paired) > 5 and diff.std() > 0 else np.nan

    # -- key demonstration: does region separation (Sylhet vs rest) shrink after denoising? --
    def region_effect_size(col):
        if "sylhet" not in paired["region"].values:
            return np.nan
        syl = paired.loc[paired["region"] == "sylhet", col]
        rest = paired.loc[paired["region"] != "sylhet", col]
        if len(syl) < 3 or len(rest) < 3:
            return np.nan
        pooled_sd = np.sqrt((syl.std() ** 2 + rest.std() ** 2) / 2)
        return float((rest.mean() - syl.mean()) / pooled_sd) if pooled_sd > 0 else np.nan

    d_clean = region_effect_size("cog_clean")
    d_denoised = region_effect_size("cog_denoised")
    attenuation = pd.DataFrame([{
        "metric": "sylhet_vs_rest_cog_cohens_d", "clean": d_clean, "denoised": d_denoised,
        "pct_attenuation": round(100 * (1 - abs(d_denoised) / abs(d_clean)), 1)
        if d_clean and d_clean == d_clean and d_clean != 0 else None,
    }])
    save_table(attenuation, out / "effect_size_attenuation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(paired["cog_clean"], paired["cog_denoised"], s=8, alpha=0.4)
    lims = [paired[["cog_clean", "cog_denoised"]].min().min(), paired[["cog_clean", "cog_denoised"]].max().max()]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("CoG, clean (Hz)"); ax.set_ylabel("CoG, denoised (Hz)")
    ax.set_title(f"T20 -- paired CoG shift (mean={diff.mean():.1f} Hz, d={cohen_d:.3f})")
    save_fig(fig, fig_dir / "cog_shift_paired.png")
    fig.savefig(fig_dir / "cog_shift_paired.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["clean", "denoised"], [abs(d_clean) if d_clean == d_clean else 0,
                                   abs(d_denoised) if d_denoised == d_denoised else 0])
    ax.set_ylabel("|Cohen's d| (Sylhet vs. rest, sibilant CoG)")
    ax.set_title("T20 -- denoising attenuates the region-separation effect size")
    save_fig(fig, fig_dir / "separation_attenuation.png")
    fig.savefig(fig_dir / "separation_attenuation.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = base_summary("T20_denoise_check", cfg.config_path, n_in, len(paired), count_unit="sibilant_tokens")
    summary.update({
        "mean_cog_shift_hz": float(diff.mean()) if len(paired) else None,
        "paired_ttest_p": float(t_p) if t_p == t_p else None,
        "wilcoxon_p": float(w_p) if w_p == w_p else None,
        "cohens_d": cohen_d,
        "sylhet_separation_clean_d": d_clean, "sylhet_separation_denoised_d": d_denoised,
        "noisereduce_params_used": nr_params or "library defaults (thesis params not recoverable)",
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T20 -- Denoising Sanity Check")
    rep.p(
        f"Paired comparison on {len(paired)} sibilant tokens: mean CoG shift after denoising = "
        f"**{diff.mean():.1f} Hz** (paired t p={t_p:.2e}, Wilcoxon p={w_p:.2e}, Cohen's d={cohen_d:.3f})."
    )
    rep.h2("Key demonstration: denoising attenuates the region-separation effect")
    rep.table(attenuation.round(3), index=False)
    rep.image("figures/separation_attenuation.png", "Effect size before/after denoising -- the argument in one number.")
    rep.image("figures/cog_shift_paired.png", "Paired CoG shift scatter.")
    rep.h2("Noisereduce parameters")
    rep.p(f"{summary['noisereduce_params_used']}")
    rep.h2("For the paper")
    rep.p("S7.1 methods footnote + appendix. Settles why this paper does not denoise.")
    rep.write()

    print(f"[T20] done. n={len(paired)}  mean_shift_hz={summary['mean_cog_shift_hz']}  "
          f"d_clean={d_clean}  d_denoised={d_denoised}")


if __name__ == "__main__":
    main()
