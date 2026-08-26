#!/usr/bin/env python3
"""T21 -- publication-quality figure regeneration/collection.

Every analyze.py in this repo already saves each figure as BOTH .png
(150 dpi, for reports) and .pdf (vector, for LaTeX) per spec invariant 5 --
so "regeneration" is mostly re-running the upstream task if its numbers
changed, and this script's job is the last mile: collect the headline
figure from each completed task into paper/figures/ with a stable,
citable name, and refuse to silently overwrite a figure with a missing
counterpart format.

This is deliberately a MANIFEST-DRIVEN copy step, not a re-plotting step --
re-plotting belongs in each task's own analyze.py so there is exactly one
source of truth per figure.

Run: python release/collect_figures.py
"""
import os
import shutil
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (paper figure name, source relative to repo root, required)
FIGURE_MANIFEST = [
    ("fig_coverage_heatmap", "Part_A/A1_unique_sentences/reports/coverage_heatmap.png", False),
    ("fig_wpm_by_region_vad", "Part_A/A7_vad_speaking_rate/reports/figures/wpm_by_region", True),
    ("fig_snr_vs_wer", "Part_A/A8_audio_quality_snr/reports/figures/snr_vs_wer_scatter", True),
    ("fig_cer_distribution_by_region", "Part_B/alignment_verification/reports/figures/cer_distribution_by_region", True),
    ("fig_loso_accuracy_with_ci", "Part_C/C3_LOSO_within_gender/reports/figures/loso_accuracy_with_ci", True),
    ("fig_distance_heatmap_embedding", "Part_C/C4_distance_statistics/reports/figures/distance_heatmap", True),
    ("fig_dendrogram_embedding", "Part_C/C4_distance_statistics/reports/figures/dendrogram_embedding", True),
    ("fig_pctV_deltaC_scatter", "Part_D/D2_duration_rhythm/reports/figures/pctV_deltaC_scatter", True),
    ("fig_vowel_quadrilateral", "Part_D/D3_vowel_space/reports/figures/vowel_quadrilateral_overlay", True),
    ("fig_sibilant_cog_density", "Part_D/D4_consonants/reports/figures/cog_density_by_region", True),
    ("fig_f0_range_by_region", "Part_D/D5_intonation/reports/figures/f0_range_by_region", True),
    ("fig_dendrogram_phonetic", "Part_D/D6_phonetic_distance/reports/figures/dendrogram_phonetic", True),
    ("fig_mds_procrustes_overlay", "Part_D/D7_convergence/reports/figures/mds_procrustes_overlay", True),
    ("fig_dendrogram_tanglegram", "Part_D/D7_convergence/reports/figures/dendrograms_side_by_side", True),
    ("fig_denoise_separation_attenuation", "Part_E/E1_denoise_check/reports/figures/separation_attenuation", True),
]


def main():
    dest_dir = os.path.join(_REPO_ROOT, "paper", "figures")
    os.makedirs(dest_dir, exist_ok=True)

    copied, missing = [], []
    for name, rel, is_stem in FIGURE_MANIFEST:
        exts = [".png", ".pdf"] if is_stem else [""]
        srcs = [os.path.join(_REPO_ROOT, rel + e) for e in exts] if is_stem else [os.path.join(_REPO_ROOT, rel)]
        for src in srcs:
            if not os.path.exists(src):
                missing.append(src)
                continue
            ext = os.path.splitext(src)[1]
            dst = os.path.join(dest_dir, name + ext)
            shutil.copy2(src, dst)
            copied.append(dst)

    print(f"[T21] copied {len(copied)} figure files to {dest_dir}")
    if missing:
        print(f"[T21] {len(missing)} source figures not found yet (upstream task not run/finished):")
        for m in missing:
            print(f"       - {os.path.relpath(m, _REPO_ROOT)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
