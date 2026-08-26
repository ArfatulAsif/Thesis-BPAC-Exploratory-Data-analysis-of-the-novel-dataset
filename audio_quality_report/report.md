# Audio-quality diagnostic — per region

Scanned **12750** files across **8** regions. Audio read at native sample rate; nothing modified.

## Per-region summary

| region     |   n_files |   n_empty |   n_load_errors | sample_rates   | channels   |   dur_med_s |   dur_p95_s |   dur_max_s |   rms_dbfs_med |   peak_dbfs_med |   snr_db_med |   clip_pct_med |   silence_pct_med |
|:-----------|----------:|----------:|----------------:|:---------------|:-----------|------------:|------------:|------------:|---------------:|----------------:|-------------:|---------------:|------------------:|
| chattogram |      1500 |         0 |               0 | 44100/48000    | 1/2        |       7.07  |      16.481 |      96.36  |        -26.417 |          -7.076 |       38.736 |              0 |            16.183 |
| dhaka      |      1500 |         0 |               0 | 44100/48000    | 1/2        |       7.28  |      16.503 |      79.71  |        -21.583 |          -6.015 |       42.27  |              0 |            21.503 |
| faridpur   |      1750 |         0 |               0 | 44100/48000    | 1/2        |       6.885 |      15.071 |      97.93  |        -24.109 |          -6.02  |       44.645 |              0 |            18.182 |
| khulna     |      1500 |         0 |               0 | 44100/48000    | 1/2        |       6.82  |      16.335 |      83.383 |        -26.547 |          -7.686 |       32.165 |              0 |            17.578 |
| mymensingh |      1250 |         0 |               0 | 44100/48000    | 1/2        |       6.89  |      16.181 |      88.99  |        -20.513 |          -5.168 |       31.674 |              0 |            17.168 |
| rajshahi   |      1500 |         0 |               0 | 44100/48000    | 1/2        |       6.664 |      14.304 |      77.79  |        -20.051 |          -6.015 |       38.116 |              0 |            22.048 |
| rangpur    |      1250 |         0 |               0 | 44100/48000    | 1/2        |       6.57  |      15.503 |      89.17  |        -24.005 |          -6.366 |       31.588 |              0 |            17.506 |
| sylhet     |      2500 |         0 |               0 | 44100/48000    | 1/2        |       6.13  |      14.081 |      93.25  |        -22.979 |          -6.04  |       41.719 |              0 |            14.815 |

## Automatic flags (is a region different from the rest?)

`*_z` are robust z-scores vs the median region; `|z|>=3.5`, a sample-rate/channel mismatch, or dead files mark a region SUSPECT.

| region     |   dur_med_s_z |   rms_dbfs_med_z |   snr_db_med_z |   silence_pct_med_z |   clip_pct_med_z | sr_mismatch   | multi_sr_within_region   | multi_ch_within_region   | has_empty_files   |   max_abs_z | SUSPECT   |
|:-----------|--------------:|-----------------:|---------------:|--------------------:|-----------------:|:--------------|:-------------------------|:-------------------------|:------------------|------------:|:----------|
| rajshahi   |         -0.63 |             0.96 |          -0.04 |                3.04 |                0 | False         | True                     | True                     | False             |        3.04 | True      |
| dhaka      |          1.42 |             0.53 |           0.52 |                2.67 |                0 | False         | True                     | True                     | False             |        2.67 | True      |
| sylhet     |         -2.4  |             0.14 |           0.44 |               -1.84 |                0 | False         | True                     | True                     | False             |        2.4  | True      |
| rangpur    |         -0.94 |            -0.14 |          -0.92 |               -0.02 |                0 | False         | True                     | True                     | False             |        0.94 | True      |
| chattogram |          0.72 |            -0.82 |           0.04 |               -0.92 |                0 | False         | True                     | True                     | False             |        0.92 | True      |
| mymensingh |          0.12 |             0.83 |          -0.91 |               -0.25 |                0 | False         | True                     | True                     | False             |        0.91 | True      |
| khulna     |         -0.11 |            -0.85 |          -0.84 |                0.02 |                0 | False         | True                     | True                     | False             |        0.85 | True      |
| faridpur   |          0.11 |            -0.17 |           0.83 |                0.43 |                0 | False         | True                     | True                     | False             |        0.83 | True      |

## Verdict

**Regions that look different from the rest:** rajshahi, dhaka, sylhet, rangpur, chattogram, mymensingh, khulna, faridpur.

- **dhaka**: mixed sample rates (44100/48000); mixed channels (1/2).
- **rajshahi**: mixed sample rates (44100/48000); mixed channels (1/2).

## How to read this
- If a focus region is **not** flagged here, its high ASR error is probably **not** an audio-quality problem — check the reference/parallel alignment instead (does `sound_NNN.wav` actually say sentence id N?).
- If it **is** flagged (different sample rate, much lower SNR, much louder/quieter, or far longer clips), that recording-condition difference is a plausible cause of the per-region error and is a real confound to address.

Files: `per_file_audio_stats.csv`, `per_region_audio_summary.csv`, `region_quality_flags.csv`, and the PNG figures.