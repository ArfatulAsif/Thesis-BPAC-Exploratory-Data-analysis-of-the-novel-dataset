# C0 — Corpus Normalization & Diagnostics

Wrote a clean 16 kHz mono corpus to `C:\Users\redwa\OneDrive\Desktop\Thesis EDA\Thesis-BPAC-Exploratory-Data-analysis-of-the-novel-dataset\data_clean` (peak-normalized=True). C1/C2 read this tree.

## Normalization totals

- Input clips: **12750**; written: **0**; already present: 12747; dead/empty dropped: **3**; load failures: 0.
## T5 -- re-freeze from verified_manifest.csv

Re-normalized from `../verified_manifest.csv` (Phase 1): **0** utterances excluded by T4 review (vs. the plain dead-file drop above), corrected `sentence_id` mappings honored. See Part_B/alignment_verification/reports/report.md for the ASR-based alignment verification rate (T3/T4) that this corpus reflects -- **check whether that rate is still `provisional` before citing this freeze as final.**

- `checksums.csv`: sha256 for all **12747** files in `data_clean/`.
- `CORPUS_FROZEN.json` -> `C:\Users\redwa\OneDrive\Desktop\Thesis EDA\Thesis-BPAC-Exploratory-Data-analysis-of-the-novel-dataset\CORPUS_FROZEN.json` (checksum_of_checksums=`e2124c04809ef425...`).
## Per-region summary

| region     |   n_input |   n_written |   n_exists_skipped |   n_dead_skipped |   n_load_failed |   median_in_dur_s |
|:-----------|----------:|------------:|-------------------:|-----------------:|----------------:|------------------:|
| mymensingh |      1250 |           0 |               1248 |                2 |               0 |             6.89  |
| faridpur   |      1750 |           0 |               1749 |                1 |               0 |             6.885 |
| chattogram |      1500 |           0 |               1500 |                0 |               0 |             7.07  |
| dhaka      |      1500 |           0 |               1500 |                0 |               0 |             7.28  |
| khulna     |      1500 |           0 |               1500 |                0 |               0 |             6.82  |
| rajshahi   |      1500 |           0 |               1500 |                0 |               0 |             6.664 |
| rangpur    |      1250 |           0 |               1250 |                0 |               0 |             6.57  |
| sylhet     |      2500 |           0 |               2500 |                0 |               0 |             6.13  |

## Parallel-alignment check (duration-based)

For each shared sentence, clip durations should be similar across regions. A region with many duration-deviant sentences may have `sound_NNN.wav` not matching sentence id N. Higher `pct_suspect` = more alignment-suspect.

| region     |   n_sentences |   n_suspect |   mean_rel_dev |   pct_suspect | focus   |
|:-----------|--------------:|------------:|---------------:|--------------:|:--------|
| dhaka      |           250 |           3 |      0.125269  |           1.2 | True    |
| khulna     |           250 |           2 |      0.0630826 |           0.8 | False   |
| rajshahi   |           250 |           1 |      0.0611212 |           0.4 | True    |
| rangpur    |           250 |           1 |      0.0439117 |           0.4 | False   |
| chattogram |           250 |           0 |      0.0965734 |           0   | False   |
| faridpur   |           250 |           0 |      0.0511916 |           0   | False   |
| mymensingh |           250 |           0 |      0.0614619 |           0   | False   |
| sylhet     |           250 |           0 |      0.0631834 |           0   | False   |

- Inspect `alignment_suspect_cells.csv` for the specific (region, sentence) pairs; spot-check those files against the CSV.
## Next

- Run C1 and C2 — they now read the clean corpus and avoid the format confound.