# T16 -- Consonants: Sibilant Spectral Moments + VOT

**Sample-rate exception**: sibilant and VOT tokens are read from RAW audio resampled to 32 kHz, not data_clean/ (16 kHz) -- see module docstring / invariant 2.

## Sibilant CoG by region

| region     |    n |   cog_mean_hz |   cog_median_hz |   cog_z_mean |   cog_z_sem |
|:-----------|-----:|--------------:|----------------:|-------------:|------------:|
| chattogram | 6155 |       5000.63 |         5167.82 |            0 |        0.01 |
| dhaka      | 6121 |       5418.8  |         5472.08 |            0 |        0.01 |
| faridpur   | 7050 |       5555.66 |         5508.89 |            0 |        0.01 |
| khulna     | 6025 |       5635.3  |         5711.48 |           -0 |        0.01 |
| mymensingh | 5139 |       5812.01 |         5763.34 |           -0 |        0.01 |
| rajshahi   | 5787 |       5396.25 |         5410.73 |           -0 |        0.01 |
| rangpur    | 5112 |       5602.42 |         5660.33 |           -0 |        0.01 |
| sylhet     | 9670 |       5515.17 |         5461.25 |            0 |        0.01 |

- Sylhet CoG shift vs. corpus mean: **28.7 Hz** (0.029 SD units).
![Sibilant CoG density by region -- likely the most-quoted number in the paper.](figures/cog_density_by_region.png)

*Sibilant CoG density by region -- likely the most-quoted number in the paper.*

## VOT (word-initial four-way stops) -- PROVISIONAL, automatic only

74482/74590 candidates got an automatic VOT estimate. **Manual verification not yet done** -- `vot_manual_check_worklist.csv` (296 rows, ~37/region) needs a human to listen and correct `manual_vot_ms`. Report automatic-vs-manual agreement once reviewed; if poor, report VOT on the hand-checked subsample only.

| region     | phone   |    n |   mean_vot_ms |   sd_vot_ms |
|:-----------|:--------|-----:|--------------:|------------:|
| chattogram | D       |  137 |          4.32 |       16.08 |
| chattogram | Dh      |   36 |          2.5  |        5.48 |
| chattogram | T       |  209 |          7.56 |       20.93 |
| chattogram | Th      |   35 |         18.51 |       22.2  |
| chattogram | b       | 1880 |          2.48 |       10    |
| chattogram | bh      |  328 |          4.4  |       13.78 |
| chattogram | d       | 1077 |          2.53 |       10.16 |
| chattogram | dh      |  108 |          3.57 |       10.84 |
| chattogram | g       |  434 |          1.63 |        6.32 |
| chattogram | gh      |  126 |          2.05 |        7.55 |
| chattogram | k       | 1645 |          6.77 |       16.29 |
| chattogram | kh      |   89 |         15.39 |       22.6  |
| chattogram | p       | 1580 |          3.39 |       12.01 |
| chattogram | ph      |  204 |         26.79 |       45.17 |
| chattogram | t       |  764 |          3.23 |       11    |
| chattogram | th      |  207 |         14.86 |       25.88 |
| dhaka      | D       |  137 |          2.47 |        9.51 |
| dhaka      | Dh      |   35 |          2.57 |        7.78 |
| dhaka      | T       |  209 |          7.18 |       26.36 |
| dhaka      | Th      |   33 |         10.61 |       13.2  |
| dhaka      | b       | 1868 |          1.39 |        6.82 |
| dhaka      | bh      |  328 |          3.12 |        9.74 |
| dhaka      | d       | 1071 |          1.26 |        7.54 |
| dhaka      | dh      |  108 |          2    |        8.72 |
| dhaka      | g       |  430 |          1.21 |        4.73 |
| dhaka      | gh      |  126 |          4.16 |       11.07 |
| dhaka      | k       | 1632 |          5.76 |       16.9  |
| dhaka      | kh      |   89 |         11.55 |       21.15 |
| dhaka      | p       | 1574 |          3.26 |       10.01 |
| dhaka      | ph      |  201 |         16.89 |       30    |
| dhaka      | t       |  762 |          2.43 |       10.38 |
| dhaka      | th      |  208 |         13.04 |       25.08 |
| faridpur   | D       |  157 |          4.03 |       19.89 |
| faridpur   | Dh      |   42 |          3.81 |        8.26 |
| faridpur   | T       |  234 |          9.22 |       27.17 |
| faridpur   | Th      |   39 |         27.79 |       22.22 |
| faridpur   | b       | 2185 |          2.33 |        9.78 |
| faridpur   | bh      |  383 |          2.56 |        9.12 |
| faridpur   | d       | 1242 |          1.96 |        9.36 |
| faridpur   | dh      |  125 |          5.07 |       11.61 |

## For the paper

S7.5. The sibilant CoG result is likely the paper's most-quoted number.
