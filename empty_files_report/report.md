# Empty / dead .wav files — inventory

Scanned **12750** files. Found **3** empty/dead (0.02%).

- In the **scored** set (conforming name + valid sentence id): **3**  ← these affected B1/B2
- Non-conforming / not scored: **0**

## Reasons

| reason   |   count |
|:---------|--------:|
| silent   |       3 |

## By region

| region     |   n_files |   n_empty |   n_empty_scored |   n_empty_nonconforming |   pct_empty |
|:-----------|----------:|----------:|-----------------:|------------------------:|------------:|
| mymensingh |      1250 |         2 |                2 |                       0 |        0.16 |
| faridpur   |      1750 |         1 |                1 |                       0 |        0.06 |
| dhaka      |      1500 |         0 |                0 |                       0 |        0    |
| chattogram |      1500 |         0 |                0 |                       0 |        0    |
| khulna     |      1500 |         0 |                0 |                       0 |        0    |
| rajshahi   |      1500 |         0 |                0 |                       0 |        0    |
| rangpur    |      1250 |         0 |                0 |                       0 |        0    |
| sylhet     |      2500 |         0 |                0 |                       0 |        0    |

## What to do
- `empty_files.csv` is the actionable list (scored empties first).
- **Empties in the scored set** are the ones that hurt your B1/B2 numbers (blank hypothesis vs a real reference = ~100% error for that clip). Either re-record them, or exclude them and re-run B1/B2.
- `delete_empty_files.sh` / `.ps1` contain ready removal commands for the **scored** empties only — **review them before running**; nothing was deleted.
- Non-conforming empties were never scored, but their count per region tells you which recording sessions produced the most failed captures.
