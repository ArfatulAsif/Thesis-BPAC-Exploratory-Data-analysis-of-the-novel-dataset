# T12 -- Forced Alignment: Parsed TextGrids

Parsed **12635/12747** utterances' TextGrids into 959881 phone tokens and 170033 word tokens. 112 utterances have no TextGrid (alignment failed or were skipped).

## Sanity checks

- Phones under 10ms: 22342 (2.328%) -- the aligner squeezing a phone it couldn't confidently place.
## Per-region alignment stats

| region     |   n_phones |   n_utterances |   mean_dur_s |   pct_short |
|:-----------|-----------:|---------------:|-------------:|------------:|
| chattogram |     114190 |           1495 |       0.08   |      2.3636 |
| dhaka      |     113603 |           1491 |       0.0768 |      2.389  |
| faridpur   |     132447 |           1740 |       0.0761 |      2.3119 |
| khulna     |     113724 |           1489 |       0.0783 |      2.3539 |
| mymensingh |      95179 |           1246 |       0.0816 |      2.323  |
| rajshahi   |     113374 |           1489 |       0.0701 |      2.4027 |
| rangpur    |      94990 |           1243 |       0.0768 |      2.3497 |
| sylhet     |     182374 |           2442 |       0.0727 |      2.2059 |
