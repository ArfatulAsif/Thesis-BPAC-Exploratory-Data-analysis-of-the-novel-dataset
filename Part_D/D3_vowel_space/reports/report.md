# T15 -- Vowel Space (Lobanov-Normalized F1/F2)

203425/205714 vowel tokens retained (98.89%) after duration/adjacency/formant-range filtering. Lobanov-normalized per speaker to remove vocal-tract-length differences while preserving dialect differences.

## Retention by region

| region     |   stage_candidate |   stage_retained |   retention_pct |
|:-----------|------------------:|-----------------:|----------------:|
| chattogram |             25397 |            25029 |           98.55 |
| dhaka      |             25524 |            25342 |           99.29 |
| faridpur   |             28351 |            28125 |           99.2  |
| khulna     |             25188 |            24845 |           98.64 |
| mymensingh |             21976 |            21800 |           99.2  |
| rajshahi   |             23278 |            22924 |           98.48 |
| rangpur    |             19808 |            19590 |           98.9  |
| sylhet     |             36192 |            35770 |           98.83 |

## Vowel space area by region

| region     |   vowel_space_area |   n_vowel_types |
|:-----------|-------------------:|----------------:|
| sylhet     |              2.636 |               8 |
| mymensingh |              2.616 |               8 |
| faridpur   |              2.388 |               8 |
| rajshahi   |              2.321 |               8 |
| rangpur    |              2.246 |               8 |
| chattogram |              2.083 |               8 |
| dhaka      |              1.943 |               8 |
| khulna     |              1.924 |               8 |

## Pillai scores (near-vowel-pair distinctness; near 0 = merged)

| region     | pair   |   pillai |
|:-----------|:-------|---------:|
| chattogram | e-i    |   0.3246 |
| chattogram | o-u    |   0.1836 |
| chattogram | O-o    |   0.0395 |
| chattogram | a-O    |   0.3013 |
| dhaka      | e-i    |   0.2987 |
| dhaka      | o-u    |   0.2015 |
| dhaka      | O-o    |   0.0284 |
| dhaka      | a-O    |   0.2974 |
| faridpur   | e-i    |   0.2255 |
| faridpur   | o-u    |   0.1924 |
| faridpur   | O-o    |   0.0707 |
| faridpur   | a-O    |   0.4543 |
| khulna     | e-i    |   0.2444 |
| khulna     | o-u    |   0.1988 |
| khulna     | O-o    |   0.0258 |
| khulna     | a-O    |   0.2954 |
| mymensingh | e-i    |   0.2988 |
| mymensingh | o-u    |   0.1896 |
| mymensingh | O-o    |   0.0631 |
| mymensingh | a-O    |   0.5327 |
| rajshahi   | e-i    |   0.2356 |
| rajshahi   | o-u    |   0.1665 |
| rajshahi   | O-o    |   0.0635 |
| rajshahi   | a-O    |   0.4437 |
| rangpur    | e-i    |   0.2594 |
| rangpur    | o-u    |   0.2152 |
| rangpur    | O-o    |   0.0407 |
| rangpur    | a-O    |   0.3913 |
| sylhet     | e-i    |   0.403  |
| sylhet     | o-u    |   0.2096 |
| sylhet     | O-o    |   0.0469 |
| sylhet     | a-O    |   0.4518 |

## Figures

![8-region vowel quadrilateral overlay -- main paper figure.](figures/vowel_quadrilateral_overlay.png)

*8-region vowel quadrilateral overlay -- main paper figure.*

![Vowel ellipses, small multiples by region.](figures/vowel_ellipses_by_region.png)

*Vowel ellipses, small multiples by region.*

## For the paper

S7.4. The overlaid quadrilateral is a main paper figure.
