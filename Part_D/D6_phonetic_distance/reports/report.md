# T18 -- Fused Phonetic Distance Matrix

Fused 7 feature blocks (phone_duration, rhythm, vowel_formants, vowel_space_area, pillai, sibilant_cog, f0) -> 74 standardized features across 8 regions. Cophenetic correlation of the Ward dendrogram: **0.6826**.

## Feature blocks used

- `phone_duration`: 40 feature(s)
- `rhythm`: 7 feature(s)
- `vowel_formants`: 16 feature(s)
- `vowel_space_area`: 1 feature(s)
- `pillai`: 4 feature(s)
- `sibilant_cog`: 1 feature(s)
- `f0`: 5 feature(s)
## Branch support (bootstrap over features)

| split                                                               |   support_pct |
|:--------------------------------------------------------------------|--------------:|
| mymensingh,faridpur,dhaka,chattogram,khulna,rajshahi,rangpur,sylhet |         100   |
| dhaka,khulna                                                        |          82.9 |
| mymensingh,rajshahi                                                 |          67.9 |
| faridpur,chattogram,rangpur,sylhet                                  |          57.7 |
| mymensingh,dhaka,khulna,rajshahi                                    |          54   |
| faridpur,rangpur                                                    |          38.3 |
| faridpur,chattogram,rangpur                                         |          26.4 |

## Feature-block sensitivity

| dropped_block    |   corr_with_full_matrix |
|:-----------------|------------------------:|
| phone_duration   |                  0.511  |
| rhythm           |                  0.98   |
| vowel_formants   |                  0.9348 |
| vowel_space_area |                  0.9992 |
| pillai           |                  0.9956 |
| sibilant_cog     |                  1      |
| f0               |                  0.9911 |

## PCA

PC1 explains 34.0% of variance. Top-loading features:

| feature   |    PC1 |    PC2 |    PC3 |
|:----------|-------:|-------:|-------:|
| pctV      |  0.194 |  0.048 |  0.006 |
| O         |  0.192 |  0.042 | -0.065 |
| y         |  0.187 |  0.047 | -0.022 |
| a         |  0.185 |  0.044 | -0.04  |
| k         | -0.184 | -0.006 |  0.103 |
| p         | -0.183 | -0.059 |  0.024 |
| g         | -0.182 |  0.058 |  0.011 |
| ng        | -0.18  | -0.005 | -0.092 |
| Dh        | -0.179 | -0.009 | -0.024 |
| t         | -0.175 | -0.082 | -0.012 |

![Ward dendrogram of the fused phonetic distance matrix.](figures/dendrogram_phonetic.png)

*Ward dendrogram of the fused phonetic distance matrix.*

![PCA of the region x feature matrix.](figures/pca_biplot.png)

*PCA of the region x feature matrix.*

## For the paper

S7.7.
