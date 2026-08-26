# Part D — Phonetic Accent Map (BPAC)

The section that turns a corpus description into a scientific contribution.
Part C (S6) shows an accent signal *exists* in the pooled embeddings; Part D
(S7) shows *what it is* -- phone durations, rhythm, vowel space, sibilants,
VOT, and intonation, all measured off forced-aligned phones and compared
across the 8 regions because every speaker read the same 250 sentences.

All of Part D is CPU-bound. No GPU required.

```
Part_D/
├── config.yml
├── bpac_phon/                      (settings, manifest, textnorm, reporting, cli, audio)
├── D0_lexicon/analyze.py           -> reports/   (T11 -- start during Phase 1, long lead time)
├── D1_alignment/analyze.py         -> reports/   (T12 -- MFA train + force-align)
├── D2_duration_rhythm/analyze.py   -> reports/   (T13-T14 -- alignment confidence, phone duration, rhythm)
├── D3_vowel_space/analyze.py       -> reports/   (T15 -- Lobanov-normalized F1/F2)
├── D4_consonants/analyze.py        -> reports/   (T16 -- sibilant spectral moments + VOT)
├── D5_intonation/analyze.py        -> reports/   (T17 -- averaged F0 contours per sentence per region)
├── D6_phonetic_distance/analyze.py -> reports/   (T18 -- fused 8x8 phonetic distance matrix + dendrogram)
├── D7_convergence/analyze.py       -> reports/   (T19 -- Mantel test vs. Part C's embedding distance matrix)
├── cache/
├── requirements.txt, pyproject.toml, run_all.sh
```

## Status

Scaffolding only (T1). D0-D7 `analyze.py` scripts do not exist yet -- they
land as T11-T19 are implemented, in the order above (D0 and D1 are
prerequisites for everything else). `run_all.sh` skips any step whose
`analyze.py` is missing, so it stays runnable throughout this phase.

## Invariants

1. Every task here reads `data_root` (`data_clean/`, 16 kHz) **except**
   D4/T16 sibilant analysis, which reads `raw_data_root` (raw 44.1/48 kHz)
   resampled to `sibilant_analysis_sr` (32 kHz) -- at 16 kHz the Nyquist
   limit truncates the /s/ vs /sh/ spectral distinction the analysis exists
   to measure.
2. `pipeline` is perfectly confounded with `gender` (lab=male, app=female).
   Never claim to separate them.
3. `bpac_phon.manifest.build_manifest()` prefers `verified_manifest.csv`
   (Phase 1, T4) once it exists; until then it falls back to a plain
   directory walk of `data_root` so the package is usable pre-Phase-1.
