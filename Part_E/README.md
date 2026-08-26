# Part E — Validation (BPAC)

Signal-processing validation only. **There is no human listening study
anywhere in this repo.** Regional-origin and accent labels are validated at
recruitment (documented in the paper's S3), not by a listening evaluation --
no task here scores generated audio, and no model consumes listener data.

```
Part_E/
├── config.yml
├── bpac_valid/                (settings, manifest, textnorm, reporting, cli, audio)
├── E1_denoise_check/analyze.py -> reports/   (T20 -- denoising sanity check)
├── requirements.txt, pyproject.toml, run_all.sh
```

## Status

Scaffolding only (T1). `E1_denoise_check/analyze.py` lands as T20 is
implemented, and depends on Part_D's D4/T16 sibilant-token output
(`sibilant_tokens.parquet`). `run_all.sh` skips E1 until then.

## What E1 will show

The thesis trained voice-cloning models on a denoised variant of BPAC; this
paper deliberately does not denoise. E1 demonstrates why: spectral-gating
denoisers suppress broadband noise-like energy, which is exactly what
fricatives and aspiration bursts are made of -- so denoising would attenuate
the very cues Part D's phonetic measurements (T16 sibilant CoG, VOT) depend
on. One paired comparison (denoised vs. `data_clean/`) settles the "why
didn't you denoise?" question for reviewers.
