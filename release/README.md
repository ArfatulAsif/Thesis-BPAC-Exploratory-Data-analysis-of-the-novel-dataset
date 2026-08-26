# release/

Dataset packaging scripts and metadata for the public BPAC release.

Scaffolding only (T1) — packaging scripts land alongside T21 (publication
figure regeneration) and the later release tasks. This directory will hold:

- a packaging script that assembles the release tarball/zip from
  `data_clean/`, `verified_manifest.csv` (T4), `checksums.csv` (T5),
  `bpac_lexicon.txt` (T11), the MFA acoustic model + TextGrids (T12), and
  `CORPUS_FROZEN.json` (T5)
- dataset metadata (datasheet / data statement, license, versioning)
- the Code Availability statement content referenced by `paper/main.tex`
