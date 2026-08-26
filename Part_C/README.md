# Part C v2 — Cleaned-Corpus Accent Analysis (BPAC)

A corrected, more rigorous Part C. It first **normalizes** the corpus (removing
the recording-format confound), then runs **improved** versions of C1 and C2 that
exploit the parallel design and a leave-one-speaker-out test to settle the
speaker-vs-accent question.

```
<project_root>/
├── Part_C2/
│   ├── config.yml
│   ├── bpac_accent/        (settings, manifest, normalize, embeddings, distances, separability, ...)
│   ├── C0_Normalize_corpus/analyze.py            -> reports/   (run FIRST)
│   ├── C1_Region_accent_distance_matrix/analyze.py -> reports/
│   ├── C2_Speaker_vs_accent_embedding_separability/analyze.py -> reports/
│   ├── cache/             (embedding cache, layer-tagged)
│   ├── requirements.txt, pyproject.toml, run_all.sh
├── data/                  (RAW corpus — input to C0)
└── data_clean/            (CLEAN corpus — written by C0, read by C1/C2)
```

## Why v2 exists

The first Part C ran on the raw corpus and showed (a) the tightest C1 cluster
coincided with the recording-format split, and (b) C2 separated by speaker, not
accent. Two causes: the **format confound** (44.1k/48k, stereo/mono) and
**method choices** that emphasized speaker over accent. v2 fixes both.

### What changed
1. **C0 normalization.** All clips → uniform **16 kHz mono**, empty/dead files
   dropped, optional peak-normalize, written to a clean tree. Removes the format
   fingerprint that on-the-fly resampling can't fully erase.
2. **Rajshahi/data diagnostics.** Per-region empty breakdown + a duration-based
   **parallel-alignment** check (flags regions whose `sound_NNN.wav` durations
   deviate per sentence — a proxy for misalignment).
3. **Middle-layer WavLM.** `layer: "middle"` — the most *phonetic* layer. The
   last layer drifts toward WavLM's pretraining objective (speaker/denoising),
   which suppressed accent and inflated speaker in v1.
4. **Parallel-aware C1 distance.** `ssl_controlled` subtracts the per-sentence
   mean (removes content) and averages over speaker pairs (removes speaker).
5. **Leave-one-speaker-out (LOSO) region classification** — the confound-proof
   test: can region be predicted on *unseen* speakers? If yes, accent is real
   and not speaker memorization.
6. **Per-sentence silhouette** — region separation for identical text.

## Models

| role | model | note |
|---|---|---|
| content / accent (C1, C2) | `microsoft/wavlm-large`, **middle layer**, mean-pooled | phonetic detail |
| speaker (C2) | `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN) | speaker identity |

## Setup & run

```bash
cd Part_C2
pip install -r requirements.txt
pip install umap-learn        # optional (UMAP; t-SNE works without it)

python C0_Normalize_corpus/analyze.py                 # writes ../data_clean
python C1_Region_accent_distance_matrix/analyze.py    # extracts WavLM (cached)
python C2_Speaker_vs_accent_embedding_separability/analyze.py
# or: ./run_all.sh
```

C0 must run first (C1/C2 read `../data_clean`). C1/C2 share the WavLM cache.

## Reading the results

**C0** — `per_region_summary.csv` (written vs dead-dropped per region),
`alignment_by_region.csv` (`pct_suspect`: higher = more duration-deviant
sentences → check `alignment_suspect_cells.csv` and spot-check those files).

**C1** — `distance_matrix.csv`, `distance_from_standard.csv`, heatmap,
dendrogram. With format now uniform, a surviving cluster is **dialectal**, not a
recording artifact. `method: loso_confusion` instead derives distance from how
often regions are confused on unseen speakers (and prints LOSO accuracy).

**C2 — the confound test.** Primary metric is **LOSO region accuracy** vs chance:
- ≥ 2× chance → accent generalizes across speakers ⇒ **genuine**.
- 1.3–2× chance → real but modest; report with the speaker-count caveat.
- ≈ chance → accent axis is weak vs speaker identity; state the confound honestly.

`silhouette_scores.csv` includes a `*_speaker_residualized` row. **Caveat:**
because every speaker is in one region (nested), removing the speaker mean also
removes the region mean, so a ≈0 residualized score is *expected by design* and
does **not** mean accent is absent — it quantifies the confound. LOSO is the
valid test. `per_sentence_region_silhouette` controls content (but still carries
the speaker confound).

## Config knobs

- `paths.raw_data_root` / `paths.data_root` (raw → clean).
- `normalize.*` (target_sr, mono, peak_normalize, min_ms, align_dur_tol, rajshahi_focus).
- `models.ssl.layer` ("middle" | int | -1), `models.ssl.pooling`.
- `c1_distance.method` (ssl_controlled | ssl_pooled | loso_confusion | mfcc_dtw),
  `per_sentence_center`, `loso_classifier`.
- `c2_separability` (residualize_speaker, loso_region_classifier,
  per_sentence_silhouette, projection, sample_per_speaker).

## Honest framing

If, on clean audio, LOSO is well above chance and a dialectal cluster survives,
report that the corpus supports accent conversion. If LOSO stays near chance,
that's still a legitimate finding for an accent-*conversion* paper: it shows
accent is a subtle axis entangled with speaker identity, which is precisely what
timbre-preserving conversion must disentangle — and motivates your method.
