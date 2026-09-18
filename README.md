# BPAC: How Detectable Is Regional Accent in Read Bangla?

**A Parallel Corpus and Its Phonetic Structure**

<div align="center">

[![python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.8.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
[![cudnn](https://img.shields.io/badge/cuDNN-8.9.7-orange?logo=nvidia&logoColor=white)](https://developer.nvidia.com/rdp/cudnn-archive) <br>
[![dataset](https://img.shields.io/badge/🤗%20Dataset-BPAC-yellow)](https://huggingface.co/datasets/ArfatulAsif/BPAC)
[![doi](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F10299-blue)](https://doi.org/10.57967/hf/10299)
[![license](https://img.shields.io/badge/Code-MIT-green.svg?labelColor=gray)](LICENSE)
[![data license](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)



</div>

This repository contains the corpus construction, verification, and complete
analysis pipeline behind **BPAC** (the *Bangla Parallel Accent Corpus*) — 51
speakers from 8 regions of Bangladesh, each reading an identical 250-sentence
script, and the five-part pipeline that measures how much regional accent
survives that design.

Every table, figure, and statistic in the companion manuscript traces back to a
specific script and `reports/` output in this repo. See
[Where every number comes from](#where-every-number-comes-from).

---

## Results at a glance

<br>
<br>


| Question | Answer | Source |
|---|---|---|
| Does region change ASR error on identical text? | Yes, consistently across two architectures. Sylhet worst (CER 0.0931 / 0.0840), Mymensingh best (0.0675 / 0.0604). | `Part_B/*/reports/` |
| Is region classifiable on **unseen speakers**? | Pooled: 0.2233 accuracy, 1.69× stratified chance — **not significant** (permutation *p* = 0.386). | `Part_C/C3_LOSO_within_gender/` |
| …within the lab-recorded half? | 0.3111, 2.19× chance, *p* = 0.0099 (floor of a 100-permutation test). Carried by 2 of 8 regions. | `Part_C/C3_LOSO_within_gender/` |
| …within the app-recorded half? | 0.1153 — **at or below chance** (0.91×). | `Part_C/C3_LOSO_within_gender/` |
| Is it speaker-identity leakage? | No. ECAPA speaker embeddings reach only 1.28× vs. WavLM's 1.69× on the same pooled speakers. | `Part_C/C3_.../ecapa_control.json` |
| Is the famous sibilant shift real? | Not as a pooled spectral effect. Across 51,059 tokens the regions the source literature actually names deviate *opposite* to the prediction. | `Part_D/D4_consonants/` |
| Which phonetic dimension organizes regions? | Dropping phone duration moves the fused matrix most (*r* = 0.511) — but confounded with block size and gender ratio. | `Part_D/D6_phonetic_distance/` |
| Do neural and phonetic views agree? | No. Mantel *r* = 0.138, *p* = 0.46. Neither tracks geography. | `Part_D/D7_convergence/` |


> **Read the nulls as findings, not shortfalls.** The corpus is built to make
> accent detection as clean as possible; a subtle, condition-dependent result
> from read, content-controlled speech is a *conservative lower bound*, not
> evidence that accent is absent.

<br>
<br>

---

<br>
<br>

## About this research

Bangla regional accent has been described dialectologically for over a century,
but almost entirely through impressionistic transcription rather than acoustic
measurement — and almost never under conditions that let speakers from different
regions be compared on identical content. This project asks the question
directly: **is regional accent in read Bangla actually detectable from the
acoustic signal — and if so, what is it made of?**

To answer it we built BPAC: 51 native speakers across 8 regions, each reading an
identical, diversity-optimized 250-sentence script, so that **region and speaker
vary while linguistic content is held fixed by design**. That parallel structure
is what lets an acoustic difference be attributed to *accent* rather than to
*what was said*, and is — to our knowledge — the first time a Bangla speech
corpus has been built this way specifically for accent research.

Detectability is then measured two independent ways, chosen because they fail
differently:

<br>

1. **Self-supervised neural embeddings** (WavLM-Large, middle layer) evaluated
   by strict **leave-one-speaker-out** region classification, with speaker-level
   permutation testing and a speaker-verification-embedding control. A random
   train/test split would measure voice memorization at least as much as accent;
   LOSO is the only fair test.
<br>

3. **Classical acoustic-phonetic dialectometry** — phone duration, rhythm
   typology, Lobanov-normalized vowel space, sibilant spectral center of
   gravity, and intonation — measured off Montreal Forced Aligner boundaries
   from an acoustic model trained on this corpus and validated with a
   held-out-region alignment check.

The two resulting 8×8 region-distance matrices are then compared to each other
and to geography.

---

<br>
<br>

## The corpus

**Script construction.** Text was sourced from twelve topical domains (national
affairs, business, education, entertainment, health, international, literature,
news, politics, religion, science, sports), segmented and normalized. 100
candidate 400-sentence subsets were generated and scored for lexical diversity;
the highest-scoring were manually reviewed for naturalness, grammaticality and
structural variance; one was curated down to the final **250-sentence** script
read by every speaker. Full methodology: `Dataset collection/Dataset Part.pdf`.

<br>

| Script property | Value | | Script property | Value |
|---|---|---|---|---|
| Sentences | 250 | | Distinct characters | 81 |
| Word tokens | 3,385 | | Grapheme clusters | 10,844 |
| Unique word types | 2,012 | | Unicode code points | 19,113 (non-space) |
| Type–token ratio | 0.594 | | Vowel:consonant ratio | 0.578 |
| Hapax legomena | 1,485 | | Sentence length | 13.5 words (2–122) |


<br>


**Recruitment and recording.** 51 native speakers (26 male, 25 female), aged
19–25, across Sylhet, Chattogram, Dhaka, Faridpur, Khulna, Mymensingh, Rajshahi
and Rangpur. Regional origin and native-accent status were established at
recruitment from self-reported district of origin — there is no post-hoc
listening study anywhere in this pipeline. Male participants were recorded in a
sound-treated room at the SUST Audio Laboratory; female participants recorded
remotely through a purpose-built Android app (*SUST Kotha*) after online
training on minimizing background noise.

<br>

> ⚠️ **The recording pipeline is perfectly confounded with speaker gender**
> (lab = male, app = female). Every Part treats this as a hard structural
> constraint, never as something to explain away: no analysis here claims to
> separate recording condition from gender, and the central detectability
> experiments are re-run *within* each subset to hold the confound constant.

<br>

| Property | Value |
|---|---|
| Regions | 8 — Sylhet, Faridpur, Chattogram, Dhaka, Khulna, Rajshahi, Mymensingh, Rangpur |
| Speakers | 51 (26 male, 25 female) |
| Speakers per region | Sylhet 10 · Faridpur 7 · Chattogram/Dhaka/Khulna/Rajshahi 6 · Mymensingh/Rangpur 5 |
| Sentences per speaker | 250 — identical script for all (fully parallel, zero missing cells) |
| Total utterances | 12,750 raw → **12,747** after dropping 3 empty recordings |
| Total audio | 27 h 44 m (27.74 h); mean utterance 7.83 s |
| Recording formats | Lab: 44.1/48 kHz, stereo, 16–24 bit · App: 44.1 kHz, mono, 16 bit |
| Working corpus | 16 kHz mono, peak-normalized to −1 dBFS |

<br>
<br>

**Manifest integrity.** [`verified_manifest.csv`](verified_manifest.csv) is the
corrected, fully-reviewed audio↔sentence mapping and the source of truth for
every downstream Part. An internal audit (`Part_B/alignment_verification/`)
CER-scored every utterance against all 250 candidate sentences — a 12,747 × 250
matrix, >3.18 M comparisons — and found exactly one systematic error: a
29-utterance sentence-numbering rotation for a single Rajshahi speaker, which is
corrected in place (`corrected` / `original_sentence_id` columns record what
changed). 99.773% of utterances required no correction. As shipped the manifest
has **0 excluded** and **0 needs-review** rows.

---

<br>
<br>

## Privacy and anonymization

Informed consent (signed paper form for lab participants, equivalent online form
for app participants — blank templates in `Dataset collection/`) explicitly
authorized public release of the recordings together with **district/region,
gender and age**, while **withholding participant names and contact details**.

<br>
<br>

This repository is published accordingly:

- **Speaker identifiers are pseudonymous.** Every speaker is referenced only as
  `<region>_<gender>_<NN>` (e.g. `sylhet_male_04`), where `NN` is a sequential
  index within that region-and-gender group. These codes are collision-free and
  carry no personal information.
- **No name-bearing column ships.** The `speaker_name` column the manifest code
  derives locally from folder names is **removed** from every released CSV and
  Parquet file. The schema docs in `common/T2_manifest_schema/reports/` note
  this explicitly.
- **Utterance IDs are derived from the pseudonymous codes**
  (`<region>_<gender>_<NN>_<sentence>`), including inside the cached embedding
  archives (`Part_C/cache/*.npz`) and all Part D token tables.
- **Forced-alignment artifacts are not tracked.** The ~50,770 generated `.lab` /
  `.TextGrid` files are excluded via `.gitignore` (they are regenerable from
  `Part_D/D1_alignment/`).
- **Absolute local paths are normalized** to `<PROJECT_ROOT>` in all committed
  reports.

<br>
<br>

Anonymization is lossless for the science: speaker counts, region×gender
balance, fold structure and every reported statistic are unchanged. Region and
gender are retained deliberately — both are released under consent and both are
analysis variables.

---

<br>
<br>

## Repository layout

```
.
├── Part_A/              Corpus descriptive EDA (bpac_eda) — A1–A8
├── Part_B/              ASR benchmark + alignment verification (bpac_stt)
├── Part_C/              Accent structure / embedding separability (bpac_accent) — C0–C4
├── Part_D/              Phonetic accent map (bpac_phon) — D0–D7
├── Part_E/              Signal-processing validation (bpac_valid) — E1
├── common/              Shared manifest / provenance / consistency-check code
├── Dataset collection/  Blank consent forms + dataset-construction methodology (PDFs)
├── Literature Review/   Source PDFs reviewed for the manuscript
├── release/             Figure-collection helper for the manuscript
├── verified_manifest.csv   Corrected, fully-reviewed utterance manifest (source of truth)
├── audio_quality_report/   Standalone per-file audio QA output
├── empty_files_report/     Dead/empty-recording audit
├── check_audio_quality.py, mark_empty_files.py   Standalone audio-QA utilities
├── Makefile             make part_a .. part_e / make all / make clean
├── requirements-lock.txt   Exact pinned versions used for the published numbers
└── CITATION.cff, LICENSE   Code citation metadata (MIT)
```

Each Part is a self-contained Python package with its own `config.yml`,
`requirements.txt`, `analyze.py` scripts, and `reports/` outputs
(`report.md` + `summary.json` + CSVs + figures per analysis) — and its own
`README.md` documenting its analyses, config knobs, and invariants.

`.gitignore` excludes the audio corpus, generated alignment artifacts, and paper
drafts. See `.gitignore` for the exact list.

---


<br>
<br>

## Where every number comes from

| Manuscript content | Produced by | Report artifact |
|---|---|---|
| Corpus composition, duration, parallel coverage | `Part_A/A1`–`A3` | `reports/report.md`, `summary.json` |
| Script lexical/character statistics | `Part_A/A4`–`A5` | `reports/report.md` |
| Speaking rate (VAD-corrected), F0 by region×gender | `Part_A/A6`–`A7` | `reports/report.md` |
| WADA-style SNR audit, SNR-vs-WER null | `Part_A/A8` | `reports/report.md` |
| Per-region WER/CER, both ASR models | `Part_B/wav2vec2_stt`, `whisper_stt` | `reports/report.md`, `region_metrics.csv` |
| Alignment verification rate, the Rajshahi rotation | `Part_B/alignment_verification` | `reports/report.md`, `verified_manifest.csv` |
| Region×region accent distance, dendrogram | `Part_C/C1`, `C4` | `distance_matrix.csv`, `bootstrap CIs`, Mantel |
| LOSO accuracy, permutation tests, ECAPA control | `Part_C/C3` | `summary.json`, `permutation_test_*.json`, `ecapa_control.json` |
| Lexicon coverage, phone set | `Part_D/D0` | `coverage.json`, `bpac_lexicon.txt` |
| Alignment + held-out-region robustness check | `Part_D/D1` | `heldout_region_summary.json`, `phone_tokens.parquet` |
| Phone duration, rhythm typology (%V, ΔC, nPVI) | `Part_D/D2` | `rhythm_metrics_by_region.csv`, `phone_duration_mixedmodel.csv` |
| Vowel space area, Pillai scores | `Part_D/D3` | `reports/report.md`, `vowel_tokens.parquet` |
| Sibilant CoG by region (51,059 tokens) | `Part_D/D4` | `sibilant_cog_by_region.csv` |
| Region-average F0 contours | `Part_D/D5` | `f0_contours_normalized.parquet` |
| Fused 8×8 phonetic matrix, block sensitivity, PCA | `Part_D/D6` | `reports/report.md` |
| Neural↔phonetic convergence, geography tests | `Part_D/D7` | `mantel_results.json`, `geographic_agreement.json` |
| Denoising sanity check | `Part_E/E1` | `summary.json` |

---

<br>
<br>

## Getting the data

This repository is the **code**, not the corpus. Audio is intentionally not
committed. A fresh clone needs it supplied separately:

- **Released, anonymized audio** (recommended) — the published dataset:
  🤗 **[huggingface.co/datasets/ArfatulAsif/BPAC](https://huggingface.co/datasets/ArfatulAsif/BPAC)**
  · DOI [10.57967/hf/10299](https://doi.org/10.57967/hf/10299) · CC BY-NC 4.0.
  It ships both a 16 kHz mono working corpus and the original-rate audio:

  ```
  BPAC/
  ├── data/              16 kHz mono working corpus (peak −1 dBFS)
  │   ├── metadata.csv
  │   └── <region>/
  │       ├── chattogram_female_001_001.wav
  │       ├── chattogram_female_001_002.wav
  │       └── ...
  └── data_original/     original audio at native sample rate / channels
      ├── metadata.csv
      └── <region>/...
  ```

<br>
<br>

  The Hugging Face release derives its own anonymous speaker sequence, so its
  IDs are **not** guaranteed to correspond one-to-one with the
  `<region>_<gender>_<NN>` codes in this repository. Join on the dataset's own
  `metadata.csv` when working from the released audio, and on
  [`verified_manifest.csv`](verified_manifest.csv) when working from this
  codebase.

- **Your own recordings** — place them beside `Part_A/`, `Part_B/`, … in the
  layout each Part's `config.yml` documents:

  ```
  data/
  ├── BPAC/
  │   └── <region>/
  │       └── speaker_<gender>_<NN>/
  │           ├── sound_001.wav
  │           ├── ...
  │           └── sound_250.wav
  └── best_diverse_sentences.csv      the 250-sentence script (id,sentence)
  ```

  `speaker_<gender>_<NN>` is unique **within** a region; the region directory
  supplies the rest of the identity. The normalized 16 kHz copy used by Parts
  C–E is generated for you by `Part_C/C0_Normalize_corpus/analyze.py`.

---

<br>
<br>


## Environment

Python 3.9 (the published numbers were produced on CPython 3.9, Windows).
Each Part pins its own dependencies; `requirements-lock.txt` at the repo root
records the exact versions used for the published Part A/B/C results.

<br>
<br>

```bash
git clone https://github.com/ArfatulAsif/Thesis-BPAC-Exploratory-Data-analysis-of-the-novel-dataset.git
cd Thesis-BPAC-Exploratory-Data-analysis-of-the-novel-dataset
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

A CUDA GPU is strongly recommended for Part B (two ASR models over 12,747
utterances) and Part C (WavLM + ECAPA embedding extraction). Part D's forced
alignment requires the **Montreal Forced Aligner**, installed separately via
conda.

---

<br>
<br>
<br>
<br>

## Running the pipeline

Every `analyze.py` reads its paths and hyperparameters from that Part's
`config.yml` and takes only `--config` — **edit `config.yml`, don't pass ad-hoc
flags.** Run analyses in the order shown; later steps depend on earlier output.

<br>
<br>

### Part A — corpus descriptive EDA

```bash
cd Part_A && pip install -r requirements.txt

python A1_unique_sentences/analyze.py       # sentence/utterance coverage per speaker
python A2_audio_length/analyze.py           # audio duration per region/speaker/gender
python A3_utterance_count/analyze.py        # utterance counts per region/speaker
python A4_character_count/analyze.py        # character-frequency profile of the script
python A5_word_count/analyze.py             # word-frequency / Zipf profile of the script
python A6_speaking_rate_pitch/analyze.py    # speaking rate (WPM) + F0 by region (slow — reads every file)
python A7_vad_speaking_rate/analyze.py      # VAD-based speaking rate (replaces energy-threshold trimming)
python A8_audio_quality_snr/analyze.py      # WADA-style SNR estimate + SNR-vs-WER correlation
cd ..
```

<br>
<br>

### Part B — ASR benchmark + alignment verification

```bash
cd Part_B && pip install -r requirements.txt

python wav2vec2_stt/analyze.py              # fast CTC model — run first
python whisper_stt/analyze.py               # slower encoder-decoder model
python create_report.py                     # combined cross-model WER/CER report

# Alignment verification (produces verified_manifest.csv at the repo root):
python alignment_verification/analyze.py                            # CER-scores every utterance against all 250 sentences
python alignment_verification/spotcheck.py --reviewer <your-name>   # interactive: listen + decide on flagged utterances
python alignment_verification/resolve.py                            # applies decisions -> ../verified_manifest.csv
cd ..
```

<br>
<br>

### Part C — accent structure

```bash
cd Part_C && pip install -r requirements.txt
pip install umap-learn                      # optional (UMAP; t-SNE works without it)

python C0_Normalize_corpus/analyze.py                             # builds the normalized 16 kHz corpus (run first)
python C1_Region_accent_distance_matrix/analyze.py                # WavLM region-distance matrix + dendrogram
python C2_Speaker_vs_accent_embedding_separability/analyze.py     # speaker-vs-accent separability + LOSO
python C3_LOSO_within_gender/analyze.py                           # within-gender LOSO + permutation + ECAPA control
python C4_distance_statistics/analyze.py                          # bootstrap CIs + Mantel test on C1's distances
cd ..
```

<br>
<br>

### Part D — phonetic accent map

```bash
cd Part_D && pip install -r requirements.txt

python D0_lexicon/analyze.py                          # builds the Bangla pronunciation lexicon

# D1 forced alignment — Montreal Forced Aligner, installed separately via conda:
python D1_alignment/prepare_corpus.py                 # converts the corpus into MFA's layout (speaker/utt.wav + .lab)
conda run -n aligner mfa train D1_alignment/mfa_corpus \
    D0_lexicon/reports/bpac_lexicon.txt \
    D1_alignment/reports/bpac_mfa_acoustic_model.zip \
    --output_directory D1_alignment/reports/textgrids --clean
python D1_alignment/finalize.py                       # TextGrids -> phone_tokens.parquet / word_tokens.parquet

# Held-out-region robustness check — confirms the aligner learned general Bangla
# phonetics rather than memorizing dialect-specific pronunciation:
python D1_alignment/prepare_heldout_split.py
conda run -n aligner mfa train D1_alignment/mfa_corpus_no_sylhet \
    D0_lexicon/reports/bpac_lexicon.txt \
    D1_alignment/reports/bpac_mfa_acoustic_model_no_sylhet.zip \
    --output_directory D1_alignment/reports/textgrids_no_sylhet --clean
conda run -n aligner mfa align D1_alignment/mfa_corpus_sylhet_only \
    D0_lexicon/reports/bpac_lexicon.txt \
    D1_alignment/reports/bpac_mfa_acoustic_model_no_sylhet.zip \
    D1_alignment/reports/textgrids_sylhet_heldout --clean
python D1_alignment/heldout_align_and_compare.py      # compares held-out vs. full-model phone boundaries

python D2_duration_rhythm/analyze.py       # phone duration + rhythm typology (%V, ΔC, nPVI)
python D3_vowel_space/analyze.py           # Lobanov-normalized F1/F2 vowel space
python D4_consonants/analyze.py            # sibilant CoG + VOT (reads full-bandwidth raw audio, not the 16 kHz copy)
python D5_intonation/analyze.py            # averaged F0 contours per sentence per region
python D6_phonetic_distance/analyze.py     # fuses D2–D5 into an 8×8 phonetic distance matrix
python D7_convergence/analyze.py           # Mantel test: phonetic (D6) vs. neural-embedding (C1) distance
cd ..
```

<br>
<br>


### Part E — validation

```bash
cd Part_E && pip install -r requirements.txt

python E1_denoise_check/analyze.py          # denoising sanity check (depends on D4's sibilant tokens)
cd ..
```

<br>
<br>

### Running several at once

```bash
make all          # runs each Part's run_all.sh in order (A -> B -> C -> D -> E)
make clean        # removes reports/ and __pycache__ output (the audio corpus is untouched)
```

> Each Part's `run_all.sh` chains only its **core** `analyze.py` steps — Part A's
> covers A1–A6 (not A7/A8), Part C's covers C0–C2 (not C3/C4), Part B's covers
> the two ASR models and combined report (not alignment verification). For the
> full set, run the individual commands above. `make check` is a placeholder and
> is not yet implemented.

---


<br>
<br>


## Reproducibility notes and known caveats

These are documented in the Parts' own READMEs and reports; they are collected
here so they are visible before anyone re-uses a number.

- **Recording condition is inseparable from gender.** Nothing in this pipeline
  can attribute an effect to one rather than the other. Subset results are
  reported for both halves, always.
- **Permutation tests use 100 permutations**, not 1,000 — LDA refit is the
  bottleneck. The exact (Phipson–Smyth) estimator therefore floors at
  *p* = 0.0099. Permutation runs use a reduced representation (PCA-20, capped at
  60 utterances/speaker), so their observed accuracies are on a different scale
  from the headline PCA-50 figures and must be read against their own nulls.
- **Speaker-resampling intervals fold duplicate draws**, so each replicate is a
  random ~63% subsample rather than a true bootstrap. The intervals are
  conservative and are stability checks, not calibrated CIs.
- **PCA and standardization are fitted once** on the full utterance matrix
  before the LOSO loop rather than inside each fold. The projection is
  unsupervised — no region label crosses the fold boundary — but this is a
  departure from fully nested cross-validation.
- **Per-phone duration mixed models do not all converge**; no per-phone regional
  duration effect survives FDR correction. Duration deviations in the reports
  are descriptive, not significance-tested.
- **The leave-one-block-out hierarchy is confounded with block size** (duration
  40 features vs. sibilant 1) and with regional gender ratio, since Part D
  features are gender-pooled.
- **VOT is provisional.** Automatic estimates exist for 74,482 tokens but the
  296-token manual verification worklist is not complete; VOT is excluded from
  every claim.
- **The corpus is deliberately not denoised** — spectral gating suppresses the
  same broadband energy that the sibilant and stop measurements depend on. See
  `Part_E/E1_denoise_check/`.

---

<br>
<br>

## Dataset availability

🤗 **[huggingface.co/datasets/ArfatulAsif/BPAC](https://huggingface.co/datasets/ArfatulAsif/BPAC)**
— DOI [10.57967/hf/10299](https://doi.org/10.57967/hf/10299), released under
**CC BY-NC 4.0** (non-commercial use with attribution). The release comprises
the original recordings at their native formats, a 16 kHz mono working corpus,
per-utterance metadata, and the 250-sentence script.


<br>
<br>

## Citation

The companion manuscript is not yet published — **cite the dataset directly
until then**:

```bibtex
@dataset{bpac2026,
  title        = {BPAC: Bangla Parallel Accent Corpus},
  author       = {Asif, Arfatul Islam and Khan, Unayes Ahmed and
                  Joy, Sanjoy Das and Rahman, Mohammad Shahidur},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/ArfatulAsif/BPAC}},
  doi          = {10.57967/hf/10299},
  note         = {Dataset accompanying a manuscript in preparation, "BPAC: How
                  Detectable Is Regional Accent in Read Bangla? A Parallel
                  Corpus and Its Phonetic Structure."}
}
```

For the analysis code specifically, see [`CITATION.cff`](CITATION.cff). Both
will be updated to the final paper citation once the manuscript has a venue and
DOI.

<br>
<br>

## License

| Component | License |
|---|---|
| Code in this repository | **MIT** — see [`LICENSE`](LICENSE) |
| The BPAC corpus | **CC BY-NC 4.0** — terms on the Hugging Face dataset card |

## Acknowledgements

We thank all participants in this study, and Towhidul Islam Tayef for
assistance with audio recording at the SUST Audio Laboratory. Conducted at the
Department of Computer Science and Engineering, Shahjalal University of Science
and Technology (SUST), Sylhet.
