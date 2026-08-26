# BPAC: How Detectable Is Regional Accent in Read Bangla?

**A Parallel Corpus and Its Phonetic Structure**

This repository contains the corpus construction, verification, and full
analysis pipeline behind BPAC (the **Bangla Parallel Accent Corpus**) and
its companion manuscript, *"BPAC: How Detectable Is Regional Accent in
Read Bangla? A Parallel Corpus and Its Phonetic Structure"* (manuscript in
preparation — see [Completed Paper/](Completed%20Paper/)).

## About this research

Bangla regional accent has been described dialectologically for over a
century, but almost entirely through impressionistic transcription rather
than acoustic measurement, and almost never under conditions that let
speakers from different regions be compared on identical content. This
project asks the question directly: **is regional accent in read Bangla
actually detectable from the acoustic signal — and if so, what is it made
of?**

To answer it, we built **BPAC**: 51 native speakers from 8 regions of
Bangladesh, each reading an identical, phonetically diverse 250-sentence
script — so that region and speaker vary while linguistic content is held
fixed by design. This parallel structure (every sentence, in every region)
is what makes the analysis possible, and is, to our knowledge, the first
time a Bangla speech corpus has been built this way specifically for
accent research (see the corpus comparison in the paper's §2.1).

We then measured detectability two independent ways:

1. **Neural / self-supervised embeddings** (WavLM) with leave-one-speaker-out
   (LOSO) region classification — the only fair test of whether a
   classifier has learned *accent* rather than memorized individual voices.
2. **Classical acoustic-phonetic dialectometry** — phone duration, rhythm
   typology, vowel formants and vowel-space geometry, sibilant spectral
   center of gravity, and intonation, all measured off Montreal-Forced-Aligner
   phone boundaries and validated with a held-out-region alignment check.

**Headline finding**: regional accent is real but subtle and unevenly
distributed. It is classifiable above chance on unseen speakers overall,
but the signal is concentrated almost entirely in the lab-recorded half of
the corpus and statistically indistinguishable from chance in the
app-recorded half. The dominant phonetic axis of regional variation turns
out to be consonant/rhythm timing, not the Sylheti sibilant shift often
cited anecdotally — which, traced back to its primary source, is not
acoustically measured or even localized to Sylhet in the literature that
is usually credited with it, and which we find to be a near-null effect
once measured directly. Full results, statistics, and discussion are in
[`Completed Paper/BPAC_paper.md`](Completed%20Paper/BPAC_paper.md) /
[`BPAC_paper.pdf`](Completed%20Paper/BPAC_paper.pdf).

## How the dataset was organized

BPAC's defining property is that it is **parallel**: every one of the 51
speakers read the *same* 250-sentence script, so `sound_042.wav` from a
Sylhet speaker and `sound_042.wav` from a Rajshahi speaker are guaranteed
to contain the same sentence. This is what lets the analysis attribute an
acoustic difference to *accent* rather than to *what was said*.

**Script construction.** Text was sourced from twelve topical domains
(national affairs, business, education, entertainment, health,
international, literature, news, politics, religion, science, sports),
segmented into sentences, and normalized. 100 candidate 400-sentence
subsets were generated and scored for lexical diversity; the three most
diverse were manually reviewed for naturalness, grammaticality, and
readability; one was selected and manually curated down to the final,
standardized **250-sentence** script read by every speaker. (Full
methodology: `Dataset collection/Dataset Part.pdf`, and paper §3.2.)

**Recruitment and recording.** 51 native speakers (25 female, 26 male),
aged 19–25, were recruited across 8 regions — Sylhet, Chattogram, Dhaka,
Faridpur, Khulna, Mymensingh, Rajshahi, Rangpur — with regional origin and
native-accent status verified at recruitment (there is no post-hoc
listening study anywhere in this pipeline; see `Part_E/README.md`). Male
participants were recorded in a sound-treated laboratory via a dedicated
audio interface; female participants recorded independently through a
purpose-built Android app (*SUST Kotha*) after remote training on
minimizing background noise. **Recording pipeline is therefore perfectly
confounded with gender** (lab = male, app = female) — every Part in this
pipeline treats that as a hard constraint, never as something to explain
away. Informed consent (written form for lab participants, an equivalent
Google Form for app participants — both under `Dataset collection/`)
explicitly authorized public release of voice recordings together with
district/region, gender, and age, while withholding participant names and
contact details from any public release.

**On-disk layout.**

```
data/<region>/speaker_<gender>_<id>_<name>/sound_<NNN>.wav   # raw corpus (44.1/48 kHz; gitignored, not in this repo)
data/best_diverse_sentences.csv                              # the 250-sentence script, columns: id, sentence
verified_manifest.csv                                        # one row per utterance; the source of truth
data_clean/<region>/.../sound_<NNN>.wav                      # normalized 16 kHz mono, written by Part_C/C0 (gitignored)
```

`sound_<NNN>.wav` links to sentence `id == int(NNN)` in
`best_diverse_sentences.csv` — that mapping is the corpus's entire parallel
structure. `verified_manifest.csv` (repo root) is the corrected,
fully-reviewed manifest: an internal audit (`Part_B/alignment_verification/`)
CER-scored every utterance's audio against its labeled sentence and found
one systematic labeling error — a 29-utterance sentence-numbering rotation
for one Rajshahi speaker — which is corrected in this manifest
(`corrected` / `original_sentence_id` columns document exactly what
changed). As shipped, the manifest has **0 excluded and 0 needs-review**
rows across all 12,750 raw utterances (12,747 after dropping 3 dead
recordings).

| Property | Value |
|---|---|
| Regions (8) | Sylhet, Faridpur, Khulna, Chattogram, Rangpur, Mymensingh, Rajshahi, Dhaka |
| Speakers | 51 (25 female, 26 male) |
| Speakers per region | Sylhet 10 · Faridpur 7 · Chattogram/Dhaka/Khulna/Rajshahi 6 each · Mymensingh/Rangpur 5 each |
| Sentences per speaker | 250 (parallel — identical script for all) |
| Utterances per region | Sylhet 2500 · Faridpur ~1750 · Chattogram/Dhaka/Khulna/Rajshahi 1500 each · Mymensingh/Rangpur 1250 each |
| Total raw utterances | 12,750 (12,747 after dropping 3 dead recordings) |
| Total audio | ~27.74 hours |
| Raw audio size | ~14 GB (`data/`) |
| Normalized audio size | ~3.1 GB (`data_clean/`, 16 kHz mono, peak-normalized) |
| Recording pipelines | lab (male) / SUST Kotha Android app (female) — confounded with gender by design |

**Public release.** `data/` and `data_clean/` are not tracked in this git
repository (too large) and are not included as-is in any public release,
because the raw folder structure embeds participant first names (consent
explicitly bars publishing those). An anonymized, publication-ready copy —
speaker/utterance IDs rederived as collision-free `<region>_<gender>_<seq>`
codes, names stripped, 16 kHz `data_clean/` audio only — is built by
[`build_hf_release.py`](build_hf_release.py) into `HF_Release/BPAC/`
(also gitignored locally) for publishing to Hugging Face under
CC BY-NC 4.0; see [`HF_Release/BPAC/README.md`](HF_Release/BPAC/README.md)
for the dataset card.

## Research conducted

The pipeline runs as five parts, each an independent Python package with
its own `config.yml`, `analyze.py` scripts, and `reports/` outputs
(`report.md` + `summary.json` + CSVs + figures per analysis):

| Part | Package | What it does |
|---|---|---|
| **Part A** — EDA | `bpac_eda` | Corpus descriptive statistics: sentence/utterance coverage, audio length, character/word frequency, VAD-based speaking rate, F0 by region, SNR / audio-quality auditing (A1–A8). |
| **Part B** — ASR benchmark & verification | `bpac_stt` | WER/CER benchmarking with two independent ASR models (wav2vec2-XLSR, Whisper); `alignment_verification/` CER-scores every utterance against its labeled sentence to catch mislabeling, producing `verified_manifest.csv`. |
| **Part C** — Accent structure | `bpac_accent` | Normalizes the corpus to 16 kHz mono (C0, removing the recording-format confound), then measures accent detectability via a WavLM region-distance matrix (C1), speaker-vs-accent embedding separability with leave-one-speaker-out region classification (C2), a within-gender LOSO control that isolates the pipeline/gender confound (C3), and distance-matrix statistics (C4). |
| **Part D** — Phonetic accent map | `bpac_phon` | Builds a Bangla pronunciation lexicon (D0), trains a Montreal Forced Aligner acoustic model from scratch and force-aligns the corpus with a held-out-region generalization check (D1), then measures phone duration & rhythm typology (D2), Lobanov-normalized vowel space (D3), sibilant spectral center of gravity & VOT (D4), intonation contours (D5), fuses everything into an 8×8 phonetic distance matrix (D6), and tests convergence between the phonetic and neural-embedding views of regional structure (D7). |
| **Part E** — Validation | `bpac_valid` | Signal-processing sanity check only (E1): demonstrates why the corpus is deliberately *not* denoised — spectral-gating denoisers would suppress the same broadband energy that Part D's sibilant/VOT measurements depend on. No human listening study is used anywhere in this pipeline; regional-origin labels are validated at recruitment, not by post-hoc listener judgment. |

Every number, table, and figure in the paper traces back to a specific
script and `reports/` output in one of these five parts — see each Part's
own `README.md` for its analyses, config knobs, and how to interpret its
outputs, and `paper_template/PROVENANCE.md` for the number-by-number
mapping used while writing the manuscript.

## Repository layout

```
.
├── Part_A/              EDA (bpac_eda) — A1–A8
├── Part_B/              ASR benchmark + alignment verification (bpac_stt)
├── Part_C/              Accent-structure / embedding separability (bpac_accent) — C0–C4
├── Part_D/              Phonetic accent map (bpac_phon) — D0–D7
├── Part_E/              Validation (bpac_valid) — E1
├── common/              Shared manifest/provenance/consistency-check code
├── Dataset collection/  Consent forms + dataset-construction methodology (source PDFs)
├── Literature Review/   Source PDFs reviewed for the paper's §2
├── Completed Paper/     THE current manuscript: BPAC_paper.md / .pdf, figures/, build_pdf.py
├── paper_template/      Manuscript scaffolding + PROVENANCE.md (number → source-file mapping)
├── paper/, paper_by_me/ Earlier/legacy paper drafts (superseded by Completed Paper/)
├── release/             Dataset-packaging helper scripts
├── HF_Release/BPAC/     Anonymized, Hugging-Face-ready release copy (built, gitignored)
├── build_hf_release.py  Builds HF_Release/BPAC/ from verified_manifest.csv + data_clean/
├── verified_manifest.csv  Corrected, fully-reviewed utterance manifest (source of truth)
├── CORPUS_FROZEN.json   Checksum/size snapshot of data_clean/ at freeze time
├── data/, data_clean/   Raw / normalized audio (gitignored — see "Getting the data" below)
├── Makefile             make part_a .. part_e / make all / make check / make clean
├── CITATION.cff, LICENSE  Code citation metadata (MIT) — dataset license is CC BY-NC 4.0, declared separately in HF_Release/BPAC/README.md
```

## Getting the data

The raw (`data/`) and normalized (`data_clean/`) audio are not committed to
this git repository — they're large binary corpora, not source code. To
run the pipeline yourself:

- **If you have the original recordings**: place them at `data/` as a
  sibling of `Part_A/`, `Part_B/`, etc., following the
  `data/<region>/speaker_<gender>_<id>_<name>/sound_<NNN>.wav` layout
  above, plus `data/best_diverse_sentences.csv`. `data_clean/` is then
  generated for you by `Part_C/C0_Normalize_corpus/analyze.py`.
- **If you only need the released, anonymized audio** (no participant
  names, 16 kHz mono, CC BY-NC 4.0): use the Hugging Face dataset once
  published (link to be added here), or build it locally yourself with
  `python build_hf_release.py` if you already have `data_clean/` and
  `verified_manifest.csv`.

## Step-by-step: running the project

```bash
# 0. Clone and get the raw corpus in place (see "Getting the data" above)
git clone <this-repo>
cd Thesis-BPAC-Exploratory-Data-analysis-of-the-novel-dataset
```

Every `analyze.py` below reads its paths/hyperparameters from that Part's
`config.yml` and takes only `--config` (defaults to `Part_X/config.yml`) —
edit `config.yml`, don't pass ad-hoc flags. Run each analysis individually,
in the order shown (later steps in a Part depend on earlier ones' output).

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
python A8_audio_quality_snr/analyze.py      # WADA-SNR estimate + SNR-vs-WER correlation
cd ..
```

### Part B — ASR benchmark + alignment verification

```bash
cd Part_B && pip install -r requirements.txt

python wav2vec2_stt/analyze.py              # fast ASR model — run first
python whisper_stt/analyze.py               # slower ASR model
python create_report.py                     # combined cross-model WER/CER report

# Alignment verification (produces verified_manifest.csv at the repo root):
python alignment_verification/analyze.py    # CER-scores every utterance against all 250 sentences, flags mismatches
python alignment_verification/spotcheck.py --reviewer <your-name>   # interactive: listen + decide on flagged utterances
python alignment_verification/resolve.py    # applies reviewed decisions -> ../verified_manifest.csv
cd ..
```

### Part C — accent structure

```bash
cd Part_C && pip install -r requirements.txt
pip install umap-learn                      # optional (UMAP; t-SNE works without it)

python C0_Normalize_corpus/analyze.py                             # writes ../data_clean (run first)
python C1_Region_accent_distance_matrix/analyze.py                # WavLM region-distance matrix + dendrogram
python C2_Speaker_vs_accent_embedding_separability/analyze.py     # speaker-vs-accent separability + LOSO region classification
python C3_LOSO_within_gender/analyze.py                           # within-gender LOSO (isolates the pipeline/gender confound)
python C4_distance_statistics/analyze.py                          # bootstrap CIs + permutation tests on C1's distances
cd ..
```

### Part D — phonetic accent map

```bash
cd Part_D && pip install -r requirements.txt

python D0_lexicon/analyze.py                          # builds the Bangla pronunciation lexicon

# D1 forced alignment — Montreal Forced Aligner, installed separately via conda:
python D1_alignment/prepare_corpus.py                 # data_clean/ -> mfa_corpus/ (speaker/utt.wav + .lab)
conda run -n aligner mfa train D1_alignment/mfa_corpus \
    D0_lexicon/reports/bpac_lexicon.txt \
    D1_alignment/reports/bpac_mfa_acoustic_model.zip \
    --output_directory D1_alignment/reports/textgrids --clean
python D1_alignment/finalize.py                        # TextGrids -> phone_tokens.parquet / word_tokens.parquet

# D1 held-out-region robustness check (confirms the aligner isn't just
# memorizing regional pronunciation — see Part_D/README.md invariant #3):
python D1_alignment/prepare_heldout_split.py            # mfa_corpus -> mfa_corpus_no_sylhet + mfa_corpus_sylhet_only
conda run -n aligner mfa train D1_alignment/mfa_corpus_no_sylhet \
    D0_lexicon/reports/bpac_lexicon.txt \
    D1_alignment/reports/bpac_mfa_acoustic_model_no_sylhet.zip \
    --output_directory D1_alignment/reports/textgrids_no_sylhet --clean
conda run -n aligner mfa align D1_alignment/mfa_corpus_sylhet_only \
    D0_lexicon/reports/bpac_lexicon.txt \
    D1_alignment/reports/bpac_mfa_acoustic_model_no_sylhet.zip \
    D1_alignment/reports/textgrids_sylhet_heldout --clean
python D1_alignment/heldout_align_and_compare.py        # compares held-out vs. original phone boundaries

python D2_duration_rhythm/analyze.py       # phone duration + rhythm typology (%V, ΔC, nPVI)
python D3_vowel_space/analyze.py           # Lobanov-normalized F1/F2 vowel space
python D4_consonants/analyze.py            # sibilant spectral center of gravity + VOT (reads raw data/, not data_clean/)
python D5_intonation/analyze.py            # averaged F0 contours per sentence per region
python D6_phonetic_distance/analyze.py     # fuses D2–D5 into an 8x8 phonetic distance matrix + dendrogram
python D7_convergence/analyze.py           # Mantel test: phonetic distance (D6) vs. neural-embedding distance (C1)
cd ..
```

### Part E — validation

```bash
cd Part_E && pip install -r requirements.txt

python E1_denoise_check/analyze.py          # denoising sanity check (depends on Part D's D4 sibilant-token output)
cd ..
```

### Running everything at once (optional)

```bash
make all          # runs each Part's run_all.sh in order (A -> B -> C -> D -> E)
make check        # cross-checks every reports/summary.json against CORPUS_FROZEN.json
make clean        # removes all reports/ and __pycache__ output (data/, data_clean/ untouched)
```

Note that each Part's `run_all.sh` currently chains only its core
`analyze.py` steps (e.g. Part A's covers A1–A6, not yet A7/A8; Part C's
covers C0–C2, not yet C3/C4; Part B's covers the two ASR models + combined
report, not the alignment-verification scripts) — for the full set of
analyses shown above, run the individual commands.

**Building the paper PDF:**

```bash
cd "Completed Paper"
python build_pdf.py     # renders BPAC_paper.md -> BPAC_paper.pdf
```

**Building the anonymized public dataset release:**

```bash
python build_hf_release.py   # requires verified_manifest.csv + data_clean/ already built
```

This writes `HF_Release/BPAC/` (anonymized audio + `metadata.csv` +
`README.md` dataset card), ready to upload to Hugging Face.

## Dataset availability

The corpus is being released on Hugging Face under **CC BY-NC 4.0**
(non-commercial use with attribution); see
[`HF_Release/BPAC/README.md`](HF_Release/BPAC/README.md) for the full
dataset card, license terms, and loading instructions. **The companion
research paper is not yet published** — cite the dataset directly until
then (citation format in the dataset card).

## Citation

See [`CITATION.cff`](CITATION.cff) for the code citation. The manuscript
(`Completed Paper/BPAC_paper.md`) and the dataset card
(`HF_Release/BPAC/README.md`) both carry an interim citation format; both
will be updated once the paper has a venue and DOI.

## License

Code in this repository is MIT-licensed (`LICENSE`). The dataset itself is
licensed separately under CC BY-NC 4.0 — see `HF_Release/BPAC/README.md`.
