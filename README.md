# BPAC: How Detectable Is Regional Accent in Read Bangla?

**A Parallel Corpus and Its Phonetic Structure**

This repository contains the corpus construction, verification, and full
analysis pipeline behind BPAC (the **Bangla Parallel Accent Corpus**) — the
dataset and codebase behind the manuscript *"BPAC: How Detectable Is
Regional Accent in Read Bangla? A Parallel Corpus and Its Phonetic
Structure"* (manuscript in preparation).

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
accent research.

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
once measured directly. Full results, statistics, and discussion are
written up in the companion manuscript (in preparation).

## How the dataset was organized

BPAC's defining property is that it is **parallel**: every one of the 51
speakers read the *same* 250-sentence script, so recording 42 from a
Sylhet speaker and recording 42 from a Rajshahi speaker are guaranteed to
contain the same sentence. This is what lets the analysis attribute an
acoustic difference to *accent* rather than to *what was said*.

**Script construction.** Text was sourced from twelve topical domains
(national affairs, business, education, entertainment, health,
international, literature, news, politics, religion, science, sports),
segmented into sentences, and normalized. 100 candidate 400-sentence
subsets were generated and scored for lexical diversity; the three most
diverse were manually reviewed for naturalness, grammaticality, and
readability; one was selected and manually curated down to the final,
standardized **250-sentence** script read by every speaker. (Full
methodology: `Dataset collection/Dataset Part.pdf`.)

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

**Utterance identity.** Every speaker's recordings are numbered so the
sentence identifier is explicit — a speaker's recording of sentence 42 of
the shared script is always labeled as such — so any two speakers', or any
two regions', renditions of "the same sentence" can be retrieved and
compared directly. `verified_manifest.csv` (tracked, at the repo root) is
the corrected, fully-reviewed record of this mapping: an internal audit
(`Part_B/alignment_verification/`) CER-scored every utterance's audio
against its labeled sentence and found one systematic labeling error — a
29-utterance sentence-numbering rotation for one Rajshahi speaker — which
is corrected in this manifest (`corrected` / `original_sentence_id`
columns document exactly what changed). As shipped, the manifest has **0
excluded and 0 needs-review** rows across all 12,750 raw utterances
(12,747 after dropping 3 dead recordings).

| Property | Value |
|---|---|
| Regions (8) | Sylhet, Faridpur, Khulna, Chattogram, Rangpur, Mymensingh, Rajshahi, Dhaka |
| Speakers | 51 (25 female, 26 male) |
| Speakers per region | Sylhet 10 · Faridpur 7 · Chattogram/Dhaka/Khulna/Rajshahi 6 each · Mymensingh/Rangpur 5 each |
| Sentences per speaker | 250 (parallel — identical script for all) |
| Utterances per region | Sylhet 2500 · Faridpur ~1750 · Chattogram/Dhaka/Khulna/Rajshahi 1500 each · Mymensingh/Rangpur 1250 each |
| Total raw utterances | 12,750 (12,747 after dropping 3 dead recordings) |
| Total audio | ~27.74 hours |
| Recording pipelines | lab (male) / SUST Kotha Android app (female) — confounded with gender by design |

**Public release.** The raw recordings are not distributed as collected —
the original folder/file-naming scheme embeds participant first names,
which consent explicitly bars from public release. An anonymized,
publication-ready copy (speaker/utterance IDs rederived as collision-free
`<region>_<gender>_<seq>` codes, participant names stripped entirely) is
built by [`build_hf_release.py`](build_hf_release.py) for publishing to
Hugging Face under CC BY-NC 4.0.

## Research conducted

The pipeline runs as five parts, each an independent Python package with
its own `config.yml`, `analyze.py` scripts, and `reports/` outputs
(`report.md` + `summary.json` + CSVs + figures per analysis):

| Part | Package | What it does |
|---|---|---|
| **Part A** — EDA | `bpac_eda` | Corpus descriptive statistics: sentence/utterance coverage, audio length, character/word frequency, VAD-based speaking rate, F0 by region, SNR / audio-quality auditing (A1–A8). |
| **Part B** — ASR benchmark & verification | `bpac_stt` | WER/CER benchmarking with two independent ASR models (wav2vec2-XLSR, Whisper); `alignment_verification/` CER-scores every utterance against its labeled sentence to catch mislabeling, producing `verified_manifest.csv`. |
| **Part C** — Accent structure | `bpac_accent` | Normalizes the corpus (C0, removing the recording-format confound), then measures accent detectability via a WavLM region-distance matrix (C1), speaker-vs-accent embedding separability with leave-one-speaker-out region classification (C2), a within-gender LOSO control that isolates the pipeline/gender confound (C3), and distance-matrix statistics (C4). |
| **Part D** — Phonetic accent map | `bpac_phon` | Builds a Bangla pronunciation lexicon (D0), trains a Montreal Forced Aligner acoustic model from scratch and force-aligns the corpus with a held-out-region generalization check (D1), then measures phone duration & rhythm typology (D2), Lobanov-normalized vowel space (D3), sibilant spectral center of gravity & VOT (D4), intonation contours (D5), fuses everything into an 8×8 phonetic distance matrix (D6), and tests convergence between the phonetic and neural-embedding views of regional structure (D7). |
| **Part E** — Validation | `bpac_valid` | Signal-processing sanity check only (E1): demonstrates why the corpus is deliberately *not* denoised — spectral-gating denoisers would suppress the same broadband energy that Part D's sibilant/VOT measurements depend on. No human listening study is used anywhere in this pipeline; regional-origin labels are validated at recruitment, not by post-hoc listener judgment. |

Every number, table, and figure in the manuscript traces back to a
specific script and `reports/` output in one of these five parts — see
each Part's own `README.md` for its analyses, config knobs, and how to
interpret its outputs.

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
├── Literature Review/   Source PDFs reviewed for the manuscript's literature review
├── release/             Dataset-packaging helper scripts
├── build_hf_release.py  Builds an anonymized, publication-ready release copy of the corpus
├── verified_manifest.csv  Corrected, fully-reviewed utterance manifest (source of truth)
├── CORPUS_FROZEN.json   Checksum/size snapshot of the normalized corpus at freeze time
├── check_audio_quality.py, mark_empty_files.py  Standalone audio-QA utilities
├── Makefile             make part_a .. part_e / make all / make check / make clean
├── CITATION.cff, LICENSE  Code citation metadata (MIT)
```

This repo's `.gitignore` excludes large or generated content that isn't
meant to be version-controlled — the raw and normalized audio corpus, the
built publication-ready dataset release, and this project's paper drafts
among them. See `.gitignore` at the repo root for the exact list.

## Getting the data

This repository is the *code*, not the corpus itself. The audio is
intentionally not committed to git (large binary data), so a fresh clone
needs it supplied separately before the pipeline can run:

- **If you have the original recordings**: place them alongside `Part_A/`,
  `Part_B/`, etc., following the layout each Part's `config.yml` and
  `README.md` document (region → speaker → numbered sentence recordings,
  plus the 250-sentence script as a CSV). The normalized copy used by
  Parts C–E is then generated for you by
  `Part_C/C0_Normalize_corpus/analyze.py`.
- **If you only need the released, anonymized audio** (no participant
  names, CC BY-NC 4.0): use the Hugging Face dataset once published (link
  to be added here), or build a local copy yourself with
  `python build_hf_release.py` once you already have the normalized
  corpus and `verified_manifest.csv` in place.

## Step-by-step: running the project

```bash
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

python C0_Normalize_corpus/analyze.py                             # builds the normalized corpus copy (run first)
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
python D1_alignment/prepare_corpus.py                 # converts the normalized corpus into MFA's expected layout (speaker/utt.wav + .lab)
conda run -n aligner mfa train D1_alignment/mfa_corpus \
    D0_lexicon/reports/bpac_lexicon.txt \
    D1_alignment/reports/bpac_mfa_acoustic_model.zip \
    --output_directory D1_alignment/reports/textgrids --clean
python D1_alignment/finalize.py                        # TextGrids -> phone_tokens.parquet / word_tokens.parquet

# D1 held-out-region robustness check (confirms the aligner isn't just
# memorizing regional pronunciation — see Part_D/README.md invariant #3):
python D1_alignment/prepare_heldout_split.py            # splits the prepared MFA corpus into a no-Sylhet training set and a Sylhet-only held-out set
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
python D4_consonants/analyze.py            # sibilant spectral center of gravity + VOT (reads full-bandwidth raw audio, not the normalized copy)
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

### Building the anonymized public dataset release

```bash
python build_hf_release.py   # requires verified_manifest.csv + the normalized corpus already built
```

This builds a local, anonymized, publication-ready copy of the corpus
(audio + `metadata.csv` + a dataset card), ready to upload to Hugging Face.

### Running everything at once (optional)

```bash
make all          # runs each Part's run_all.sh in order (A -> B -> C -> D -> E)
make check        # cross-checks every reports/summary.json against CORPUS_FROZEN.json
make clean        # removes all reports/ and __pycache__ output (the audio corpus itself is untouched)
```

Note that each Part's `run_all.sh` currently chains only its core
`analyze.py` steps (e.g. Part A's covers A1–A6, not yet A7/A8; Part C's
covers C0–C2, not yet C3/C4; Part B's covers the two ASR models + combined
report, not the alignment-verification scripts) — for the full set of
analyses shown above, run the individual commands.

## Dataset availability

The corpus is being released on Hugging Face under **CC BY-NC 4.0**
(non-commercial use with attribution). **The companion research paper is
not yet published** — cite the dataset directly until then; see the
dataset's own card on Hugging Face (link to be added here once published)
for the citation format, full license terms, and loading instructions.

## Citation

See [`CITATION.cff`](CITATION.cff) for the code citation. A dataset-specific
citation format is provided on the Hugging Face dataset card once
published; both will be updated to the final paper citation once the
manuscript has a venue and DOI.

## License

Code in this repository is MIT-licensed (`LICENSE`). The dataset itself is
licensed separately under CC BY-NC 4.0 (full terms on the Hugging Face
dataset card once published).
