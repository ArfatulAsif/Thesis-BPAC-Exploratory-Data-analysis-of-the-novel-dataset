# Exploratory Data Analysis — Bangla Parallel Accent Corpus (Procedure, as executed)
<br>
*Supports an accent-conversion paper as its dataset/characterization section. Documents the procedure actually followed: corpus descriptive statistics, speaking rate & pitch, whole-data and region-wise STT, region accent-distance matrix, the speaker-vs-accent confound test, and a corpus-normalization + corrected re-analysis stage (Part C2) added after the EDA itself revealed a recording-format confound. Preprocessing for Parts A–C was minimal and applied to copies; Part C2 added a full normalization pass. Raw files were never modified.*
---
## Preprocessing (what was actually applied)
The EDA used **two preprocessing regimes**, because the analysis itself revealed that the "minimal" regime was insufficient for the embedding-based work.

**Regime 1 — minimal & uniform (Parts A, B, and the initial Part C).** Only what each measurement needs to be valid; no denoising, no loudness normalization.
- **Text:** Unicode **NFC** normalization before any counting or scoring; one fixed "character" convention (grapheme cluster, reported); one fixed word tokenizer; a stated vowel/consonant grapheme scheme.
- **Audio length:** one fixed basis (raw file duration vs speech-only), used everywhere and stated.
- **Model-based steps (STT, distance, embeddings):** resample to **16 kHz mono** (format requirement). For STT, the **same** NFC + punctuation normalization on reference and hypothesis before scoring. For distance, uniform leading/trailing **silence trimming**.
- **Golden rule:** any light processing applied **identically to every region**, on copies, with raw audio untouched.

**Regime 2 — full normalization (Part C2 / stage C0).** The data-quality audit (Section "Data-quality audit" below) showed the corpus mixes recording formats (44.1 kHz/48 kHz, mono/stereo) **split along regional lines**, and contains many empty/dead files. Because self-supervised embeddings are sensitive to recording conditions, on-the-fly resampling alone did not erase the format fingerprint, so a dedicated normalization stage was added: every clip **resampled to a single 16 kHz mono format, empty/dead files dropped, levels peak-normalized**, written to a **clean corpus copy** that the corrected C1/C2 then consume. This regime supersedes the "no loudness normalization" rule for the accent-structure analyses specifically.

> Why both: Regime 1 keeps the descriptive and STT measurements faithful to the recordings; Regime 2 is required so the embedding geometry reflects **accent**, not **recording pipeline**. The discovery that forced Regime 2 is itself an EDA result.
---
# Part A — Corpus descriptive statistics *(executed under Regime 1)*
### A1. Unique sentences
- **Compute:** number of distinct standard-Bengali sentences; confirm the set is shared across all speakers/regions (parallel property).
- **Preprocessing:** NFC on transcripts.
- **Report:** total unique sentences; a sentence×speaker coverage check.
- **Outcome:** parallel structure confirmed; the coverage check flagged a missing parallel cell and a Faridpur directory-capitalization inconsistency, both corrected by normalization.
### A2. Audio length — total, per speaker, per region
- **Compute:** total corpus duration; per speaker; per region; per gender.
- **Preprocessing:** fixed duration basis, stated and held constant.
- **Report:** headline total; per-speaker duration bar chart; per-region table (and per-gender within region).
- **Look for:** speakers/regions with markedly less audio — imbalance worsens the speaker–accent confound (only ~6 speakers/region).
### A3. Total utterance count
- **Compute:** total utterances; per region, per speaker, per gender.
- **Report:** counts table at each aggregation level.
- **Look for:** uneven utterance counts across regions.
### A4. Character count (vowel / consonant)
- **Compute:** total characters; split into vowels and consonants under a stated Bangla grapheme scheme (independent vowels, vowel signs/kar, consonants, conjuncts, hasanta).
- **Preprocessing:** NFC; stated grapheme convention.
- **Report:** total character count; vowel vs consonant counts and ratio; optional character-frequency distribution.
- **Note:** text is shared across regions → reported **once**; characterizes the script, not the accent.
### A5. Word count (with per-word frequency)
- **Compute:** total tokens; unique types; per-word frequency table; type–token ratio.
- **Preprocessing:** NFC; stated tokenizer.
- **Report:** totals; word-frequency table / top-N; Zipf-style distribution.
- **Note:** shared across regions → reported once; documents vocabulary coverage of the sentence set.
### A6. Speaking rate & pitch (F0)
- **Purpose:** lightweight sociolinguistic characterization (pace and pitch per region), made clean by the parallel design.
- **Compute:** **speaking rate (WPM)** per utterance → aggregated per region/speaker/gender; **pitch (F0)** mean/SD/range per region via Praat (`parselmouth`), with a `librosa.pyin` fallback.
- **Preprocessing:** WPM on **speech-only duration** (silence trimmed), same basis across regions; F0 with stated per-gender floor/ceiling, constant across regions.
- **Report:** per-region WPM and mean-F0 table; violin/box plots of WPM and F0 by region, **faceted by gender**.
- **Outcome / caveat:** confirmed that **F0 is dominated by gender**, so all pitch (and rate) results are reported **split by gender**, then read for region effects within each gender — pooling would conflate gender with accent.

> A1, A4, A5 describe the fixed **content** (identical for every region). A2, A3, A6 are the **per-region/per-speaker** quantities — A2/A3 prove balance; A6 is the first place accent-related delivery differences appear.
---
# Part B — Speech-to-Text (STT) analysis *(executed under Regime 1)*
### B1. Whole-data STT — best model vs other SOTA
- **Purpose:** establish overall transcription accuracy and benchmark two architectures.
- **Compute (as done):** ran **two architecturally distinct, general (not regionally-adapted) Bangla models** over all 9,750 utterances — **Whisper-medium** (`bengaliAI/tugstugi_bengaliai-asr_whisper-medium`, encoder–decoder) and **XLS-R wav2vec2** (`shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice`, CTC) — and computed **WER and CER** for each. General models were used deliberately so accent shows up as degradation rather than being absorbed by regional fine-tuning.
- **Preprocessing:** 16 kHz mono; identical NFC + punctuation normalization on reference and hypothesis; greedy decoding; **no per-region denoising**. Empty/dead audio detected at transcription time, flagged, and **excluded from scoring** so dead files cannot corrupt WER/CER.
- **Engineering:** the pipeline was decoupled **per model** (each model transcribes into a shared, resumable cache and writes its own report) with a separate **combined-report** step, so the fast model can be evaluated before committing to the slow one.
- **Report:** model-comparison table (model · WER · CER) on the full corpus.
- **Outcome:** the two models were **near-tied but split by metric** (Whisper lower WER, wav2vec2 lower CER — the expected encoder-decoder vs CTC signature); absolute errors were reasonable for read standard Bengali.
### B2. Region-wise STT — accuracy matrix
- **Purpose:** quantify how far each region departs from standard Bengali.
- **Compute:** per-region **WER and CER** for both models (reusing the cache).
- **Preprocessing:** identical to B1, applied uniformly across regions.
- **Report:** region × {WER, CER} matrix (per model + mean); per-region bar chart; a "distance from standard" ranking.
- **Outcome (important):** the ranking was **scientifically suspicious** — Dhaka, the *standard* dialect, came out **hardest**, and both models spiked on it identically. This pattern indicates a fault in the input (audio/reference), not accent, so the ranking was **not** reported as "distance from standard," and the anomaly triggered the data-quality audit below.
---
# Data-quality audit *(added in response to the B2 anomaly)*
A dedicated audit of the raw audio was run after B2, since two architecturally different models spiking on the same region points to data rather than accent.
- **Per-region audio-quality diagnostic:** sample rate, channels, duration, loudness (RMS), estimated SNR, clipping, silence, dead files, with robust outlier flagging.
- **Empty/dead-file inventory:** every zero-length / too-short / silent / corrupt `.wav` enumerated and classified as **conforming (scored)** vs **non-conforming**.
- **Outcome:** audio *quality* (SNR 35–49 dB, loudness) was comparable across regions → **noise ruled out** as the cause of Dhaka's error. But two systematic problems were exposed: (1) **mixed recording formats** — three regions (Faridpur, Mymensingh, Rangpur) are uniform 48 kHz stereo with zero empty files, the other five mix 44.1/48 kHz and mono/stereo; (2) **empty/dead files concentrated** in Dhaka, Rajshahi, Chattogram, Sylhet, Khulna. These two findings directly shaped Part C and necessitated Part C2.
---
# Part C — Accent structure, initial *(executed under Regime 1)*
### C1. Region accent-distance matrix
- **Purpose:** quantify acoustic distance between regions on identical content.
- **Compute (as done):** pooled **WavLM-Large** (`microsoft/wavlm-large`) utterance embeddings; cosine distance between same-sentence cross-region pairs, averaged over sentences and speaker pairs to fill a region×region matrix. (A classic **MFCC + DTW** baseline was also available.)
- **Preprocessing:** 16 kHz mono; uniform silence trimming; identical feature settings across regions.
- **Report:** region×region distance heatmap; dendrogram.
- **Outcome:** the tightest cluster (**Faridpur/Mymensingh/Rangpur**) coincided exactly with the **recording-format group** from the audit — i.e. the C1 structure was confounded with recording pipeline and could not be trusted as dialectal. Motivated Part C2.
### C2. Speaker-vs-accent embedding separability (the confound test)
- **Purpose:** the key check — does the corpus cluster by **accent** or by **speaker**? (~6 speakers/region, no speaker spanning two accents.)
- **Compute (as done):** **speaker embeddings** via SpeechBrain **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) and **SSL content embeddings** via WavLM; **t-SNE/UMAP** projections coloured the same points twice (by region, by speaker); **silhouette scores** for region vs speaker labels.
- **Preprocessing:** 16 kHz mono; models handle internal normalization.
- **Report:** paired projections; silhouette table; exact checkpoints stated.
- **Outcome:** in the content space, **region barely separated while speaker separated more** — the corpus clustered by **speaker, not accent**, i.e. the small-speaker-count confound was biting. Together with C1's format confound, this motivated the corrected Part C2.
---
# Part C2 — Cleaned-corpus normalization & corrected accent analysis
*A corrected, methodologically stronger re-run that removes the recording-format confound (Regime 2) and upgrades the methods. C0 produces a clean corpus; the improved C1/C2 run on it.*
### C0. Corpus normalization & alignment diagnostics
- **Purpose:** remove the recording-format confound before embeddings, and probe the recurring Rajshahi/Dhaka data anomalies.
- **Compute (as done):** read the raw corpus; **resample every clip to 16 kHz mono**, **drop empty/dead files**, **peak-normalize**, and write a **clean corpus copy** consumed by C1/C2. Plus a **duration-based parallel-alignment diagnostic**: for each shared sentence, clip durations should be similar across regions, so a region whose `sound_NNN.wav` durations deviate systematically per sentence is flagged as **alignment-suspect** (a proxy for the filename index not matching the intended sentence) — with focus on Rajshahi/Dhaka.
- **Report:** per-region normalization summary (written vs dead-dropped); per-region alignment-suspect rate; list of suspect (region, sentence) cells to spot-check.
- **Look for:** whether Rajshahi/Dhaka are flagged by the alignment check — distinguishing a data-alignment fault from genuine accent.
### C1 (improved). Region accent-distance matrix on the clean corpus
- **Compute (as done):** **middle-layer** WavLM features (the most *phonetic* layer; the last layer drifts toward speaker/denoising and was emphasizing speaker in the initial C1). Default method `ssl_controlled`: **per-sentence centering** (removes content) + **speaker-pair averaging** (reduces speaker), cosine distance. Alternative methods: `loso_confusion` (distance derived from a leave-one-speaker-out region classifier's confusions) and the `mfcc_dtw` baseline.
- **Preprocessing:** Regime 2 clean corpus (uniform 16 kHz mono).
- **Report:** distance matrix, heatmap, dendrogram, distance-from-standard ranking.
- **Outcome:** with format now uniform, the cluster **changed** (evidence the original was partly format) and the new grouping was **geographically plausible** — a surviving cluster is now dialectal rather than a recording artifact.
### C2 (improved). Speaker-vs-accent separability on the clean corpus
- **Compute (as done):** beyond silhouette, three analyses that actually settle the nested-speaker confound:
  - **Leave-one-speaker-out (LOSO) region classification** — train to predict region, evaluate on **held-out speakers**; above-chance accuracy can only come from a speaker-independent accent signal (not speaker memorization). **This is the primary, valid metric.**
  - **Speaker-residualized silhouette** — subtract each speaker's centroid, then measure region separation. **Caveat (stated in the report):** because every speaker is nested in one region, removing the speaker mean also removes the region mean, so a near-zero residualized score is *expected by design* and quantifies the confound rather than disproving accent.
  - **Per-sentence region silhouette** — controls content via the parallel design.
  - Paired **t-SNE/UMAP** projections (by region, by speaker) for the speaker and content spaces.
- **Preprocessing:** Regime 2 clean corpus; middle-layer WavLM + ECAPA.
- **Report:** LOSO accuracy vs chance + per-region recall + confusion matrix; silhouette table (incl. residualized) with the nesting caveat; per-sentence silhouette; paired projections; exact checkpoints.
- **Outcome:** LOSO recovered region on **unseen speakers above chance** (a genuine, speaker-independent accent signal), but **modest and uneven** — carried by a few divergent varieties (Sylhet strongest) while several regions are barely separable. The C1 clusters and C2 confusions **agree** on the acoustic neighborhoods, corroborating that the structure is dialectal. Near-zero whole-corpus/per-sentence silhouettes are consistent with a real-but-subtle signal in a space dominated by speaker identity.
---
## What was produced (summary)
| # | Analysis | Output | Aggregation | Regime |
|---|---|---|---|---|
| A1 | Unique sentences | count + coverage check | corpus | 1 |
| A2 | Audio length | total + bar chart + table | corpus / speaker / region / gender | 1 |
| A3 | Utterance count | counts table | corpus / region / speaker / gender | 1 |
| A4 | Character count (vowel/consonant) | totals + ratio | corpus (shared text) | 1 |
| A5 | Word count + frequencies | freq table + Zipf plot | corpus (shared text) | 1 |
| A6 | Speaking rate & pitch (F0) | per-region table + violin/box by gender | region / speaker / gender | 1 |
| B1 | Whole-data STT (2 models) | model × WER/CER table | corpus | 1 |
| B2 | Region-wise STT | region × WER/CER matrix + bar chart | region | 1 |
| — | Data-quality audit | audio-quality table + empty-file inventory | region / file | 1 |
| C1 | Accent-distance matrix (initial) | heatmap + dendrogram | region pairs | 1 |
| C2 | Speaker-vs-accent embeddings (initial) | paired t-SNE/UMAP + silhouette | speaker vs region | 1 |
| C0 | Corpus normalization + alignment diagnostics | clean corpus + per-region/alignment reports | region / file | 2 |
| C1′ | Accent-distance matrix (clean, middle-layer, controlled) | heatmap + dendrogram (+ LOSO-confusion option) | region pairs | 2 |
| C2′ | Separability (clean): LOSO + residualized + per-sentence | LOSO acc/confusion + silhouette table + projections | speaker vs region | 2 |

**The highest-value results for the thesis:** the region-wise STT matrix (B2, read with the data-quality caveat), the cleaned-corpus accent-distance matrix (C1′), and the cleaned-corpus separability test (C2′, with LOSO as the primary confound-proof metric) — together they show accent variation exists, measure its structure, and confront the speaker confound honestly. A6 adds a cheap sociolinguistic layer; the data-quality audit and C0 are the bridge that makes the Part C2 results trustworthy.
---
## One-line reporting tag for every table/figure
State: audio version used (raw / speech-only / 16 kHz model input / **C0-normalized clean**), aggregation level, that preprocessing was identical across regions, and (for model-based results) the exact model/checkpoint and — for WavLM — the **layer** used.