# Exploratory Data Analysis of the BPAC Corpus

### Data characterization for direct speech-to-speech Bangla regional accent conversion

---

## 1. Purpose and scope

This document consolidates the exploratory data analysis (EDA) carried out on the **Bangla Parallel Accent Corpus (BPAC)**, the novel dataset assembled for a thesis on *direct, textless speech-to-speech regional accent conversion* — converting the regional accent of an utterance (e.g. Sylhet → Rajshahi) while preserving the speaker's timbre, the linguistic content, and the prosody.

The EDA serves three goals. First, to **characterize** the corpus (its coverage, balance, and acoustic properties). Second, to **audit data quality**, since a newly assembled corpus is the most likely source of silent errors. Third, and most importantly for an accent-conversion paper, to **test whether the corpus actually carries a regional-accent signal**, or whether apparent "accent" structure is in fact speaker identity in disguise. The last point is the analysis a reviewer scrutinizes most closely, because the corpus design (few speakers per region, no speaker recorded in two regions) creates an inherent speaker–accent confound.

The analysis was conducted in four parts: **Part A** (descriptive characterization), **Part B** (speech-to-text analysis), **Part C** (accent-structure analysis), and **Part C2** (a corrected re-analysis on a cleaned corpus after the EDA itself revealed a recording-format confound).

---

## 2. The BPAC corpus

BPAC is a **parallel** accent corpus: every speaker reads the **same 250 standard-Bengali sentences**, so any acoustic difference between regions on a given sentence is attributable to accent (and speaker), not to differing content. This parallel design is the corpus's central methodological advantage and is exploited throughout the EDA.

The corpus contains **39 speakers** (20 male, 19 female) drawn from **8 Bangladeshi regions** — Chattogram, Dhaka, Faridpur, Khulna, Mymensingh, Rajshahi, Rangpur, and Sylhet — with roughly six speakers per region. The full set of conforming utterances is therefore 39 × 250 = **9,750 recordings**. Files are organized as `data/BPAC/<region>/speaker_<gender>_<id>_<name>/sound_<NNN>.wav`, with the parallel link given by matching the three-digit filename index to the sentence id in `best_diverse_sentences.csv`.

The regions span a wide linguistic range. Dhaka is the standard prestige variety; Dhaka and Rajshahi are relatively mild accents; Sylheti and Chittagonian are highly divergent — close to separate languages. This means accent "distance" is genuinely uneven across region pairs, a fact the EDA both relies on and confirms.

---

## 3. Part A — Descriptive characterization

Part A established the basic shape and integrity of the corpus through six descriptive analyses (A1–A6), combining text-level and audio-level characterization with all processing applied identically across regions to avoid introducing per-region bias.

On the **text side**, the analysis verified the parallel structure (every speaker should read every sentence), inventoried the 250 sentences, and computed linguistic features of the Bengali text using Unicode-correct processing: NFC normalization, grapheme-cluster segmentation, vowel/consonant codepoint classification, tokenization, and syllable counting. The completeness check immediately surfaced two minor structural issues — a missing parallel cell and a capitalization inconsistency in the Faridpur directory name — both of which were corrected by normalization.

On the **audio side**, Part A characterized per-utterance **duration**, **speaking rate** (words per minute), and **pitch** (fundamental frequency, F0), with pitch extracted via Praat (parselmouth) and a librosa fallback. A key methodological finding here was that **F0 is dominated by speaker gender**, so all pitch and rate statistics are reported split by gender rather than pooled; pooling would conflate a biological factor with the accent signal of interest.

Part A's role is foundational rather than evidential: it confirms the corpus is broadly balanced and well-formed, and it establishes the per-speaker, per-region descriptive baselines against which the later analyses are interpreted.

---

## 4. Part B — Speech-to-text (ASR) analysis

Part B used automatic speech recognition as a quantitative probe of the corpus. The logic is that the *error rate* of a standard-Bengali ASR system on each region is a proxy for how far that region's speech departs from standard Bengali — a region whose accent diverges more should be transcribed less accurately.

### 4.1 Model selection

Two architecturally distinct ASR models were used, deliberately chosen as **general** Bengali models rather than regional variants (a regionally-adapted model would mask the very accent gradient the analysis is trying to measure):

- **Whisper-medium** (`bengaliAI/tugstugi_bengaliai-asr_whisper-medium`) — an encoder–decoder model with an internal language model.
- **XLS-R wav2vec2** (`shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice`) — a CTC model without an internal language model.

Two architectures are sufficient here because the ASR analysis is a *characterization* of the corpus, not the publication-weight evaluation; the rigorous benchmarking belongs to the conversion-side evaluation. Using one encoder–decoder and one CTC model also provides architectural diversity, so agreement between them is more meaningful than agreement between two similar models.

### 4.2 B1 — Whole-corpus accuracy

Across all 9,750 utterances, the two models were **near-tied but split by metric**:

| Model | WER | CER |
|---|---|---|
| Whisper-medium (encoder–decoder) | 0.228 | 0.115 |
| XLS-R wav2vec2 (CTC) | 0.289 | 0.108 |

Whisper achieves the lower **word** error rate, while wav2vec2 achieves the lower **character** error rate. This split is not noise — it is the textbook signature of the two architectures. Whisper's internal language model produces well-formed words (low WER) but can substitute or insert plausible wrong words; the CTC model transcribes more character-faithfully (low CER) but produces more malformed word forms (higher WER). The absolute error levels (CER ≈ 0.11, WER ≈ 0.23–0.29) are reasonable for read standard Bengali and confirm both models function correctly on the corpus. (The wav2vec2 model ran with greedy decoding without its optional 5-gram language model; enabling it would likely lower its WER specifically.)

### 4.3 B2 — Region-wise accuracy

The region-wise breakdown produced a result that was **scientifically suspicious and therefore highly informative**. Ranked by mean CER across both models, Dhaka was the *hardest* region (CER 0.243), with Rajshahi second (0.147), while the genuinely divergent Sylhet (0.089) and Chattogram (0.072) sat near the *easy* end.

| Region | Mean CER |
|---|---|
| Dhaka | 0.243 |
| Rajshahi | 0.147 |
| Sylhet | 0.089 |
| Khulna | 0.081 |
| Rangpur | 0.076 |
| Chattogram | 0.072 |
| Mymensingh | 0.064 |
| Faridpur | 0.060 |

This inverts the known linguistics: Dhaka *is* the standard, so it should be the **easiest**, not the hardest, and the most divergent dialects should not be among the easiest. Furthermore, both architecturally different models spiked on Dhaka by nearly the same amount — a pattern that points to a fault in the *input* (audio or reference) rather than to genuine accent, since two different models converging on the same anomaly usually indicates a data problem. The conclusion drawn was explicit: **the B2 ranking must not be reported as "distance from standard" until the Dhaka and Rajshahi anomalies are explained.** This finding directly motivated the data-quality investigation in Section 5.

Empty/dead audio files were detected during transcription, flagged, and excluded from the error-rate computation so that dead recordings could not corrupt the WER/CER. The pipeline was also restructured so the two models run independently (each producing its own report) with a combined cross-model report, allowing the fast model to be checked before committing to the slow one.

---

## 5. Data-quality investigation

The B2 anomaly triggered a dedicated audit of the raw audio, which uncovered two systematic, corpus-wide problems that had not been visible from the descriptive statistics alone.

**Audio quality was ruled out as the cause.** Per-region signal-to-noise ratio (35–49 dB) and loudness were comparable across regions, and Dhaka's audio quality was mid-pack rather than degraded. This is an important negative result: it means Dhaka's elevated error is **not** a noise problem, which redirects the explanation toward reference alignment and the systematic issues below.

**Two real, structural problems were exposed.** First, the corpus contains **mixed recording formats**: three regions (Faridpur, Mymensingh, Rangpur) are uniformly 48 kHz stereo with zero empty files, while the other five are a mix of 44.1 kHz and 48 kHz and a mix of mono and stereo. Second, there is a **large number of empty/dead files**, concentrated heavily in the same regions that behaved anomalously in B2 — approximately 751 empty files in Dhaka, 750 in Rajshahi, 500 in Chattogram, 350 in Sylhet, and 250 in Khulna (most being non-conforming files that the analysis pipeline already skipped, but their concentration is diagnostic). An inventory tool was built to enumerate every empty/dead file and classify each as conforming (in the scored set) or non-conforming.

These findings established two things that shaped the rest of the EDA: the corpus has a **recording-format inconsistency** that splits cleanly along regional lines, and it has **dead recordings concentrated in specific regions**. Both become critical in the accent-structure analysis, because acoustic-embedding methods are far more sensitive to recording conditions than a word transcriber is.

---

## 6. Part C — Accent-structure analysis (initial)

Part C moved from "how well is each region transcribed" to the direct question: **does the corpus cluster by accent, or by speaker?** Two analyses were built, using best-in-class self-supervised speech models — **WavLM-Large** for content/accent embeddings and **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) for speaker-identity embeddings.

### 6.1 C1 — Region accent-distance matrix

C1 computed a region-by-region acoustic distance matrix by comparing WavLM embeddings of the *same sentence* across regions, averaged over sentences and speaker pairs. The result showed a tight cluster of **Faridpur, Mymensingh, and Rangpur**, with Rajshahi as an outlier.

Critically, **those three clustered regions are exactly the three that share the uniform 48 kHz stereo recording format**. The tightest "accent" cluster coincided precisely with the recording-format group identified in Section 5. This is the central cautionary finding of the initial Part C: the C1 structure could not be trusted as dialectal, because it was confounded with recording pipeline.

### 6.2 C2 — Speaker-versus-accent separability (the confound test)

C2 quantified separability using silhouette scores for region labels versus speaker labels in each embedding space:

| Embedding space | Silhouette by region | Silhouette by speaker |
|---|---|---|
| Speaker (ECAPA) | 0.045 | 0.321 |
| Content (WavLM, last layer) | −0.017 | 0.057 |

The interpretation was unambiguous and unfavorable: in the content space, region barely separated (a near-zero/negative silhouette), while speaker separated more clearly. **The corpus clustered by speaker, not by accent** — the small-speaker-count confound was biting. This is precisely the reviewer objection an accent-conversion paper must pre-empt, and on the raw corpus the objection had teeth.

Part C therefore delivered two diagnoses rather than a clean result: a **recording-format confound** in C1, and a **speaker-identity confound** in C2. Both demanded a corrected re-analysis.

---

## 7. Part C2 — Cleaned-corpus re-analysis

Part C2 is a corrected, methodologically stronger re-run that addresses the confounds Part C exposed. It introduces a normalization stage and upgrades the analysis methods.

### 7.1 C0 — Corpus normalization and alignment diagnostics

A normalization stage produces a **clean copy** of the corpus in which every clip is resampled to a uniform **16 kHz mono** format, empty/dead files are dropped, and levels are peak-normalized. This removes the recording-format fingerprint that on-the-fly resampling alone cannot fully erase (different anti-aliasing and stereo-downmix histories leave residual artifacts). C1 and C2 then operate on this clean tree.

C0 also runs a **duration-based parallel-alignment diagnostic** aimed specifically at the recurring Rajshahi problem: for each shared sentence, clip durations should be similar across regions, so a region whose `sound_NNN.wav` durations deviate systematically per sentence is flagged as alignment-suspect (a proxy for the filename index not matching the intended sentence). This gives a way to investigate whether Rajshahi's anomalies are a data-alignment issue rather than genuine accent.

### 7.2 Methodological upgrades

Four substantive improvements were made over the initial Part C, motivated by the confounds:

- **Middle-layer WavLM features.** Layer-wise studies of self-supervised speech models show the *middle* layers carry the most phonetic content, whereas the final layers drift toward the pretraining objective (which for WavLM includes speaker/denoising). Using the last layer in Part C plausibly emphasized speaker information; Part C2 uses a middle layer to foreground accent.
- **Parallel-aware distance.** The recommended C1 method subtracts the per-sentence mean embedding (removing content) and averages over speaker pairs (reducing speaker), exploiting the parallel design rather than ignoring it.
- **Leave-one-speaker-out (LOSO) region classification.** This is the confound-resistant test: a classifier is trained to predict region and evaluated on *held-out speakers*. Because the test speakers are unseen, above-chance accuracy can only come from a speaker-independent accent signal — it cannot be speaker memorization. This is the valid measure of accent for a corpus where speakers are nested within regions.
- **Per-sentence region silhouette**, which controls content by measuring region separation within each shared sentence.

### 7.3 C1 results on the clean corpus

After normalization, the distance structure **changed** — itself evidence that the original cluster was partly format. The closest pair became **Mymensingh ↔ Rangpur (0.879)**, with a low-distance block of **Faridpur / Mymensingh / Rangpur / Khulna**, and Rajshahi remained an outlier (farthest pair Rajshahi ↔ Rangpur, 1.13). Crucially, this grouping is **geographically plausible** (north-central/northern varieties grouping together), which is the sanity check a dialectal result should pass.

### 7.4 C2 results on the clean corpus — the decisive test

The leave-one-speaker-out region classifier gave the key result:

- **LOSO region accuracy: 0.312 versus chance 0.125 — 2.5× chance** (macro-F1 0.223).

This is a **genuine, above-chance, speaker-independent accent signal**: region can be predicted for unseen speakers well above chance, so the signal is not speaker memorization. However, the per-region recalls show the signal is **modest and highly uneven**:

| Region | LOSO recall (unseen speakers) |
|---|---|
| Sylhet | 0.732 |
| Faridpur | 0.383 |
| Khulna | 0.336 |
| Rajshahi | 0.226 |
| Dhaka | 0.083 |
| Chattogram | 0.060 |
| Mymensingh | 0.000 |
| Rangpur | 0.000 |

A few divergent accents (Sylhet strongly; Faridpur, Khulna, Rajshahi moderately) carry the accuracy, while several regions (Mymensingh, Rangpur, Chattogram, Dhaka) are essentially indistinguishable from unseen speakers in this representation.

The **confusion matrix is linguistically coherent and agrees with C1**: the regions C1 found acoustically close are exactly the regions C2 confuses — Mymensingh is most often misread as Sylhet, Rangpur as Khulna — and Sylhet acts as an "attractor" that other regions are misclassified into, consistent with Sylheti being the most distinct variety. This agreement between two independent analyses indicates the structure is real rather than artifactual.

The whole-corpus and per-sentence silhouettes remained near zero (content-space region silhouette −0.006; per-sentence −0.079). This is **not** in conflict with the positive LOSO result: silhouette asks whether regions form clean clusters filling the space, while LOSO asks whether a decision boundary separates them on unseen speakers. A weak silhouette together with above-chance LOSO is the classic signature of **a real but subtle signal embedded in a space dominated by a stronger factor (speaker)**. The speaker-residualized silhouette was likewise near zero, but this is *expected by design*: because every speaker belongs to one region (speakers are nested in regions), subtracting the speaker mean also removes the region mean, so this metric quantifies the confound rather than disproving accent.

---

## 8. Consolidated findings

The four parts converge on a single, consistent picture.

The corpus is broadly well-formed and balanced (Part A), and two general Bengali ASR systems transcribe it with reasonable, architecture-consistent error rates (Part B, B1). The corpus carries a **genuine but subtle regional-accent signal**: after removing a recording-format confound, a leave-one-speaker-out classifier recovers region from unseen speakers at 2.5× chance (Part C2). That signal is **uneven across regions** — strong for divergent varieties such as Sylhet, weak or absent for several others — and it is **dominated by speaker identity** in the overall acoustic geometry. Two independent analyses (the C1 distance matrix and the C2 confusion matrix) agree on the acoustic neighborhoods between accents, which corroborates that the structure is dialectal.

Equally important, the EDA functioned as a data audit and caught problems that would otherwise have contaminated the results: a **recording-format inconsistency** that split along regional lines and initially masqueraded as accent structure, and **dead/empty recordings** concentrated in specific regions. Both were diagnosed and the former was remediated by normalization.

---

## 9. Data-quality issues and remediation

| Issue | Where found | Status |
|---|---|---|
| Missing parallel cell; Faridpur directory capitalization | Part A completeness check | Corrected by normalization |
| Anomalous Dhaka ASR error (standard dialect "hardest") | Part B (B2) | Diagnosed as data fault, not accent; not reported as "distance from standard" |
| Empty/dead recordings concentrated in Dhaka, Rajshahi, Chattogram, Sylhet, Khulna | Data-quality audit | Inventoried; excluded from scoring; removed in C0 normalization |
| Mixed recording formats (44.1 k/48 k, mono/stereo) split by region | Data-quality audit | Confirmed to confound C1; removed by C0 normalization to 16 kHz mono |
| Recording-format confound in accent embeddings | Part C (C1) | Removed in Part C2 (cleaned corpus) |
| Rajshahi flagged by three independent analyses (B2 anomaly, empty files, C1 outlier) | Parts B, C, C2 | Alignment diagnostic added in C0; recommended to verify before finalizing |

---

## 10. Limitations

The most consequential limitation is structural: the corpus has roughly **six speakers per region and no speaker recorded in more than one region**, so **speaker identity and regional accent are nested and cannot be fully disentangled within the data itself.** The EDA establishes this rigorously (the speaker-residualization collapse), and it means the valid evidence for accent comes from cross-speaker generalization (LOSO), not from subtracting speaker identity. The corpus is also **read** speech (not spontaneous), of moderate scale (≈40 speakers, 250 sentences), and the recovered accent signal — while genuine — is **subtle and uneven** across regions.

The ASR-based region ranking (B2) and the un-normalized accent results (Part C) should be treated as **superseded** by the cleaned-corpus analysis (Part C2) and were retained here primarily because the discrepancies between them are themselves diagnostic.

---

## 11. Implications for the accent-conversion framework

The EDA supports building an accent-conversion framework on this corpus, but the supporting argument is specific and should be stated honestly. The corpus contains **real, speaker-independent regional-accent variation that is recoverable but entangled with speaker identity.** That entanglement is not a weakness of the dataset; it is the **core technical challenge the conversion framework exists to solve** — if accent and speaker were trivially separable, timbre-preserving accent conversion would be unnecessary. The uneven separability across regions is also practically useful: it indicates which conversion *directions* are most tractable (e.g. those involving the strongly separable Sylhet) and which (e.g. the acoustically adjacent Mymensingh–Rangpur) are inherently harder.

The appropriate framing for the thesis is therefore that BPAC is **a novel parallel resource with genuine-but-subtle accent structure that motivates and enables a disentanglement-based conversion approach** — not a corpus that cleanly separates accents. This framing is both more defensible and more compelling for a conversion paper, because it turns the confound into the motivation. The decisive validation of the framework will come from the **conversion-side evaluation** (objective accent/speaker-similarity/content metrics plus a listening test), which this EDA characterizes and motivates but does not itself provide.

---

## Appendix — Models and tools

**ASR models (Part B):** `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` (Whisper, encoder–decoder); `shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice` (XLS-R wav2vec2, CTC).

**Accent/embedding models (Parts C, C2):** `microsoft/wavlm-large` (self-supervised content/accent embeddings; middle layer in Part C2); `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN speaker embeddings).

**Key analyses:** WER/CER scoring (jiwer); silhouette scores and leave-one-speaker-out classification (scikit-learn); hierarchical clustering and dendrograms (scipy); t-SNE/UMAP projections; pitch via Praat (parselmouth); audio I/O and resampling via soundfile and librosa.

**Reproducibility:** every stage is configuration-driven, with uniform preprocessing applied identically across regions (NFC text normalization; 16 kHz mono audio; identical feature and scoring settings), and exact model checkpoints recorded because embedding-based results depend on them.