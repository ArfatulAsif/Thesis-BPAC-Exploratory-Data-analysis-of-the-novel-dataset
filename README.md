# Thesis-BPAC-Exploratory-Data-analysis-of-the-novel-dataset


<br>
<br>


## 2 From each `dataset` paper, fill this table:


| # / Dimension | What to record | Example: OOD-Speech |
|---|---|---|
| **Citation** *(added)* | Authors, venue, year, link | Rakib et al., arXiv 2023, 2305.09688 |
| **1. Total corpus length** | Hours of audio + total raw utterances | 1177.94 h; scripted + spontaneous |
| **2. Size & recording format** | Disk size; sample rate; bit depth; channels; file format | Report from paper/repo (often partial) |
| **3. Linguistic density** | Sentence count; token count; unique word types | Per paper |
| **4. Speaker demographics** | Speaker count; gender; age; regional distribution | 22,645 speakers, South Asian |
| **5. Linguistic register** | SCBB vs regional/dialectal | Mixed (scripted standard + OOD spontaneous) |
| **6. Provenance & accessibility** | Institution/authors; license (CC-BY/CC-0/restricted) | Bengali.AI; check license |
| **7. Thematic domain coverage** | Script/source domains (news, talk show, literature…) | 17 domains (test) |
| **8. Parallel matrix structure** | Parallel (same sentences, multiple speakers) vs non-parallel | Non-parallel |
| **9. Accent/dialect axis** *(added)* | Is parallelism across *accents/regions*? Which regions? Accent vs dialect? | No accent axis |
| **10. Modality / annotation** *(added, optional)* | Audio-only vs audio+transcript; alignment, phonetic labels | Audio + transcript |



## 3 Corpus Selection and Text Acquisition

To ensure the text prompts for the parallel accent corpus encompass a broad and representative vocabulary, we compiled a diverse foundational text corpus. We sourced articles and paragraphs across twelve distinct categorical domains:

1. **Bangladesh** (National Affairs)
2. **Business** (Economics and Commerce)
3. **Education**
4. **Entertainment**
5. **Health** (Medicine and Wellness)
6. **International** (Global Affairs)
7. **Literature**
8. **News** (Current Events)
9. **Politics**
10. **Religion**
11. **Science** (Technology and Research)
12. **Sports**

### Sentence Extraction and Subset Generation

From this aggregated text corpus, we segmented the raw text into individual sentences, normalizing the data to remove non-standard characters. From the resulting pool of sentences, we systematically generated 100 distinct subsets, with each subset comprising exactly 400 sentences.

### Lexical Diversity Filtering

To guarantee a phonetically and lexically rich corpus suitable for capturing wide acoustic variations, we evaluated each of the 100 subsets based on their overall word diversity (unique vocabulary count and type-token ratio). The three subsets exhibiting the highest lexical diversity were isolated for further processing.

### Manual Inspection and Final Selection

The top three subsets underwent a rigorous manual inspection to assess sentence naturalness, grammatical correctness, and ease of readability for the recording subjects. Following this review, one optimal subset of 400 sentences was chosen as the strongest candidate. Finally, from this single subset, we manually curated the definitive set of **250 sentences**. This final set serves as the standardized, parallel reading script utilized by all speakers across the different regional dialect groups during the voice recording phase.


<br>
<br>





# 4 Audio collectoin methodolgoy:

- report labrotry settings
- Report software and equipement used
- Report our android app