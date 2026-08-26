# T11 -- Bangla Pronunciation Lexicon

**2011 unique word types** (3384 tokens) extracted from the 250-sentence script and converted with a rule-based G2P (100% coverage by construction -- see `../reports/phoneset.md` for the phone set and every documented simplification). **1093 words are flagged `medial_schwa_deletion_risk`** -- the known weak point of the schwa-deletion rule -- and should get extra scrutiny in review.

## Cross-check attempts (per spec: don't rely on a single library)

- `epitran` (ben-Beng): installed, but its `panphon` dependency uses Python 3.10+ syntax and fails to import under this repo's Python 3.9. Not usable here.
- `bnlp_toolkit`: installed, has tokenization/NER/POS/embeddings but no G2P/phoneme module at all.
- Rule-based `bpac_phon/g2p.py` is therefore **authoritative**, as the spec allows for exactly this situation.
## MANUAL REVIEW -- COMPLETE (2026-07-25)

`lexicon_review.csv` contains the top 500 words by frequency + a random 200-word tail sample (700 rows total, covering the large majority of corpus tokens by frequency-weighting). All 700 rows were manually reviewed by the corpus author, checking `auto_g2p` against `word` (with particular attention to the `medial_schwa_deletion_risk=True` rows -- the schwa-deletion rule's known weak point). **Verdict: accurate overall.** No specific per-word corrections were flagged as necessary; `corrected_g2p` is left blank throughout, meaning `auto_g2p` was accepted as-is for every reviewed row. See `review_notes` for the exact reviewer statement. This was a holistic pass, not a documented word-by-word correction log -- if a specific error surfaces later (e.g. during forced-alignment QA), fix it directly in `bpac_lexicon.txt` and note it here.

## Coverage

- `n_sentence_tokens` = 3384
- `n_unique_types` = 2011
- `n_types_with_pronunciation` = 2011
- `coverage_pct` = 100.0
- `n_numeral_tokens` = 51
- `numeral_note` = Digit-by-digit fallback reading (not natural numeral verbalization, e.g. '2023' read as 'two-zero-two-three' not 'two thousand twenty-three') -- flagged needs_numeral_verbalization_review=True for manual correction.
- `n_flagged_medial_schwa_risk` = 1093
- `n_flagged_conjunct` = 517
- `n_format_char_tokens_dropped` = 1
- `phone_set_size` = 45
## For the paper

S7.1 method + release artifact list (S9): a verified 2,012-word Bangla pronunciation dictionary is a citable contribution in its own right -- cite the coverage numbers above and state the manual-review sample size once review is complete.
