# T4 -- Alignment Resolution

All 300 T3-flagged utterances have been reviewed. Final alignment verification rate: **99.773%**.

## Headline numbers

- `n_flagged_by_t3` = 300
- `n_resolution_rows` = 523
- `n_corrected` = 29
- `n_excluded_by_review` = 0
- `n_ok` = 483
- `n_unsure` = 0
- `n_still_pending_review` = 0
- `alignment_verification_rate` = 0.99773
- `provisional` = False
## Reviewer agreement

{'n_reviewers': 1, 'n_shared_utterances': 0, 'cohen_kappa': None, 'note': 'Need >=2 reviewers with overlapping utterance_ids (see spotcheck.py --sample 0.2) for a real kappa; null until then.'}

## How to review

```
cd Part_B/alignment_verification
python spotcheck.py --reviewer <your_name>
# optional second opinion on a 20% sample, for the kappa check:
python spotcheck.py --reviewer <second_name> --sample 0.2
python resolve.py --config ../config.yml
```

## For the paper

S3.6: cite `alignment_verification_rate` as the corpus's alignment-verification rate ONLY once `provisional=false`. `verified_manifest.csv` (repo root) is the canonical manifest every downstream task (T5+, all of Part D) should read.
