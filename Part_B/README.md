# Part B — Speech-to-Text (STT) Analysis (BPAC)

STT analysis with the two models run **independently**, each producing its own
report, plus a combined cross-model report.

```
<project_root>/
├── Part_B/
│   ├── config.yml                 <- paths + MODELS + inference + scoring
│   ├── bpac_stt/                  <- shared package (settings, manifest, asr, scoring, reporting)
│   ├── wav2vec2_stt/analyze.py    -> reports/   (fast model — run first)
│   ├── whisper_stt/analyze.py     -> reports/   (slow model)
│   ├── create_report.py           -> combined_report/   (combines whatever is cached)
│   ├── cache/predictions.csv      <- shared cache (each model writes its own hyp column)
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── run_all.sh
└── data/                          <- sibling of Part_B/
    ├── BPAC/<region>/speaker_<gender>_<id>_<name>/sound_<NNN>.wav
    └── best_diverse_sentences.csv
```

## Why it's split

Whisper is much slower than wav2vec2. With separate scripts you can run wav2vec2
first, see its WER/CER, and decide before paying for the Whisper pass. Each
script transcribes **only its own model** into the shared cache, so order and
timing are flexible; the corpus is transcribed once per model.

## Models (config.yml)

| key | type | HF id |
|---|---|---|
| `whisper_bengaliai_medium` | whisper (encoder-decoder) | `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` |
| `wav2vec2_xlsr_bengali` | wav2vec2 (CTC) | `shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice` |

## Empty / dead audio

Empty, too-short, or silent files are detected during transcription and flagged
(`is_empty_audio`) in the cache. With `scoring.exclude_empty_audio: true`
(default) they are **excluded from WER/CER**, so dead recordings don't corrupt
the numbers. Each report shows how many were excluded (`n_excluded_empty_audio`).

## Setup

```bash
cd Part_B
pip install -r requirements.txt
# optional: install the wav2vec2 5-gram LM decoder (often lowers WER):
pip install pyctcdecode kenlm
```

> **Low-VRAM GPUs (RTX 3050, 4 GB):** defaults fit — `batch_size: 1`, `fp16: true`,
> greedy decoding, `chunk_length_s: 30` (handles long clips). Transcription is
> **resumable** (checkpoints every `save_every` utterances).

## Run order

```bash
# 1) fast model first — check whether results improved
python wav2vec2_stt/analyze.py

# 2) slow model when ready
python whisper_stt/analyze.py

# 3) combined report (run any time — uses whatever models are in the cache)
python create_report.py

# or all three:
./run_all.sh
```

`create_report.py` works even if only wav2vec2 has run — it reports the single
model and notes the other is missing. Re-run it after Whisper for the full
cross-model comparison.

## Outputs

| Script | Folder | Key outputs |
|---|---|---|
| `wav2vec2_stt/analyze.py` | `wav2vec2_stt/reports/` | `summary.json`, `region_metrics.csv`, `region_metrics.png`, `report.md` |
| `whisper_stt/analyze.py` | `whisper_stt/reports/` | same set, for Whisper |
| `create_report.py` | `combined_report/` | `model_comparison.csv`, `wer_matrix.csv`, `cer_matrix.csv`, `region_ranking.csv`, `region_*_by_model.png`, `report.md` |

## Config knobs (config.yml)

- `models`: list of `{key, type, hf_id, language, label}`.
- `audio.target_sr`: 16000 (16 kHz mono, uniform).
- `inference`: `device`, `batch_size`, `fp16`, `whisper_beam_size`,
  `chunk_length_s`, `whisper_max_new_tokens`, `save_every`, `max_utterances`,
  `force_retranscribe`.
- `scoring`: `normalize` (NFC + punctuation), `remove_punctuation`,
  `primary_metric` (cer|wer), `exclude_empty_audio`.

## Notes

- **Preprocessing:** 16 kHz mono; identical NFC + punctuation normalization on
  reference AND hypothesis; no per-region denoising.
- **Lean on CER** for the region ranking (Bangla orthography makes WER noisier).
- A consistent region ranking across both models is a robust signal; disagreement
  = model-dependence. The ranking should corroborate Part C.
