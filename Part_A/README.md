# Part A — Corpus Descriptive EDA (BPAC)

Packaged analyses for **Part A** of the BPAC exploratory data analysis. Each
sub-analysis lives in its own folder, runs independently, and writes all outputs
(CSVs, figures, `report.md`, `summary.json`) into its own `reports/` subfolder.

**All paths and hyperparameters live in `config.yml`** — the single source of
truth. Scripts take only `--config` (and default to `Part_A/config.yml`).

```
<project_root>/
├── Part_A/                       <- this package
│   ├── config.yml                <- ALL paths + hyperparameters
│   ├── bpac_eda/                 <- shared package (settings, manifest, text, audio, reporting)
│   ├── A1_unique_sentences/analyze.py      -> reports/
│   ├── A2_audio_length/analyze.py          -> reports/
│   ├── A3_utterance_count/analyze.py       -> reports/
│   ├── A4_character_count/analyze.py       -> reports/
│   ├── A5_word_count/analyze.py            -> reports/
│   ├── A6_speaking_rate_pitch/analyze.py   -> reports/
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── run_all.sh
└── data/                         <- sibling of Part_A/
    ├── BPAC/<region>/speaker_<gender>_<id>_<name>/sound_<NNN>.wav
    └── best_diverse_sentences.csv            # columns: id, sentence (250 rows)
```

`sound_<NNN>.wav` links to sentence `id == int(NNN)` (the parallel mapping).

## config.yml controls everything

Relative paths in `config.yml` are resolved **relative to `config.yml` itself**
(i.e. `Part_A/`), so the scripts work from any working directory. Because
`Part_A/` and `data/` are siblings, the default is:

```yaml
paths:
  data_root: "../data"          # -> <project_root>/data
  bpac_subdir: "BPAC"
  sentences_csv: null           # null -> <data_root>/best_diverse_sentences.csv
  reports_dirname: "reports"
corpus:
  n_sentences: 250
audio:
  analysis_sr: 16000
  default_top_db: 30
  pitch_ranges: { male: [75, 300], female: [100, 500] }
a2_audio_length: { mode: "raw", top_db: 30 }          # raw | speech
a4_character_count: { top_chars: 30 }
a5_word_count: { top_words: 30 }
a6_speaking_rate_pitch: { top_db: 30, max_per_speaker: null, compute_f0: true }
```

To change anything (data location, F0 ranges, A6 sampling, top-N, etc.) edit
`config.yml` — **do not** pass CLI flags for parameters.

## Setup

```bash
cd Part_A
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
# optional, to import bpac_eda from anywhere:
pip install -e .
```

Scripts add `Part_A/` to `sys.path` automatically, so they run without
`pip install -e .`.

> **Bangla figure labels:** install a Bangla font (e.g. *Noto Sans Bengali*) to
> render glyphs in the A4/A5 bar charts; otherwise figures fall back to code
> points / ranks (the exact Bangla text is always in the CSVs). The font search
> list is `fonts.bangla_candidates` in `config.yml`.
>
> **F0 (A6):** uses Praat via `praat-parselmouth`; falls back to `librosa.pyin`.

## Run commands

With `config.yml` in place (and `data/` as a sibling of `Part_A/`), just run:

```bash
python A1_unique_sentences/analyze.py
python A2_audio_length/analyze.py
python A3_utterance_count/analyze.py
python A4_character_count/analyze.py
python A5_word_count/analyze.py
python A6_speaking_rate_pitch/analyze.py      # reads every file; slow (see max_per_speaker)
```

Use a different config file with `--config`:

```bash
python A1_unique_sentences/analyze.py --config /path/to/other.yml
```

Run everything:

```bash
chmod +x run_all.sh
./run_all.sh                 # uses ./config.yml
./run_all.sh /path/to/other.yml
```

## Outputs per analysis

| Analysis | Key outputs (in `reports/`) |
|---|---|
| A1 | `summary.json`, `per_speaker_completeness.csv`, `missing_cells.csv`, `coverage_heatmap.png` |
| A2 | `summary.json`, `per_region_duration.csv`, `per_speaker_duration.csv`, `per_gender_duration.csv`, `region_gender_duration.csv`, `per_speaker_duration.png`, `per_region_duration.png` |
| A3 | `summary.json`, `per_region_counts.csv`, `per_speaker_counts.csv`, `region_gender_counts.csv`, `speakers_per_region.csv`, `per_region_counts.png` |
| A4 | `summary.json`, `category_counts.csv`, `char_frequency.csv`, `category_counts.png`, `top_characters.png` |
| A5 | `summary.json`, `word_frequency.csv`, `top_words.png`, `zipf.png` |
| A6 | `summary.json`, `utterance_metrics.csv`, `per_region_gender.csv`, `wpm_by_region.png`, `f0_by_region.png` |

Each analysis also writes a human-readable `report.md`.

## Conventions

- **Text:** Unicode NFC before counting/scoring. "Character" = grapheme clusters
  and code points. Vowels = independent vowels + vowel signs (kar); consonants =
  consonant letters; rest reported as separate categories.
- **Audio length:** `a2_audio_length.mode` is `raw` (header) or `speech`
  (silence-trimmed); A2 states which it used.
- **F0 (A6):** per-gender floor/ceiling from `audio.pitch_ranges`; always read
  region F0 differences **within** a gender (gender dominates F0).
- **Uniformity:** identical processing for every region; raw audio is never
  modified (everything is computed on the fly).
