"""Load Part B configuration from config.yml (single source of truth).

Relative paths resolve relative to the YAML file's directory.
"""
import copy
from pathlib import Path

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise ImportError("pyyaml is required: pip install pyyaml") from e

DEFAULTS = {
    "paths": {
        "data_root": "../data",
        "bpac_subdir": "BPAC",
        "sentences_csv": None,
        "reports_dirname": "reports",
        "predictions_cache": "cache/predictions.csv",
        "duration_suspect_cells": "../Part_C/C0_Normalize_corpus/reports/alignment_suspect_cells.csv",
        "verified_manifest": "../verified_manifest.csv",
    },
    "corpus": {"sentences_filename": "best_diverse_sentences.csv", "n_sentences": 250},
    "audio": {"target_sr": 16000},
    "models": [
        {"key": "whisper_bengaliai_medium", "type": "whisper",
         "hf_id": "bengaliAI/tugstugi_bengaliai-asr_whisper-medium", "language": "bn",
         "label": "Whisper-medium (Bengali.AI)"},
        {"key": "wav2vec2_xlsr_bengali", "type": "wav2vec2_ctc",
         "hf_id": "shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice", "language": None,
         "label": "XLS-R wav2vec2 (CTC)"},
    ],
    "inference": {"device": "auto", "batch_size": 1, "fp16": True, "whisper_beam_size": 1,
                  "whisper_max_new_tokens": None, "chunk_length_s": 30, "save_every": 200,
                  "max_utterances": None, "force_retranscribe": False},
    "scoring": {"normalize": True, "remove_punctuation": True, "primary_metric": "cer",
                "exclude_empty_audio": True},
    "t3_alignment_verification": {
        "primary_model": "whisper_bengaliai_medium",
        "confirm_model": "wav2vec2_xlsr_bengali",
        "offdiag_cer_margin": 0.25,
        "highcer_percentile": 99,
        "spotcheck_worklist_top_n": 300,
        "known_suspect_sentences": [1, 3, 22, 63, 83, 85, 103, 105],
    },
    "logging": {"verbose": True},
}

# bpac_stt/ -> Part_B/  (config.yml lives in Part_B/)
PART_B_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PART_B_DIR / "config.yml"


def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve(path_str, base_dir):
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p).resolve()


class Settings(dict):
    @property
    def config_path(self): return self["_resolved"]["config_path"]
    @property
    def data_root(self): return self["_resolved"]["data_root"]
    @property
    def sentences_csv(self): return self["_resolved"]["sentences_csv"]
    @property
    def predictions_cache(self): return self["_resolved"]["predictions_cache"]
    @property
    def duration_suspect_cells(self): return self["_resolved"]["duration_suspect_cells"]
    @property
    def verified_manifest(self): return self["_resolved"]["verified_manifest"]
    @property
    def bpac_subdir(self): return self["paths"]["bpac_subdir"]
    @property
    def reports_dirname(self): return self["paths"]["reports_dirname"]
    @property
    def n_sentences(self): return self["corpus"]["n_sentences"]
    @property
    def target_sr(self): return self["audio"]["target_sr"]
    @property
    def models(self): return self["models"]
    @property
    def device(self): return self["inference"]["device"]
    @property
    def batch_size(self): return self["inference"]["batch_size"]
    @property
    def whisper_beam_size(self): return self["inference"]["whisper_beam_size"]
    @property
    def fp16(self): return bool(self["inference"].get("fp16", True))
    @property
    def whisper_max_new_tokens(self): return self["inference"].get("whisper_max_new_tokens", None)
    @property
    def chunk_length_s(self): return self["inference"].get("chunk_length_s", 30)
    @property
    def save_every(self): return self["inference"].get("save_every", 200)
    @property
    def max_utterances(self): return self["inference"]["max_utterances"]
    @property
    def force_retranscribe(self): return bool(self["inference"]["force_retranscribe"])
    @property
    def normalize(self): return bool(self["scoring"]["normalize"])
    @property
    def remove_punctuation(self): return bool(self["scoring"]["remove_punctuation"])
    @property
    def primary_metric(self): return self["scoring"]["primary_metric"]
    @property
    def exclude_empty_audio(self): return bool(self["scoring"].get("exclude_empty_audio", True))
    @property
    def verbose(self): return bool(self["logging"]["verbose"])
    # T3
    @property
    def t3(self): return self["t3_alignment_verification"]

    def reports_dir_for(self, analysis_file):
        return Path(analysis_file).resolve().parent / self.reports_dirname


def load_config(config_path=None):
    config_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    user = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    else:
        print(f"[config] WARNING {config_path} not found; using built-in defaults.")
    merged = _deep_merge(DEFAULTS, user)

    base_dir = config_path.parent
    data_root = _resolve(merged["paths"]["data_root"], base_dir)
    scsv = merged["paths"]["sentences_csv"]
    sentences_csv = (_resolve(scsv, base_dir) if scsv
                     else data_root / merged["corpus"]["sentences_filename"])
    pred_cache = _resolve(merged["paths"]["predictions_cache"], base_dir)
    dur_suspect = _resolve(merged["paths"].get("duration_suspect_cells"), base_dir)
    verified_manifest = _resolve(merged["paths"].get("verified_manifest"), base_dir)
    merged["_resolved"] = {
        "config_path": config_path, "config_dir": base_dir,
        "data_root": data_root, "sentences_csv": sentences_csv,
        "predictions_cache": pred_cache,
        "duration_suspect_cells": dur_suspect,
        "verified_manifest": verified_manifest,
    }
    cfg = Settings(merged)
    if cfg.verbose:
        print(f"[config] loaded {config_path}")
        print(f"[config] data_root         = {data_root}")
        print(f"[config] sentences_csv     = {sentences_csv}")
        print(f"[config] predictions_cache = {pred_cache}")
        print(f"[config] models            = {[m['key'] for m in cfg.models]}")
    return cfg


def select_model(cfg, type_=None, key=None):
    """Pick a model from cfg.models by key (preferred) or by type."""
    for m in cfg.models:
        if key and m.get("key") == key:
            return m
    for m in cfg.models:
        if type_ and m.get("type") == type_:
            return m
    raise SystemExit(f"No model with key={key!r} / type={type_!r} found in config.models "
                     f"(have: {[m.get('key') for m in cfg.models]}).")
