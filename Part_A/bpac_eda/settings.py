"""Load Part A configuration from config.yml (single source of truth).

Relative paths in the YAML are resolved relative to the YAML file's directory,
so analyses work regardless of the current working directory.
"""
import copy
from pathlib import Path

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise ImportError("pyyaml is required: pip install pyyaml") from e

# Fallback defaults (used if a key is absent from config.yml).
DEFAULTS = {
    "paths": {
        "data_root": "../data",
        "data_clean_root": "../data_clean",
        "verified_manifest": "../verified_manifest.csv",
        "bpac_subdir": "BPAC",
        "sentences_csv": None,
        "reports_dirname": "reports",
    },
    "corpus": {"sentences_filename": "best_diverse_sentences.csv", "n_sentences": 250},
    "audio": {
        "analysis_sr": 16000,
        "default_top_db": 30,
        "pitch_ranges": {"male": [75, 300], "female": [100, 500]},
    },
    "fonts": {
        "bangla_candidates": [
            "Noto Sans Bengali", "Noto Serif Bengali", "Kalpurush", "Siyam Rupali",
            "Mukti", "AdorshoLipi", "Lohit Bengali", "Nikosh", "Vrinda", "Shonar Bangla",
        ]
    },
    "logging": {"verbose": True},
    "a2_audio_length": {"mode": "raw", "top_db": 30},
    "a4_character_count": {"top_chars": 30},
    "a5_word_count": {"top_words": 30},
    "a6_speaking_rate_pitch": {"top_db": 30, "max_per_speaker": None, "compute_f0": True},
}

# bpac_eda/ -> Part_A/  (config.yml lives in Part_A/)
PART_A_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PART_A_DIR / "config.yml"


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
    """Dict with attribute-style accessors for resolved paths/params."""

    @property
    def config_path(self): return self["_resolved"]["config_path"]
    @property
    def data_root(self): return self["_resolved"]["data_root"]
    @property
    def data_clean_root(self): return self["_resolved"]["data_clean_root"]
    @property
    def verified_manifest(self): return self["_resolved"]["verified_manifest"]
    @property
    def sentences_csv(self): return self["_resolved"]["sentences_csv"]
    @property
    def bpac_subdir(self): return self["paths"]["bpac_subdir"]
    @property
    def reports_dirname(self): return self["paths"]["reports_dirname"]
    @property
    def n_sentences(self): return self["corpus"]["n_sentences"]
    @property
    def analysis_sr(self): return self["audio"]["analysis_sr"]
    @property
    def default_top_db(self): return self["audio"]["default_top_db"]
    @property
    def pitch_ranges(self):
        return {k: tuple(v) for k, v in self["audio"]["pitch_ranges"].items()}
    @property
    def bangla_candidates(self): return self["fonts"]["bangla_candidates"]
    @property
    def verbose(self): return bool(self["logging"]["verbose"])

    def reports_dir_for(self, analysis_file):
        """The reports/ directory next to a given analyze.py file."""
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
    data_clean_root = _resolve(merged["paths"].get("data_clean_root", "../data_clean"), base_dir)
    verified_manifest = _resolve(merged["paths"].get("verified_manifest"), base_dir)
    scsv = merged["paths"]["sentences_csv"]
    sentences_csv = (
        _resolve(scsv, base_dir) if scsv
        else (data_root / merged["corpus"]["sentences_filename"])
    )
    merged["_resolved"] = {
        "config_path": config_path,
        "config_dir": base_dir,
        "data_root": data_root,
        "data_clean_root": data_clean_root,
        "verified_manifest": verified_manifest,
        "sentences_csv": sentences_csv,
    }
    cfg = Settings(merged)
    if cfg.verbose:
        print(f"[config] loaded {config_path}")
        print(f"[config] data_root      = {data_root}")
        print(f"[config] sentences_csv  = {sentences_csv}")
    return cfg
