"""Load Part E configuration from config.yml (single source of truth).

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
        "data_root": "../data_clean",
        "bpac_subdir": "BPAC",
        "sentences_csv": None,
        "sibilant_tokens": "../Part_D/D4_consonants/reports/sibilant_tokens.parquet",
        "reports_dirname": "reports",
    },
    "corpus": {"sentences_filename": "best_diverse_sentences.csv", "n_sentences": 250},
    "e1_denoise_check": {
        "n_sample_tokens": 2000,
        "denoiser": "noisereduce",
        "noisereduce_params": {},
    },
    "logging": {"verbose": True},
}

# bpac_valid/ -> Part_E/  (config.yml lives in Part_E/)
PART_E_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PART_E_DIR / "config.yml"


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
    def sentences_csv(self): return self["_resolved"]["sentences_csv"]
    @property
    def sibilant_tokens(self): return self["_resolved"]["sibilant_tokens"]
    @property
    def bpac_subdir(self): return self["paths"]["bpac_subdir"]
    @property
    def reports_dirname(self): return self["paths"]["reports_dirname"]
    @property
    def n_sentences(self): return self["corpus"]["n_sentences"]
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
    scsv = merged["paths"]["sentences_csv"]
    sentences_csv = (
        _resolve(scsv, base_dir) if scsv
        else (data_root / merged["corpus"]["sentences_filename"])
    )
    sibilant_tokens = _resolve(merged["paths"]["sibilant_tokens"], base_dir)
    merged["_resolved"] = {
        "config_path": config_path,
        "config_dir": base_dir,
        "data_root": data_root,
        "sentences_csv": sentences_csv,
        "sibilant_tokens": sibilant_tokens,
    }
    cfg = Settings(merged)
    if cfg.verbose:
        print(f"[config] loaded {config_path}")
        print(f"[config] data_root = {data_root}")
    return cfg
