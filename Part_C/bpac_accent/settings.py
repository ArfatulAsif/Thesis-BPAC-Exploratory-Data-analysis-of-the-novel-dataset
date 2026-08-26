"""Load Part C configuration from config.yml (single source of truth).

Relative paths resolve relative to the YAML file's directory.
"""
import copy
from pathlib import Path

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise ImportError("pyyaml is required: pip install pyyaml") from e

DEFAULTS = {
    "paths": {"raw_data_root": "../data", "data_root": "../data_clean",
              "bpac_subdir": "BPAC", "sentences_csv": None,
              "reports_dirname": "reports", "cache_dir": "cache",
              "verified_manifest": "../verified_manifest.csv"},
    "corpus": {"sentences_filename": "best_diverse_sentences.csv", "n_sentences": 250},
    "normalize": {"target_sr": 16000, "mono": True, "peak_normalize": True,
                  "peak_dbfs": -1.0, "min_ms": 200, "silence_peak_eps": 1.0e-4,
                  "overwrite": False, "rajshahi_focus": ["rajshahi", "dhaka"],
                  "align_dur_tol": 0.40},
    "audio": {"target_sr": 16000, "top_db": 30, "exclude_empty_audio": True},
    "models": {
        "ssl": {"hf_id": "microsoft/wavlm-large", "layer": "middle", "pooling": "mean"},
        "speaker": {"source": "speechbrain/spkrec-ecapa-voxceleb"},
    },
    "inference": {"device": "auto", "batch_size": 1, "fp16": True,
                  "save_every": 300, "max_utterances": None},
    "c1_distance": {"method": "ssl_controlled", "metric": "cosine", "standardize": True,
                    "per_sentence_center": True, "standard_region": "dhaka",
                    "loso_classifier": "lda", "mfcc_n": 13,
                    "dtw_max_sentences": 60, "dtw_max_speakers_per_region": 4},
    "c2_separability": {"projection": "both", "standardize": True,
                        "residualize_speaker": True, "loso_region_classifier": "lda",
                        "per_sentence_silhouette": True, "tsne_perplexity": 30,
                        "umap_neighbors": 15, "umap_min_dist": 0.1,
                        "sample_per_speaker": None, "random_state": 42},
    "logging": {"verbose": True},
}

PART_C_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PART_C_DIR / "config.yml"


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
    def raw_data_root(self): return self["_resolved"]["raw_data_root"]
    @property
    def data_root(self): return self["_resolved"]["data_root"]
    @property
    def sentences_csv(self): return self["_resolved"]["sentences_csv"]
    @property
    def cache_dir(self): return self["_resolved"]["cache_dir"]
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
    def top_db(self): return self["audio"]["top_db"]
    @property
    def exclude_empty_audio(self): return bool(self["audio"].get("exclude_empty_audio", True))
    # models
    @property
    def ssl_id(self): return self["models"]["ssl"]["hf_id"]
    @property
    def ssl_layer(self): return self["models"]["ssl"].get("layer", -1)
    @property
    def ssl_pooling(self): return self["models"]["ssl"].get("pooling", "mean")
    @property
    def speaker_source(self): return self["models"]["speaker"]["source"]
    # inference
    @property
    def device(self): return self["inference"]["device"]
    @property
    def batch_size(self): return self["inference"]["batch_size"]
    @property
    def fp16(self): return bool(self["inference"]["fp16"])
    @property
    def save_every(self): return self["inference"].get("save_every", 300)
    @property
    def max_utterances(self): return self["inference"].get("max_utterances", None)
    @property
    def verbose(self): return bool(self["logging"]["verbose"])

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
    raw_data_root = _resolve(merged["paths"].get("raw_data_root", "../data"), base_dir)
    data_root = _resolve(merged["paths"]["data_root"], base_dir)
    scsv = merged["paths"]["sentences_csv"]
    sentences_csv = (_resolve(scsv, base_dir) if scsv
                     else data_root / merged["corpus"]["sentences_filename"])
    cache_dir = _resolve(merged["paths"]["cache_dir"], base_dir)
    verified_manifest = _resolve(merged["paths"].get("verified_manifest"), base_dir)
    merged["_resolved"] = {"config_path": config_path, "config_dir": base_dir,
                           "raw_data_root": raw_data_root,
                           "data_root": data_root, "sentences_csv": sentences_csv,
                           "cache_dir": cache_dir, "verified_manifest": verified_manifest}
    cfg = Settings(merged)
    if cfg.verbose:
        print(f"[config] loaded {config_path}")
        print(f"[config] data_root = {data_root}")
        print(f"[config] ssl={cfg.ssl_id}  speaker={cfg.speaker_source}")
    return cfg
