"""bpac_accent — shared utilities for BPAC Part C (v2) accent-structure analysis."""
from . import (settings, manifest, textnorm, normalize, embeddings, distances,
               separability, reporting, cli)
from .settings import load_config, DEFAULT_CONFIG_PATH

__all__ = ["settings", "manifest", "textnorm", "normalize", "embeddings",
           "distances", "separability", "reporting", "cli",
           "load_config", "DEFAULT_CONFIG_PATH"]
__version__ = "0.2.0"
