"""bpac_eda — shared utilities for the BPAC Part A exploratory data analysis.

Exposes manifest building, text normalization, audio helpers, and reporting.
"""
from . import config, textnorm, manifest, audio, reporting, cli, settings
from .settings import load_config, DEFAULT_CONFIG_PATH

__all__ = ["config", "textnorm", "manifest", "audio", "reporting", "cli",
           "settings", "load_config", "DEFAULT_CONFIG_PATH"]
__version__ = "0.1.0"
