"""Shared code used by every Part (bpac_eda, bpac_stt, bpac_accent, bpac_phon, bpac_valid).

Kept dependency-light (stdlib + pandas only) so importing it never pulls in a
Part's heavier requirements (torch, librosa, parselmouth, ...).
"""
