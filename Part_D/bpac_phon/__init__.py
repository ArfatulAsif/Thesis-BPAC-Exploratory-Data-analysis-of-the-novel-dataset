"""bpac_phon -- Part D: the phonetic accent map (BPAC Paper 1, new S7).

Scaffolding only (T1). D0-D7 implementations land as T11-T19 are worked.
All phonetic analysis in this package reads `data_clean/` (16 kHz) except
T16 sibilant analysis, which deliberately reads raw audio resampled to
32 kHz -- see invariants 1-2 in the BPAC Paper 1 spec.
"""
