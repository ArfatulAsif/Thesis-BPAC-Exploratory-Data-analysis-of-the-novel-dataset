"""Project-wide constants for the BPAC corpus."""

# Canonical (lowercased) region names. Faridpur is capitalized on disk; we
# always compare/group on the lowercased key (see dataset documentation).
REGIONS = [
    "chattogram",
    "dhaka",
    "faridpur",
    "khulna",
    "mymensingh",
    "rajshahi",
    "rangpur",
    "sylhet",
]

# Number of parallel sentences every speaker reads.
N_SENTENCES = 250

# Per-gender pitch floor/ceiling (Hz) for F0 estimation.
PITCH_RANGES = {
    "male": (75, 300),
    "female": (100, 500),
}

# Default analysis sample rate for any model/audio-based step (format only).
ANALYSIS_SR = 16000

# librosa.effects.trim threshold (dB below peak) used for "speech-only" duration.
DEFAULT_TOP_DB = 30

# Default audio filename pattern and transcript filename.
SENTENCES_FILENAME = "best_diverse_sentences.csv"
