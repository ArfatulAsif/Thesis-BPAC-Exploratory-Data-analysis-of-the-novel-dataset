"""Text normalization and Bangla-aware character/word utilities.

Conventions (state these in any report):
  * All text is Unicode NFC-normalized before counting or scoring.
  * "Character" is reported two ways: Unicode grapheme clusters (human-visible)
    and raw code points.
  * Vowel/consonant split is defined at the *code-point* level:
        vowels      = independent vowels + dependent vowel signs (kar)
        consonants  = consonant letters
        modifiers   = hasanta/virama, nukta, anusvara, visarga, chandrabindu
    (digits, punctuation, whitespace, latin, other are reported separately).
  * Word tokenization is whitespace-based after stripping Bangla danda and
    punctuation. (A Bangla-aware tokenizer such as bnlp is optional.)
"""
import re
import unicodedata

try:
    import regex as _regex  # supports \X grapheme clusters
    _HAS_REGEX = True
except Exception:  # pragma: no cover
    _HAS_REGEX = False


# ----------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------
def nfc(s) -> str:
    """Unicode NFC normalization (mandatory for Bangla counting/scoring)."""
    return unicodedata.normalize("NFC", str(s))


# ----------------------------------------------------------------------------
# Grapheme clusters
# ----------------------------------------------------------------------------
# Bengali combining marks (attach to a preceding base in the naive fallback).
_COMBINING = set(
    list(range(0x0981, 0x0984))      # chandrabindu, anusvara, visarga
    + [0x09BC]                       # nukta
    + list(range(0x09BE, 0x09C5))    # vowel signs aa..vocalic rr
    + list(range(0x09C7, 0x09C9))    # e, ai
    + list(range(0x09CB, 0x09CE))    # o, au, virama (hasanta)
    + [0x09D7]                       # au length mark
    + list(range(0x09E2, 0x09E4))    # vocalic l, ll signs
)


def grapheme_clusters(s: str):
    """Return a list of user-perceived characters (grapheme clusters)."""
    s = nfc(s)
    if _HAS_REGEX:
        return _regex.findall(r"\X", s)
    # Fallback: attach combining marks (and post-virama consonant) to the base.
    out, cur = [], ""
    for ch in s:
        cp = ord(ch)
        if cur and (cp in _COMBINING or unicodedata.combining(ch)):
            cur += ch
        elif cur and cur[-1] == "\u09CD":  # previous ended with hasanta -> conjunct
            cur += ch
        else:
            if cur:
                out.append(cur)
            cur = ch
    if cur:
        out.append(cur)
    return out


# ----------------------------------------------------------------------------
# Code-point classification (Bengali block, U+0980..U+09FF)
# ----------------------------------------------------------------------------
_INDEP_VOWELS = set(
    list(range(0x0985, 0x098D))      # a, aa, i, ii, u, uu, vocalic r, vocalic l
    + list(range(0x098F, 0x0991))    # e, ai
    + list(range(0x0993, 0x0995))    # o, au
    + [0x09E0, 0x09E1]               # vocalic rr, ll
)
_VOWEL_SIGNS = set(
    list(range(0x09BE, 0x09C5))      # aa..vocalic rr signs
    + [0x09C7, 0x09C8]               # e, ai signs
    + [0x09CB, 0x09CC]               # o, au signs
    + [0x09D7]                       # au length mark
    + [0x09E2, 0x09E3]               # vocalic l, ll signs
)
_CONSONANTS = set(
    list(range(0x0995, 0x09A9))      # ka..na
    + list(range(0x09AA, 0x09B1))    # pa..ra
    + [0x09B2]                       # la
    + list(range(0x09B6, 0x09BA))    # sha, ssa, sa, ha
    + [0x09DC, 0x09DD, 0x09DF]       # rra, rha, yya
    + [0x09CE]                       # khanda ta
)
_MODIFIERS = {0x0981, 0x0982, 0x0983, 0x09BC, 0x09CD}  # chandrabindu, anusvara, visarga, nukta, hasanta
_DIGITS = set(range(0x09E6, 0x09F0))
_BANGLA_PUNCT = {0x0964, 0x0965}     # danda, double danda

CATEGORY_ORDER = [
    "independent_vowel",
    "vowel_sign",
    "consonant",
    "modifier",
    "digit",
    "bangla_punct",
    "ascii_punct",
    "whitespace",
    "latin",
    "other",
]


def classify_codepoint(ch: str) -> str:
    cp = ord(ch)
    if cp in _INDEP_VOWELS:
        return "independent_vowel"
    if cp in _VOWEL_SIGNS:
        return "vowel_sign"
    if cp in _CONSONANTS:
        return "consonant"
    if cp in _MODIFIERS:
        return "modifier"
    if cp in _DIGITS:
        return "digit"
    if cp in _BANGLA_PUNCT:
        return "bangla_punct"
    if ch.isspace():
        return "whitespace"
    if ("a" <= ch.lower() <= "z") or ("0" <= ch <= "9"):
        return "latin"
    if not ch.isalnum():
        return "ascii_punct"
    return "other"


def is_vowel(ch: str) -> bool:
    """Vowel = independent vowel OR dependent vowel sign (kar)."""
    return classify_codepoint(ch) in ("independent_vowel", "vowel_sign")


def is_consonant(ch: str) -> bool:
    return classify_codepoint(ch) == "consonant"


# ----------------------------------------------------------------------------
# Word tokenization (whitespace, after stripping danda/punctuation)
# ----------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[।॥…!?,.;:\"'`“”‘’()\[\]{}<>/\\|@#%^&*_+=~\-—–]+")


def tokenize(s: str):
    """Whitespace tokenizer: NFC -> strip danda/punctuation -> split."""
    s = nfc(s)
    s = _PUNCT_RE.sub(" ", s)
    return [t for t in s.split() if t]


def syllable_count_bangla(s: str) -> int:
    """Approximate Bangla syllable count = number of vowel nuclei.

    Heuristic: count independent vowels + vowel signs, and add an inherent
    vowel for consonants not immediately followed by a vowel sign or hasanta.
    This is an estimate (good enough for syllables/sec), not a phonological
    parse.
    """
    s = nfc(s)
    chars = list(s)
    n = 0
    for i, ch in enumerate(chars):
        cat = classify_codepoint(ch)
        if cat in ("independent_vowel", "vowel_sign"):
            n += 1
        elif cat == "consonant":
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            ncat = classify_codepoint(nxt) if nxt else ""
            if ncat not in ("vowel_sign",) and nxt != "\u09CD":
                n += 1  # inherent vowel realized
    return max(n, 0)
