"""Rule-based Bangla grapheme-to-phoneme (G2P) converter (T11).

AUTHORITATIVE by necessity, not by first choice -- the spec asks to cross-
check against `epitran`/`bnlp_toolkit` first:
  * epitran's `panphon` dependency uses Python 3.10+ union-type syntax
    (`str | None`) and fails to import under this repo's Python 3.9.
  * bnlp_toolkit has no G2P/phoneme module at all.
See Part_D/D0_lexicon/reports/phoneset.md for the full phone set and every
documented simplification (schwa deletion, sibilant neutralization, etc.).

This is a character-by-character state machine over NFC-normalized text,
not a dictionary lookup -- it has 100% coverage by construction, at the
cost of being wrong on some words (mainly medial schwa deletion). The
top-500-by-frequency + random-200-tail manual review this module feeds is
where those errors get caught; see D0_lexicon/analyze.py.
"""
import unicodedata

# ---------------------------------------------------------------------------
# Character class tables (Bengali Unicode block, U+0980..U+09FF)
# ---------------------------------------------------------------------------
INDEP_VOWEL = {
    "অ": "O",   # অ
    "আ": "a",   # আ
    "ই": "i",   # ই
    "ঈ": "i",   # ঈ
    "উ": "u",   # উ
    "ঊ": "u",   # ঊ
    "ঋ": "ri",  # ঋ
    "এ": "e",   # এ
    "ঐ": "oi",  # ঐ
    "ও": "o",   # ও
    "ঔ": "ou",  # ঔ
}

VOWEL_SIGN = {  # kar signs -> phone (replaces the consonant's inherent vowel)
    "া": "a",   # া
    "ি": "i",   # ি
    "ী": "i",   # ী
    "ু": "u",   # ু
    "ূ": "u",   # ূ
    "ৃ": "ri",  # ৃ
    "ে": "e",   # ে
    "ৈ": "oi",  # ৈ
    "ো": "o",   # ো
    "ৌ": "ou",  # ৌ
}

CONSONANT = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "c", "ছ": "ch", "জ": "j", "ঝ": "jh", "ঞ": "n",  # ঞ -> n (palatal nasal, rare)
    "ট": "T", "ঠ": "Th", "ড": "D", "ঢ": "Dh", "ণ": "n",  # ণ -> n (retroflex nasal, merges in speech)
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "ph", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j",  # য (antastha ya) -> /j/ realized as "j" (matches জ in most modern speech)
    "র": "r",
    "ল": "l",
    "শ": "s", "ষ": "s", "স": "s",  # শ ষ স -> single canonical /s/, see phoneset.md
    "হ": "h",
    "ড়": "rr", "ঢ়": "rr",  # ড় ঢ়
    "য়": "y",   # য় (antastha ya with nukta) -> /y/ glide
}

HASANTA = "্"
CHANDRABINDU = "ঁ"
ANUSVARA = "ং"
VISARGA = "ঃ"
NUKTA = "়"

NASALIZE = {"a": "aN", "O": "ON", "o": "oN", "i": "iN", "u": "uN", "e": "eN"}

# Bangla digits 0-9 -> their spoken NAME's phones (digit-by-digit reading).
# This is a fallback, not natural numeral verbalization (a real speaker reads
# "2023" as "two thousand twenty-three", not "two zero two three") -- flagged
# via is_numeral / needs_numeral_verbalization_review for manual correction.
DIGIT_NAME_PHONES = {
    "০": ["s", "u", "n", "y", "O"],   # shunyo
    "১": ["e", "k"],                  # ek
    "২": ["d", "u", "i"],             # dui
    "৩": ["t", "i", "n"],             # tin
    "৪": ["c", "a", "r"],             # char
    "৫": ["p", "a", "n", "c"],        # pnach
    "৬": ["ch", "h", "O"],            # chhoy
    "৭": ["s", "a", "t"],             # sat
    "৮": ["a", "T"],                  # aat
    "৯": ["n", "O"],                  # noy
}
BANGLA_DIGITS = set(DIGIT_NAME_PHONES)

PHONE_SET = sorted(set(
    list(INDEP_VOWEL.values()) + list(VOWEL_SIGN.values()) + list(CONSONANT.values())
    + list(NASALIZE.values()) + ["ng", "y", "w"]
))


def _nfc(s):
    return unicodedata.normalize("NFC", s)


def word_to_phones(word: str):
    """Return (phones: list[str], flags: dict) for one word.

    flags surface cases the manual review should look at harder:
      has_conjunct, has_yaphala, has_raphala, has_visarga_dropped,
      medial_schwa_deletion_risk (word > 1 syllable with internal bare
      consonants -- the known weak point of the schwa rule).
    """
    w = _nfc(word)
    n = len(w)
    phones = []
    flags = {"has_conjunct": False, "has_yaphala": False, "has_raphala": False,
             "has_visarga_dropped": False, "medial_schwa_deletion_risk": False,
             "is_numeral": False, "needs_numeral_verbalization_review": False}

    if w and all(c in BANGLA_DIGITS for c in w):
        flags["is_numeral"] = True
        flags["needs_numeral_verbalization_review"] = True
        for c in w:
            phones.extend(DIGIT_NAME_PHONES[c])
        return phones, flags

    i = 0
    n_bare_consonants_not_final = 0
    while i < n:
        ch = w[i]
        nxt = w[i + 1] if i + 1 < n else ""

        if ch in INDEP_VOWEL:
            phones.append(INDEP_VOWEL[ch])
            i += 1
            continue

        if ch in CONSONANT:
            base = CONSONANT[ch]
            if nxt == HASANTA:
                after = w[i + 2] if i + 2 < n else ""
                if after == "য":  # ্য ya-phala
                    phones.append(base)
                    phones.append("y")
                    flags["has_yaphala"] = True
                    i += 3
                    continue
                if after == "র":  # ্র ra-phala
                    phones.append(base)
                    phones.append("r")
                    flags["has_raphala"] = True
                    i += 3
                    continue
                # generic conjunct: this consonant contributes no vowel, continue to next
                phones.append(base)
                flags["has_conjunct"] = True
                i += 2
                continue
            if nxt in VOWEL_SIGN:
                phones.append(base)
                phones.append(VOWEL_SIGN[nxt])
                i += 2
                continue
            if nxt == CHANDRABINDU:
                phones.append(base)
                phones.append(NASALIZE.get("O", "ON"))  # nasalized inherent vowel
                i += 2
                continue
            # bare consonant, no vowel sign/hasanta/chandrabindu following:
            # word-final (ignoring trailing visarga) -> schwa deletion; else inherent vowel O
            rest = w[i + 1:].lstrip(VISARGA)
            is_final = (rest == "")
            phones.append(base)
            if not is_final:
                phones.append("O")
                n_bare_consonants_not_final += 1
            i += 1
            continue

        if ch == CHANDRABINDU:
            if phones and phones[-1] in NASALIZE:
                phones[-1] = NASALIZE[phones[-1]]
            i += 1
            continue
        if ch == ANUSVARA:
            phones.append("ng")
            i += 1
            continue
        if ch == VISARGA:
            flags["has_visarga_dropped"] = True
            i += 1
            continue
        if ch == NUKTA:
            i += 1
            continue
        # whitespace/digits/punctuation/other -- skip (word-level function assumes
        # pre-tokenized, punctuation-stripped input, but stay robust)
        i += 1

    if n_bare_consonants_not_final >= 1 and len(phones) > 3:
        flags["medial_schwa_deletion_risk"] = True

    return phones, flags


def g2p(word: str) -> str:
    """Space-separated phone string for one word (MFA lexicon RHS)."""
    phones, _ = word_to_phones(word)
    return " ".join(phones)
