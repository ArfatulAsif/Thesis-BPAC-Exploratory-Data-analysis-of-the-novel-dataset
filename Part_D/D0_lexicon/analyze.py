#!/usr/bin/env python3
"""T11 -- Bangla pronunciation lexicon.

Extracts the 2,012 unique word types from the 250-sentence script,
generates pronunciations with the rule-based G2P (bpac_phon/g2p.py --
authoritative; see that module's docstring for why epitran/bnlp_toolkit
were not usable here), and writes the MFA-format lexicon plus a manual-
review worklist (top 500 by frequency + random 200 tail, per the spec).

MANUAL REVIEW IS NOT DONE BY THIS SCRIPT. `lexicon_review.csv` ships with
`reviewed=False` for every row -- per the human-in-the-loop policy agreed
for this pipeline (same as T4), this script produces a best-effort
auto-generated lexicon so T12 (forced alignment) can proceed, and clearly
flags that linguistic review is still pending. Re-run T12 after review
corrections land in `lexicon_review.csv`.

Outputs (in ./reports): bpac_lexicon.txt (MFA format), lexicon_review.csv,
phoneset.md (hand-written, see repo), coverage.json, report.md, summary.json.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import unicodedata  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.provenance import base_summary  # noqa: E402

from bpac_phon.cli import add_config_arg  # noqa: E402
from bpac_phon.g2p import PHONE_SET, word_to_phones  # noqa: E402
from bpac_phon.manifest import load_sentences  # noqa: E402
from bpac_phon.reporting import ReportWriter, ensure_dir, save_json, save_table  # noqa: E402
from bpac_phon.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from bpac_phon.textnorm import tokenize


def is_format_char_only(w):
    """True if every character is a Unicode Format character (Cf) -- e.g. a
    stray zero-width non-joiner in the source text, not a real word. MFA
    rejects lexicon entries with no phones, so these must not become
    lexicon/transcript tokens."""
    return len(w) > 0 and all(unicodedata.category(c) == "Cf" for c in w)


def main():
    ap = argparse.ArgumentParser(description="T11 Bangla pronunciation lexicon")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--review_top_n", type=int, default=500)
    ap.add_argument("--review_tail_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))

    sentences = load_sentences(cfg.sentences_csv)
    n_sentences = len(sentences)

    counts = Counter()
    n_format_char_tokens_dropped = 0
    for s in sentences.values():
        for w in tokenize(s):
            if is_format_char_only(w):
                n_format_char_tokens_dropped += 1
                continue
            counts[w] += 1
    n_types = len(counts)
    n_tokens = sum(counts.values())
    if n_format_char_tokens_dropped:
        print(f"[T11] dropped {n_format_char_tokens_dropped} stray Unicode-format-character "
              f"token(s) (e.g. zero-width non-joiner) -- not real words, not written to the lexicon.")
    print(f"[T11] {n_types} unique types, {n_tokens} tokens across {n_sentences} sentences")

    rows = []
    variant_lines = []
    for word, freq in counts.items():
        phones, flags = word_to_phones(word)
        phone_str = " ".join(phones)
        rows.append({
            "word": word, "freq": freq, "auto_g2p": phone_str,
            "has_conjunct": flags["has_conjunct"], "has_yaphala": flags["has_yaphala"],
            "has_raphala": flags["has_raphala"], "has_visarga_dropped": flags["has_visarga_dropped"],
            "medial_schwa_deletion_risk": flags["medial_schwa_deletion_risk"],
            "is_numeral": flags["is_numeral"],
            "needs_numeral_verbalization_review": flags["needs_numeral_verbalization_review"],
            "reviewed": False, "reviewer": "", "review_notes": "", "corrected_g2p": "",
        })
        variant_lines.append(f"{word}\t{phone_str}")

    lex_df = pd.DataFrame(rows).sort_values("freq", ascending=False).reset_index(drop=True)

    # -- MFA lexicon file (word<TAB>phones, one pronunciation per line) --
    lex_path = out / "bpac_lexicon.txt"
    with open(lex_path, "w", encoding="utf-8") as f:
        for _, r in lex_df.iterrows():
            f.write(f"{r['word']}\t{r['auto_g2p']}\n")

    # -- manual review worklist: top-N by frequency + random tail --
    top = lex_df.head(args.review_top_n)
    tail_pool = lex_df.iloc[args.review_top_n:]
    rng = np.random.default_rng(args.seed)
    tail_n = min(args.review_tail_n, len(tail_pool))
    tail = tail_pool.sample(tail_n, random_state=args.seed) if tail_n else tail_pool.iloc[0:0]
    review_df = pd.concat([top, tail]).reset_index(drop=True)
    review_df["in_review_set"] = True
    review_df["review_reason"] = np.where(
        review_df.index < len(top), "top_frequency", "random_tail"
    )
    save_table(review_df, out / "lexicon_review.csv", index=False)
    save_table(lex_df, out / "lexicon_full.csv", index=False)

    n_flagged_risk = int(lex_df["medial_schwa_deletion_risk"].sum())
    n_numeral = int(lex_df["is_numeral"].sum())
    coverage = {
        "n_sentence_tokens": n_tokens,
        "n_unique_types": n_types,
        "n_types_with_pronunciation": int((lex_df["auto_g2p"].str.len() > 0).sum()),
        "coverage_pct": round(100 * (lex_df["auto_g2p"].str.len() > 0).mean(), 3),
        "n_numeral_tokens": n_numeral,
        "numeral_note": "Digit-by-digit fallback reading (not natural numeral verbalization, "
                        "e.g. '2023' read as 'two-zero-two-three' not 'two thousand twenty-three') "
                        "-- flagged needs_numeral_verbalization_review=True for manual correction.",
        "n_flagged_medial_schwa_risk": n_flagged_risk,
        "n_flagged_conjunct": int(lex_df["has_conjunct"].sum()),
        "n_format_char_tokens_dropped": n_format_char_tokens_dropped,
        "phone_set_size": len(PHONE_SET),
    }
    save_json(coverage, out / "coverage.json")

    summary = base_summary("T11_lexicon", cfg.config_path, n_types, n_types, count_unit="word_types")
    summary.update(coverage)
    summary.update({
        "review_top_n": args.review_top_n,
        "review_tail_n": tail_n,
        "n_in_review_set": len(review_df),
        "manual_review_status": "NOT STARTED -- lexicon_review.csv has reviewed=False for every row. "
                                "This is expected: a human must edit that file (see report.md) before "
                                "the lexicon can be called verified. T12 can proceed on the auto-generated "
                                "lexicon in the meantime.",
        "g2p_method": "rule-based (bpac_phon/g2p.py), authoritative -- epitran unusable "
                     "(Python 3.9 incompatibility), bnlp_toolkit has no G2P module.",
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T11 -- Bangla Pronunciation Lexicon")
    rep.p(
        f"**{n_types} unique word types** ({n_tokens} tokens) extracted from the 250-sentence script "
        f"and converted with a rule-based G2P (100% coverage by construction -- see "
        f"`../reports/phoneset.md` for the phone set and every documented simplification). "
        f"**{n_flagged_risk} words are flagged `medial_schwa_deletion_risk`** -- the known weak point "
        f"of the schwa-deletion rule -- and should get extra scrutiny in review."
    )
    rep.h2("Cross-check attempts (per spec: don't rely on a single library)")
    rep.bullet("`epitran` (ben-Beng): installed, but its `panphon` dependency uses Python 3.10+ "
              "syntax and fails to import under this repo's Python 3.9. Not usable here.")
    rep.bullet("`bnlp_toolkit`: installed, has tokenization/NER/POS/embeddings but no G2P/phoneme "
              "module at all.")
    rep.bullet("Rule-based `bpac_phon/g2p.py` is therefore **authoritative**, as the spec allows for "
              "exactly this situation.")
    rep.h2("MANUAL REVIEW -- NOT YET DONE")
    rep.p(
        f"`lexicon_review.csv` contains the top {args.review_top_n} words by frequency + a random "
        f"{tail_n}-word tail sample ({len(review_df)} rows total, covering the large majority of "
        "corpus tokens by frequency-weighting). Every row has `reviewed=False`. **A linguist needs to "
        "open this CSV, check `auto_g2p` against `word`, and fill in `corrected_g2p` + `reviewed=True` "
        "+ `reviewer` + `review_notes` for any wrong entries** (prioritize rows with "
        "`medial_schwa_deletion_risk=True`). This is a genuinely human task -- there is no shortcut "
        "that doesn't risk baking G2P errors into the forced-alignment ground truth."
    )
    rep.h2("Coverage")
    for k, v in coverage.items():
        rep.bullet(f"`{k}` = {v}")
    rep.h2("For the paper")
    rep.p(
        "S7.1 method + release artifact list (S9): a verified 2,012-word Bangla pronunciation "
        "dictionary is a citable contribution in its own right -- cite the coverage numbers above and "
        "state the manual-review sample size once review is complete."
    )
    rep.write()

    print(f"[T11] done. types={n_types}  coverage={coverage['coverage_pct']}%  "
          f"schwa_risk_flagged={n_flagged_risk}  review_set={len(review_df)}")


if __name__ == "__main__":
    main()
