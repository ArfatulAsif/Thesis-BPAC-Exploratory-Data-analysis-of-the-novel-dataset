#!/usr/bin/env python3
"""A4 — Character count (vowel / consonant), on the shared sentence set.

All paths/params come from config.yml (a4_character_count.top_chars,
fonts.bangla_candidates). Outputs (in ./reports): summary.json,
category_counts.csv, char_frequency.csv, category_counts.png,
top_characters.png, report.md.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bpac_eda.cli import add_config_arg  # noqa: E402
from bpac_eda.manifest import load_sentences  # noqa: E402
from bpac_eda.reporting import (ReportWriter, ensure_dir, save_fig, save_json,  # noqa: E402
                                save_table, setup_bangla_font)
from bpac_eda.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from bpac_eda.textnorm import CATEGORY_ORDER, classify_codepoint, grapheme_clusters, nfc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="A4 character counts")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    top_chars = cfg["a4_character_count"]["top_chars"]
    font = setup_bangla_font(cfg.bangla_candidates)

    sentences = load_sentences(cfg.sentences_csv)
    text = nfc(" ".join(sentences.values()))

    cat_counts, char_counts = Counter(), Counter()
    for ch in text:
        if ch == " ":
            cat_counts["whitespace"] += 1
            continue
        cat_counts[classify_codepoint(ch)] += 1
        char_counts[ch] += 1

    n_codepoints = sum(1 for ch in text if not ch.isspace())
    n_graphemes = sum(1 for g in grapheme_clusters(text) if g.strip())
    vowels = cat_counts.get("independent_vowel", 0) + cat_counts.get("vowel_sign", 0)
    consonants = cat_counts.get("consonant", 0)

    cat_df = pd.DataFrame({"category": CATEGORY_ORDER,
                           "count": [cat_counts.get(c, 0) for c in CATEGORY_ORDER]})
    cat_df["pct_of_nonspace"] = (cat_df["count"] / max(n_codepoints, 1) * 100).round(2)
    char_df = (
        pd.DataFrame([(ch, f"U+{ord(ch):04X}", classify_codepoint(ch), cnt)
                      for ch, cnt in char_counts.items()],
                     columns=["char", "codepoint", "category", "count"])
        .sort_values("count", ascending=False).reset_index(drop=True)
    )

    summary = {
        "preprocessing": "NFC; code-point level; vowels = independent vowels + vowel signs (kar)",
        "n_sentences": len(sentences),
        "n_codepoints_nonspace": int(n_codepoints),
        "n_grapheme_clusters": int(n_graphemes),
        "n_vowels": int(vowels),
        "n_consonants": int(consonants),
        "vowel_consonant_ratio": round(vowels / consonants, 4) if consonants else None,
        "n_unique_chars": int(len(char_counts)),
        "bangla_font_for_figures": font or "NONE (Bangla labels may not render; see CSVs)",
    }
    save_json(summary, out / "summary.json")
    save_table(cat_df, out / "category_counts.csv", index=False)
    save_table(char_df, out / "char_frequency.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cat_df["category"], cat_df["count"], color="steelblue")
    ax.set_ylabel("count"); ax.set_title("A4 character category counts (shared text, NFC)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save_fig(fig, out / "category_counts.png")

    top = char_df.head(top_chars)
    labels = top["char"] if font else top["codepoint"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(top)), top["count"], color="seagreen")
    ax.set_xticks(range(len(top))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("count")
    ax.set_title(f"A4 top {top_chars} characters" + ("" if font else " (codepoints; no Bangla font)"))
    save_fig(fig, out / "top_characters.png")

    rep = ReportWriter(out, "A4 — Character Count (Vowel / Consonant)")
    rep.p("**Preprocessing:** NFC. Counts at the Unicode code-point level. Vowels = "
          "independent vowels + dependent vowel signs (kar); consonants = consonant "
          "letters; hasanta/nukta/anusvara/visarga/chandrabindu = modifiers.")
    rep.p("_Computed once on the shared sentence set; identical across all regions._")
    rep.h2("Headline")
    rep.bullet(f"Sentences: {summary['n_sentences']}")
    rep.bullet(f"Non-space code points: {summary['n_codepoints_nonspace']}; "
               f"grapheme clusters: {summary['n_grapheme_clusters']}")
    rep.bullet(f"Vowels: {summary['n_vowels']}; consonants: {summary['n_consonants']}; "
               f"V/C ratio: {summary['vowel_consonant_ratio']}")
    rep.bullet(f"Unique characters: {summary['n_unique_chars']}")
    rep.h2("Category breakdown"); rep.table(cat_df, index=False)
    rep.h2("Figures")
    rep.image("category_counts.png", "Character categories.")
    rep.image("top_characters.png", f"Top {top_chars} characters by frequency.")
    if not font:
        rep.p("> Note: no Bangla-capable font found; figure labels fall back to code "
              "points. Exact characters are in `char_frequency.csv`.")
    rep.write()

    print(f"[A4] done. vowels={vowels} consonants={consonants} "
          f"ratio={summary['vowel_consonant_ratio']} font={font}")


if __name__ == "__main__":
    main()
