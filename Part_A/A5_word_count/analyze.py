#!/usr/bin/env python3
"""A5 — Word count with per-word frequency, on the shared sentence set.

All paths/params come from config.yml (a5_word_count.top_words,
fonts.bangla_candidates). Outputs (in ./reports): summary.json,
word_frequency.csv, top_words.png, zipf.png, report.md.
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
from bpac_eda.textnorm import tokenize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="A5 word counts & frequency")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg.reports_dir_for(__file__))
    top_words = cfg["a5_word_count"]["top_words"]
    font = setup_bangla_font(cfg.bangla_candidates)

    sentences = load_sentences(cfg.sentences_csv)
    tokens, per_sentence_len = [], []
    for txt in sentences.values():
        toks = tokenize(txt)
        tokens.extend(toks)
        per_sentence_len.append(len(toks))

    counts = Counter(tokens)
    total_tokens = len(tokens)
    unique_types = len(counts)
    ttr = unique_types / total_tokens if total_tokens else 0.0

    freq_df = (pd.DataFrame(counts.items(), columns=["word", "count"])
               .sort_values("count", ascending=False).reset_index(drop=True))
    freq_df.insert(0, "rank", range(1, len(freq_df) + 1))
    freq_df["pct"] = (freq_df["count"] / total_tokens * 100).round(3)

    plen = pd.Series(per_sentence_len)
    summary = {
        "preprocessing": "NFC; whitespace tokenizer after danda/punctuation strip",
        "n_sentences": len(sentences),
        "total_tokens": int(total_tokens),
        "unique_types": int(unique_types),
        "type_token_ratio": round(ttr, 4),
        "hapax_legomena": int((freq_df["count"] == 1).sum()),
        "words_per_sentence_mean": round(float(plen.mean()), 2),
        "words_per_sentence_min": int(plen.min()),
        "words_per_sentence_max": int(plen.max()),
        "bangla_font_for_figures": font or "NONE (Bangla labels may not render; see CSV)",
    }
    save_json(summary, out / "summary.json")
    save_table(freq_df, out / "word_frequency.csv", index=False)

    top = freq_df.head(top_words)
    labels = top["word"] if font else [f"#{r}" for r in top["rank"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(top)), top["count"], color="indianred")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("count")
    ax.set_title(f"A5 top {top_words} words" + ("" if font else " (rank labels; no Bangla font)"))
    save_fig(fig, out / "top_words.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.loglog(freq_df["rank"], freq_df["count"], marker=".", linestyle="none")
    ax.set_xlabel("rank (log)"); ax.set_ylabel("frequency (log)")
    ax.set_title("A5 word-frequency distribution (Zipf)")
    save_fig(fig, out / "zipf.png")

    rep = ReportWriter(out, "A5 — Word Count & Frequency")
    rep.p("**Preprocessing:** NFC; whitespace tokenizer after stripping danda/punctuation.")
    rep.p("_Computed once on the shared sentence set; identical across all regions._")
    rep.h2("Headline")
    rep.bullet(f"Total tokens: **{summary['total_tokens']}**; unique types: "
               f"**{summary['unique_types']}**; TTR: {summary['type_token_ratio']}.")
    rep.bullet(f"Hapax legomena (count==1): {summary['hapax_legomena']}.")
    rep.bullet(f"Words/sentence: mean {summary['words_per_sentence_mean']} "
               f"(min {summary['words_per_sentence_min']}, max {summary['words_per_sentence_max']}).")
    rep.h2("Top words"); rep.p("Full table: `word_frequency.csv`. Top 20:")
    rep.table(freq_df.head(20), index=False)
    rep.h2("Figures")
    rep.image("top_words.png", f"Top {top_words} words by frequency.")
    rep.image("zipf.png", "Rank-frequency (Zipf) distribution.")
    if not font:
        rep.p("> Note: no Bangla-capable font found; top-words figure uses rank labels. "
              "Exact words are in `word_frequency.csv`.")
    rep.write()

    print(f"[A5] done. tokens={total_tokens} types={unique_types} TTR={summary['type_token_ratio']}")


if __name__ == "__main__":
    main()
