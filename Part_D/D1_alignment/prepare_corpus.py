#!/usr/bin/env python3
"""T12 step 1 -- prepare the MFA corpus directory.

MFA expects `speaker/utt.wav` + `speaker/utt.lab` (transcript). This
hardlinks (Windows has no unprivileged symlinks in this environment --
verified: os.symlink fails with WinError 1314 without Developer Mode/admin;
os.link/hardlinks work and are used instead, same effect: no duplicated
audio on disk) `data_clean/` audio into that layout and writes `.lab` files
with the SAME tokenization used to build the T11 lexicon, so OOV is
guaranteed 0 against `bpac_lexicon.txt`.

Run: python D1_alignment/prepare_corpus.py --config config.yml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import pandas as pd  # noqa: E402

from bpac_phon.cli import add_config_arg  # noqa: E402
from bpac_phon.manifest import build_manifest  # noqa: E402
from bpac_phon.reporting import ensure_dir, maybe_tqdm, save_json  # noqa: E402
from bpac_phon.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from bpac_phon.textnorm import tokenize

import unicodedata


def is_format_char_only(w):
    """Same filter as D0_lexicon/analyze.py -- stray zero-width/format
    characters are not real words and must not appear in .lab transcripts
    (they have no lexicon entry -> would be OOV)."""
    return len(w) > 0 and all(unicodedata.category(c) == "Cf" for c in w)


def main():
    ap = argparse.ArgumentParser(description="T12 -- prepare MFA corpus directory")
    add_config_arg(ap, DEFAULT_CONFIG_PATH)
    ap.add_argument("--out", default=None, help="MFA corpus dir (default: D1_alignment/mfa_corpus)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    corpus_dir = ensure_dir(args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "mfa_corpus"))

    clean_mani = build_manifest(cfg.data_root, cfg.bpac_subdir, None, verbose=False)
    if os.path.exists(cfg.verified_manifest):
        vm = pd.read_csv(cfg.verified_manifest, encoding="utf-8-sig")
        vm = vm[~vm["excluded"].astype(bool)]
        df = clean_mani[["utterance_id", "filepath", "speaker_uid"]].merge(
            vm[["utterance_id", "sentence", "sentence_id"]], on="utterance_id", how="inner",
        )
    else:
        print("[T12-prep] WARNING verified_manifest.csv not found -- using data_clean/ manifest as-is.")
        df = clean_mani
    df = df.dropna(subset=["sentence"]).reset_index(drop=True)

    n_linked, n_link_failed, n_lab = 0, 0, 0
    for r in maybe_tqdm(df.itertuples(index=False), desc="prep MFA corpus", total=len(df)):
        spk_dir = ensure_dir(corpus_dir / r.speaker_uid)
        dst_wav = spk_dir / f"{r.utterance_id}.wav"
        dst_lab = spk_dir / f"{r.utterance_id}.lab"
        if not dst_wav.exists():
            try:
                os.link(r.filepath, dst_wav)
                n_linked += 1
            except Exception as e:
                # cross-volume or filesystem without hardlink support -> fall back to copy
                import shutil
                try:
                    shutil.copy2(r.filepath, dst_wav)
                    n_linked += 1
                except Exception as e2:
                    print(f"[T12-prep] failed to link/copy {r.filepath}: {e2}")
                    n_link_failed += 1
                    continue
        toks = [t for t in tokenize(r.sentence) if not is_format_char_only(t)]
        dst_lab.write_text(" ".join(toks), encoding="utf-8")
        n_lab += 1

    summary = {
        "corpus_dir": str(corpus_dir), "n_utterances": len(df), "n_speakers": df["speaker_uid"].nunique(),
        "n_linked": n_linked, "n_link_failed": n_link_failed, "n_lab_written": n_lab,
        "link_method": "hardlink (os.link), falling back to copy if unsupported",
    }
    save_json(summary, ensure_dir(corpus_dir.parent / "reports") / "corpus_prep_summary.json")
    print(f"[T12-prep] done. {summary}")


if __name__ == "__main__":
    main()
