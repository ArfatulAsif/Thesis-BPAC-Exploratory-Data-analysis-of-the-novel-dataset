#!/usr/bin/env python3
"""T12 held-out-region robustness check, step 1 -- split the already-
prepared mfa_corpus/ into mfa_corpus_no_sylhet/ (41 speakers, for training)
and mfa_corpus_sylhet_only/ (10 speakers, for held-out alignment), via
hardlinks (cheap, no duplicated audio on disk -- same approach as
prepare_corpus.py).

Run: python D1_alignment/prepare_heldout_split.py
"""
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "mfa_corpus"
NO_SYLHET = HERE / "mfa_corpus_no_sylhet"
SYLHET_ONLY = HERE / "mfa_corpus_sylhet_only"


def link_speaker_dir(spk_dir: Path, dest_root: Path):
    dst_spk = dest_root / spk_dir.name
    dst_spk.mkdir(parents=True, exist_ok=True)
    for f in spk_dir.iterdir():
        dst_f = dst_spk / f.name
        if dst_f.exists():
            continue
        try:
            os.link(f, dst_f)
        except Exception:
            shutil.copy2(f, dst_f)


def main():
    NO_SYLHET.mkdir(exist_ok=True)
    SYLHET_ONLY.mkdir(exist_ok=True)
    n_sylhet, n_other = 0, 0
    for spk_dir in sorted(SRC.iterdir()):
        if not spk_dir.is_dir():
            continue
        if spk_dir.name.startswith("sylhet"):
            link_speaker_dir(spk_dir, SYLHET_ONLY)
            n_sylhet += 1
        else:
            link_speaker_dir(spk_dir, NO_SYLHET)
            n_other += 1
    print(f"[heldout-prep] {n_other} speakers -> {NO_SYLHET}")
    print(f"[heldout-prep] {n_sylhet} speakers -> {SYLHET_ONLY}")


if __name__ == "__main__":
    main()
