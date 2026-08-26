#!/usr/bin/env python3
"""Build an anonymized, Hugging-Face-ready release copy of BPAC.

Reads verified_manifest.csv (the corrected, fully-reviewed manifest --
0 excluded, 0 needs_review), strips speaker names from every identifier
(consent forms permit publishing voice + district/region + gender + age,
but explicitly bar publishing participant names), and copies data_clean/
audio into HF_Release/BPAC/data/<region>/<anon_utterance_id>.wav via
hardlink (falls back to copy2 across filesystems), alongside a
metadata.csv the `datasets` library's audiofolder loader reads
automatically.

Run: python build_hf_release.py
"""
import os
import shutil

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(REPO_ROOT, "HF_Release", "BPAC")
OUT_DATA = os.path.join(OUT_ROOT, "data")


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def main():
    vm = pd.read_csv(os.path.join(REPO_ROOT, "verified_manifest.csv"), encoding="utf-8-sig")
    assert vm["excluded"].sum() == 0 and vm["needs_review"].sum() == 0, "manifest not fully clean"

    # NOTE: stripping the trailing "_<name>" from speaker_uid is not safe on
    # its own -- the corpus's own speaker_id numbering has one pre-existing
    # duplicate (sylhet/female has two speakers both originally numbered 002:
    # "anika" and "shusmita"; the name was the only thing disambiguating them
    # under the old scheme). Re-derive a fresh, collision-free sequential
    # anonymous id per (region, gender) group instead of trusting the
    # original numeric speaker_id field.
    uid_table = vm[["region", "gender", "speaker_uid", "speaker_id"]].drop_duplicates()
    uid_table = uid_table.sort_values(["region", "gender", "speaker_id", "speaker_uid"])
    uid_table["anon_speaker_id"] = (
        uid_table.groupby(["region", "gender"]).cumcount() + 1
    )
    uid_table["anon_speaker_id"] = (
        uid_table["region"] + "_" + uid_table["gender"] + "_"
        + uid_table["anon_speaker_id"].astype(str).str.zfill(3)
    )
    assert uid_table["anon_speaker_id"].is_unique, "anonymization still collides"
    assert uid_table["speaker_uid"].is_unique

    vm = vm.merge(uid_table[["speaker_uid", "anon_speaker_id"]], on="speaker_uid", how="left")
    vm["anon_utterance_id"] = vm["anon_speaker_id"] + "_" + vm["sentence_id"].astype(int).astype(str).str.zfill(3)

    os.makedirs(OUT_DATA, exist_ok=True)
    rows = []
    n_linked, n_missing = 0, 0
    for r in vm.itertuples(index=False):
        region_dir = os.path.join(OUT_DATA, r.region)
        os.makedirs(region_dir, exist_ok=True)

        raw_path = r.filepath
        clean_path = raw_path.replace(
            os.sep + "data" + os.sep, os.sep + "data_clean" + os.sep
        )
        if not os.path.exists(clean_path):
            n_missing += 1
            continue

        dst_name = f"{r.anon_utterance_id}.wav"
        dst_path = os.path.join(region_dir, dst_name)
        link_or_copy(clean_path, dst_path)
        n_linked += 1

        rows.append({
            "file_name": f"{r.region}/{dst_name}",
            "utterance_id": r.anon_utterance_id,
            "speaker_id": r.anon_speaker_id,
            "region": r.region,
            "gender": r.gender,
            "pipeline": r.pipeline,
            "sentence_id": int(r.sentence_id),
            "sentence": r.sentence,
        })

    meta = pd.DataFrame(rows)
    meta = meta.sort_values(["region", "speaker_id", "sentence_id"]).reset_index(drop=True)
    meta.to_csv(os.path.join(OUT_ROOT, "metadata.csv"), index=False, encoding="utf-8-sig")

    print(f"[hf-release] linked {n_linked} files, {n_missing} missing from data_clean/")
    print(f"[hf-release] metadata.csv rows: {len(meta)}")
    print(f"[hf-release] unique anonymized speakers: {meta['speaker_id'].nunique()}")
    print(f"[hf-release] output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
