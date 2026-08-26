#!/usr/bin/env python3
"""T2 -- Manifest schema documentation.

Builds the manifest via Part A's config (raw `data/` tree, so duration_s and
the raw-format probe are meaningful) and documents every column: dtype,
description, and provenance. Also asserts the pipeline/gender confound holds
corpus-wide. Outputs (in ./reports): manifest_schema.md, summary.json,
manifest_sample.csv.
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # common/T2_.../analyze.py -> repo root
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "Part_A"))

from common.provenance import base_summary  # noqa: E402

from bpac_eda.manifest import build_manifest  # noqa: E402
from bpac_eda.reporting import ReportWriter, ensure_dir, save_json, save_table  # noqa: E402
from bpac_eda.settings import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

COLUMN_DOCS = [
    ("utterance_id", "str", "Unique key `<speaker_uid>_<sentence_id:03d>`.", "derived from directory tree + filename"),
    ("filepath", "str", "Absolute path to the .wav clip.", "directory walk"),
    ("region", "str", "Lowercased region folder name (canonical grouping key). One of the 8 BPAC regions.", "directory name (lowercased)"),
    ("region_raw", "str", "Region folder name as stored on disk (original casing, e.g. `Faridpur`).", "directory name (verbatim)"),
    ("gender", "str", "`male` or `female`, parsed from the speaker folder name.", "speaker directory name"),
    ("pipeline", "str", "`lab` or `app` -- the recording pipeline. Perfectly confounded with gender by recruitment design (lab=male, app=female); never treat as separable. See invariant 3 in the BPAC Paper 1 spec.", "derived: gender -> {male: lab, female: app}"),
    ("speaker_id", "str", "Numeric id from the speaker folder name (not globally unique on its own -- combine with region).", "speaker directory name"),
    ("speaker_name", "str", "Speaker's name/handle from the speaker folder name.", "speaker directory name"),
    ("speaker_uid", "str", "Globally unique speaker key `<region>_<gender>_<id>_<name>` (lowercased).", "derived"),
    ("sentence_id", "int", "Script sentence index 1..250, parsed from `sound_NNN.wav`.", "filename"),
    ("sentence", "str", "NFC-normalized sentence text, joined from `best_diverse_sentences.csv` by `sentence_id`.", "joined from sentences CSV"),
    ("duration_s", "float", "Raw clip duration in seconds (header probe, no full decode). Only present when `probe_durations=True`.", "soundfile header probe"),
    ("recording_sr_raw", "int", "Sample rate of the *raw* (pre-normalization) file, so format provenance survives normalization. Only present when `probe_format=True`.", "soundfile/wave header probe"),
    ("channels_raw", "int", "Channel count of the raw file. Only present when `probe_format=True`.", "soundfile/wave header probe"),
]


def main():
    ap = argparse.ArgumentParser(description="T2 manifest schema documentation")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                     help="Path to Part_A/config.yml (used as the reference raw-corpus manifest).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"))

    df = build_manifest(
        cfg.data_root, cfg.bpac_subdir, cfg.sentences_csv,
        probe_durations=True, probe_format=True, verbose=cfg.verbose,
    )
    if df.empty:
        raise SystemExit("No utterances found -- check paths.data_root in config.yml.")

    confound_ok = bool(df.groupby("pipeline")["gender"].nunique().eq(1).all())
    pipeline_counts = df.groupby(["pipeline", "gender"]).size().to_dict()

    dtypes = {c: str(df[c].dtype) for c in df.columns}
    doc_rows = []
    for name, expected_dtype, desc, provenance in COLUMN_DOCS:
        if name not in df.columns:
            continue
        doc_rows.append({
            "column": name,
            "dtype": dtypes[name],
            "description": desc,
            "provenance": provenance,
        })
    import pandas as pd
    schema_df = pd.DataFrame(doc_rows)
    save_table(schema_df, out / "column_schema.csv", index=False)
    save_table(df.head(20), out / "manifest_sample.csv", index=False)

    summary = base_summary("T2_manifest_schema", cfg.config_path, len(df), len(df))
    summary.update({
        "n_columns": int(len(df.columns)),
        "n_speakers": int(df["speaker_uid"].nunique()),
        "n_regions": int(df["region"].nunique()),
        "pipeline_gender_confound_holds": confound_ok,
        "pipeline_gender_counts": {f"{k[0]}/{k[1]}": int(v) for k, v in pipeline_counts.items()},
        "sample_rate_raw_values": sorted(int(x) for x in df["recording_sr_raw"].dropna().unique()),
        "channels_raw_values": sorted(int(x) for x in df["channels_raw"].dropna().unique()),
    })
    save_json(summary, out / "summary.json")

    rep = ReportWriter(out, "T2 -- Manifest Schema")
    rep.p(
        "The BPAC manifest is a single flat table, one row per utterance, built by walking "
        "`data/BPAC/<region>/<speaker>/sound_NNN.wav` and joining sentence text by id. "
        "This report documents every column's dtype and provenance, and verifies the "
        "pipeline/gender confound holds for the whole corpus."
    )
    rep.h2("Columns")
    rep.table(schema_df, index=False)
    rep.h2("Pipeline / gender confound")
    if confound_ok:
        rep.p(
            "Confirmed: every `pipeline` value maps to exactly one `gender` "
            f"({summary['pipeline_gender_counts']}). `pipeline` is fully derivable from "
            "`gender` (lab=male, app=female) by recruitment design -- treat any "
            "within-pipeline analysis as simultaneously within-gender, and vice versa."
        )
    else:
        rep.p("**WARNING**: pipeline/gender confound assumption violated -- investigate before proceeding.")
    rep.h2("Raw recording format")
    rep.bullet(f"raw sample rates observed: {summary['sample_rate_raw_values']}")
    rep.bullet(f"raw channel counts observed: {summary['channels_raw_values']}")
    rep.h2("For the paper")
    rep.p(
        "Section 3.6 (corpus structure) and Section 3.5 (confound disclosure) should cite "
        "this table verbatim for the manifest schema, and state the pipeline/gender confound "
        "using the counts above."
    )
    rep.write("manifest_schema.md")

    print(f"[T2] done. columns={summary['n_columns']}  utterances={summary['n_utterances_used']}  "
          f"confound_ok={confound_ok}")


if __name__ == "__main__":
    main()
