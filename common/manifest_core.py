"""Shared manifest-building core (BPAC Paper 1, T2).

`bpac_eda.manifest`, `bpac_stt.manifest`, and `bpac_accent.manifest` used to
be three near-identical copies of this logic. They now wrap this single
implementation so the manifest schema (including the `pipeline` column added
here) cannot drift between Parts.

One row per utterance, walked from `data_root/bpac_subdir/<region>/<speaker>/sound_NNN.wav`:

    utterance_id, filepath, region, region_raw, gender, pipeline,
    speaker_id, speaker_name, speaker_uid, sentence_id, sentence
    (+ duration_s if probe_durations, + recording_sr_raw/channels_raw if probe_format)
"""
import re
from pathlib import Path

import pandas as pd

SPEAKER_RE = re.compile(
    r"^speaker_(?P<gender>male|female)_(?P<id>\d+)_(?P<name>.+)$", re.IGNORECASE
)
SOUND_RE = re.compile(r"^sound_(?P<num>\d+)\.wav$", re.IGNORECASE)

# Recruitment constraint, not a measurement: the SUST lab rig recorded only
# male speakers and the SUST Kotha Android app recorded only female speakers,
# so `pipeline` is fully derivable from `gender`. Keeping pipeline as an
# explicit column (rather than leaving it implicit in gender) forces this
# confound into the open in every downstream table -- see invariant 3 in the
# BPAC Paper 1 spec: never claim to separate pipeline from gender.
PIPELINE_GENDER_CONFOUND = True
GENDER_TO_PIPELINE = {"male": "lab", "female": "app"}


def load_sentences(sentences_csv, nfc) -> dict:
    """Load id -> NFC-normalized sentence from the transcript CSV.

    `nfc` is injected by the caller (each Part has its own textnorm.nfc) so
    this module stays free of Part-specific imports.
    """
    df = pd.read_csv(sentences_csv, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    if "id" not in cols or "sentence" not in cols:
        raise ValueError(
            f"{sentences_csv} must have columns 'id' and 'sentence'; got {list(df.columns)}"
        )
    df = df.rename(columns={cols["id"]: "id", cols["sentence"]: "sentence"})
    df["id"] = df["id"].astype(int)
    df["sentence"] = df["sentence"].astype(str).map(nfc)
    return dict(zip(df["id"], df["sentence"]))


def parse_speaker(folder_name: str):
    m = SPEAKER_RE.match(folder_name)
    if not m:
        return None
    return m.group("gender").lower(), m.group("id"), m.group("name")


def probe_raw_format(path):
    """Best-effort (sample_rate, channels) from the file header only (no full decode)."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.samplerate, info.channels
    except Exception:
        pass
    try:
        import wave

        with wave.open(str(path), "rb") as w:
            return w.getframerate(), w.getnchannels()
    except Exception:
        return None, None


def build_manifest_core(
    data_root,
    bpac_subdir,
    sentences_csv,
    nfc,
    probe_durations=False,
    probe_format=False,
    duration_fn=None,
    tqdm_fn=None,
    verbose=True,
):
    """Walk data_root/bpac_subdir and return a per-utterance DataFrame.

    `duration_fn(path) -> float` and `tqdm_fn(iterable, desc=...)` are
    injected by the caller so this module has no dependency on any Part's
    audio-decoding or progress-bar stack.
    """
    data_root = Path(data_root)
    bpac = data_root / bpac_subdir
    if not bpac.is_dir():
        raise FileNotFoundError(f"BPAC directory not found: {bpac}")

    sentences = load_sentences(sentences_csv, nfc)

    rows = []
    skipped_speaker_dirs, skipped_files = [], 0
    for region_dir in sorted(p for p in bpac.iterdir() if p.is_dir()):
        region_raw = region_dir.name
        region = region_raw.lower()
        for spk_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            parsed = parse_speaker(spk_dir.name)
            if parsed is None:
                skipped_speaker_dirs.append(str(spk_dir))
                continue
            gender, spk_id, name = parsed
            speaker_uid = f"{region}_{gender}_{spk_id}_{name}".lower()
            for wav in sorted(spk_dir.glob("*.wav")):
                m = SOUND_RE.match(wav.name)
                if not m:
                    skipped_files += 1
                    continue
                sid = int(m.group("num"))
                rows.append(
                    {
                        "utterance_id": f"{speaker_uid}_{sid:03d}",
                        "filepath": str(wav),
                        "region": region,
                        "region_raw": region_raw,
                        "gender": gender,
                        "pipeline": GENDER_TO_PIPELINE[gender],
                        "speaker_id": spk_id,
                        "speaker_name": name,
                        "speaker_uid": speaker_uid,
                        "sentence_id": sid,
                        "sentence": sentences.get(sid, None),
                    }
                )

    df = pd.DataFrame(rows)
    if verbose:
        print(f"[manifest] utterances: {len(df)}")
        if len(df):
            print(f"[manifest] speakers : {df['speaker_uid'].nunique()}")
            print(f"[manifest] regions  : {df['region'].nunique()}")
        if skipped_speaker_dirs:
            print(f"[manifest] WARNING skipped {len(skipped_speaker_dirs)} non-conforming speaker dirs")
            for s in skipped_speaker_dirs[:10]:
                print(f"           - {s}")
        if skipped_files:
            print(f"[manifest] WARNING skipped {skipped_files} non-conforming .wav files")
        missing_text = int(df["sentence"].isna().sum()) if len(df) else 0
        if missing_text:
            print(f"[manifest] WARNING {missing_text} utterances have no matching sentence id in the CSV")

    if len(df):
        assert df.groupby("pipeline")["gender"].nunique().eq(1).all(), (
            "pipeline/gender confound assumption violated"
        )

    if probe_durations and len(df) and duration_fn is not None:
        iterable = df["filepath"]
        if tqdm_fn is not None:
            iterable = tqdm_fn(iterable, desc="probing durations")
        df["duration_s"] = [duration_fn(p) for p in iterable]

    if probe_format and len(df):
        iterable = df["filepath"]
        if tqdm_fn is not None:
            iterable = tqdm_fn(iterable, desc="probing raw format")
        srs, chans = [], []
        for p in iterable:
            sr, ch = probe_raw_format(p)
            srs.append(sr)
            chans.append(ch)
        df["recording_sr_raw"] = srs
        df["channels_raw"] = chans

    return df
