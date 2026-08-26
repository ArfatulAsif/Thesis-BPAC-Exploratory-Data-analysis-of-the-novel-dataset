"""Parse MFA TextGrids into tidy phone/word-level tables (T12 deliverable).

MFA's `mfa train --output_directory ...` writes one TextGrid per utterance
(mirroring the corpus's speaker/utt.TextGrid layout), each with a "words"
tier and a "phones" tier of labeled intervals. This flattens every
TextGrid into one row per phone token (and one per word token), joined
back to the manifest for region/speaker/gender/pipeline/sentence_id.
"""
from pathlib import Path

import pandas as pd

SIL_LABELS = {"", "sil", "sp", "spn", "<eps>"}


def _speaker_and_utt_from_path(tg_path: Path):
    # mfa_corpus/<speaker_uid>/<utterance_id>.TextGrid (mirrors input layout)
    return tg_path.parent.name, tg_path.stem


def parse_one_textgrid(tg_path, meta_row=None):
    """Return (word_rows, phone_rows) for one TextGrid.

    meta_row: optional dict/Series with region, gender, pipeline, sentence_id,
    speaker_uid, sentence_text -- merged onto every row if given.
    """
    from praatio import textgrid as tgio

    tg = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    speaker_uid, utterance_id = _speaker_and_utt_from_path(Path(tg_path))

    tier_names = {name.lower(): name for name in tg.tierNames}
    words_tier = tg.getTier(tier_names["words"]) if "words" in tier_names else None
    phones_tier = tg.getTier(tier_names["phones"]) if "phones" in tier_names else None

    base = {"utterance_id": utterance_id, "speaker_uid": speaker_uid}
    if meta_row is not None:
        for k in ("region", "gender", "pipeline", "sentence_id", "sentence"):
            if k in meta_row:
                base[k] = meta_row[k]

    word_rows = []
    if words_tier is not None:
        for wi, iv in enumerate(words_tier.entries):
            label = (iv.label or "").strip()
            if label.lower() in SIL_LABELS:
                continue
            word_rows.append({
                **base, "word_idx": wi, "word": label,
                "start_s": float(iv.start), "end_s": float(iv.end),
                "duration_s": float(iv.end) - float(iv.start),
            })

    phone_rows = []
    if phones_tier is not None:
        entries = phones_tier.entries
        # map each phone interval to its enclosing word (by time overlap)
        word_intervals = [(w["start_s"], w["end_s"], w["word"], w["word_idx"]) for w in word_rows]

        def find_word(t0, t1):
            mid = (t0 + t1) / 2.0
            for ws, we, wtxt, widx in word_intervals:
                if ws <= mid < we:
                    return wtxt, widx
            return None, None

        labels = [(iv.label or "").strip() for iv in entries]
        for pi, iv in enumerate(entries):
            label = labels[pi]
            if label.lower() in SIL_LABELS:
                continue
            wtxt, widx = find_word(float(iv.start), float(iv.end))
            prev_label = next((labels[j] for j in range(pi - 1, -1, -1)
                              if labels[j].lower() not in SIL_LABELS), None)
            next_label = next((labels[j] for j in range(pi + 1, len(labels))
                              if labels[j].lower() not in SIL_LABELS), None)
            phone_rows.append({
                **base, "word": wtxt, "word_idx": widx, "phone_idx": pi, "phone": label,
                "start_s": float(iv.start), "end_s": float(iv.end),
                "duration_s": float(iv.end) - float(iv.start),
                "prev_phone": prev_label, "next_phone": next_label,
            })

    return word_rows, phone_rows


def parse_textgrid_dir(textgrid_dir, manifest_df=None, progress=None):
    """Walk every .TextGrid under textgrid_dir and return (word_df, phone_df).

    manifest_df: DataFrame with utterance_id, region, gender, pipeline,
    sentence_id, sentence -- if given, merged onto every row.
    """
    textgrid_dir = Path(textgrid_dir)
    files = sorted(textgrid_dir.rglob("*.TextGrid"))
    if progress is not None:
        files = progress(files, desc="parsing TextGrids", total=len(files))

    meta_lookup = {}
    if manifest_df is not None:
        meta_lookup = manifest_df.set_index("utterance_id").to_dict(orient="index")

    all_words, all_phones = [], []
    n_failed = 0
    for f in files:
        utt_id = f.stem
        meta_row = meta_lookup.get(utt_id)
        try:
            w, p = parse_one_textgrid(f, meta_row=meta_row)
            all_words.extend(w)
            all_phones.extend(p)
        except Exception as e:
            n_failed += 1
            if n_failed <= 10:
                print(f"[textgrid_io] failed to parse {f}: {e}")

    word_df = pd.DataFrame(all_words)
    phone_df = pd.DataFrame(all_phones)

    # position_in_word (T12 schema): index of this phone within its word's phone sequence
    if len(phone_df):
        phone_df = phone_df.sort_values(["utterance_id", "phone_idx"]).reset_index(drop=True)
        phone_df["position_in_word"] = phone_df.groupby(["utterance_id", "word_idx"]).cumcount()

    if n_failed:
        print(f"[textgrid_io] {n_failed}/{len(files)} TextGrids failed to parse")
    return word_df, phone_df
