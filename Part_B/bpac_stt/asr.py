"""ASR engines and per-model cached transcription.

Two engine types via the HuggingFace `pipeline` API:
  * "whisper"      -> encoder-decoder, greedy, chunked (handles >30 s clips)
  * "wav2vec2_ctc" -> CTC (uses the repo's bundled n-gram LM if pyctcdecode is
                      installed; otherwise greedy)

Low-VRAM friendly (RTX 3050 4 GB): batch_size from config (default 1), fp16 on
CUDA, greedy decoding, chunk_length_s windows, cache cleared per batch, each
model freed before the next, and incremental checkpointing (resumable).

`ensure_predictions(cfg, models=[...])` transcribes ONLY the given model(s) into
the shared cache, so each model can be run independently. It also records an
`is_empty_audio` flag per utterance (empty / too-short / silent files), which
scoring can exclude so dead recordings don't corrupt WER/CER.

Heavy deps (torch/transformers/librosa) are imported lazily.
"""
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import pandas as pd

from .manifest import build_manifest
from .reporting import ensure_dir, maybe_tqdm


def resolve_device(pref="auto"):
    try:
        import torch

        if pref == "cpu":
            return -1
        if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
            return 0
    except Exception:
        pass
    return -1


def _empty_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _make_pipeline(dtype=None, **kw):
    from transformers import pipeline

    if dtype is not None:
        try:
            return pipeline("automatic-speech-recognition", dtype=dtype, **kw)
        except TypeError:
            return pipeline("automatic-speech-recognition", torch_dtype=dtype, **kw)
    return pipeline("automatic-speech-recognition", **kw)


class ASREngine:
    def __init__(self, model_cfg, device_index, target_sr, batch_size,
                 whisper_beam_size=1, fp16=True, max_new_tokens=None, chunk_length_s=30):
        self.cfg = model_cfg
        self.key = model_cfg["key"]
        self.type = model_cfg["type"]
        self.target_sr = target_sr
        self._min_samples = max(400, int(0.20 * target_sr))
        self.batch_size = max(1, int(batch_size))

        torch_dtype = None
        if fp16 and device_index is not None and device_index >= 0:
            try:
                import torch
                torch_dtype = torch.float16
            except Exception:
                torch_dtype = None

        kw = {"model": model_cfg["hf_id"], "device": device_index}
        if self.type == "whisper" and chunk_length_s:
            kw["chunk_length_s"] = int(chunk_length_s)
        self.pipe = _make_pipeline(dtype=torch_dtype, **kw)

        if self.type == "whisper":
            self._patch_timestamp_config()

        self._variants = self._build_variants(model_cfg, whisper_beam_size, max_new_tokens)
        self._call_kwargs = None

    def _patch_timestamp_config(self):
        try:
            genc = self.pipe.model.generation_config
            if getattr(genc, "no_timestamps_token_id", None) is None:
                tid = self.pipe.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
                if isinstance(tid, int) and tid >= 0:
                    genc.no_timestamps_token_id = tid
        except Exception:
            pass

    @staticmethod
    def _build_variants(model_cfg, beam, max_new_tokens):
        if model_cfg["type"] != "whisper":
            return [{}]
        base = {"num_beams": int(beam) if (beam and beam >= 1) else 1, "do_sample": False}
        if max_new_tokens:
            base["max_new_tokens"] = int(max_new_tokens)
        full = dict(base)
        full["task"] = "transcribe"
        if model_cfg.get("language"):
            full = {"language": model_cfg["language"], **full}
        variants = [{"generate_kwargs": dict(full)}]
        no_lang = {k: v for k, v in full.items() if k != "language"}
        if no_lang != full:
            variants.append({"generate_kwargs": dict(no_lang)})
        variants.append({"generate_kwargs": dict(base)})
        return variants

    def _load(self, path):
        """Return (audio_array, is_empty). Pads short clips; flags empty/silent."""
        import librosa
        import numpy as np

        try:
            y, _ = librosa.load(str(path), sr=self.target_sr, mono=True)
            y = np.asarray(y, dtype=np.float32)
        except Exception as e:
            print(f"[asr] WARNING could not load {path}: {e}; treating as blank.")
            y = np.zeros(0, dtype=np.float32)
        n = int(y.size)
        peak = float(np.max(np.abs(y))) if n else 0.0
        is_empty = (n == 0) or (n < self._min_samples) or (peak <= 1e-5)
        if n == 0:
            print(f"[asr] WARNING empty/zero-length audio: {path}; transcribing as blank.")
        if n < self._min_samples:
            y = np.pad(y, (0, self._min_samples - n))
        return y, is_empty

    def _run(self, inputs):
        if self._call_kwargs is not None:
            return self.pipe(inputs, batch_size=len(inputs), **self._call_kwargs)
        last_err = None
        for kw in self._variants:
            try:
                outs = self.pipe(inputs, batch_size=len(inputs), **kw)
                self._call_kwargs = kw
                gk = kw.get("generate_kwargs", {})
                if self.type == "whisper" and "language" not in gk:
                    print(f"[asr] {self.key}: generation config rejects forced language; "
                          f"decoding with defaults (generate_kwargs={gk}).")
                return outs
            except ValueError as e:
                msg = str(e).lower()
                if "language" in msg or "task" in msg or "generation config" in msg:
                    last_err = e
                    continue
                raise
        raise last_err if last_err else RuntimeError("ASR failed to resolve generate kwargs")

    def transcribe_paths(self, paths):
        """Return (texts, empties) aligned with `paths`."""
        texts, empties = [], []
        paths = list(paths)
        for i in maybe_tqdm(range(0, len(paths), self.batch_size),
                            desc=f"transcribe[{self.key}]"):
            batch = paths[i:i + self.batch_size]
            loaded = [self._load(p) for p in batch]
            inputs = [y for y, _ in loaded]
            empties.extend(bool(e) for _, e in loaded)
            outs = self._run(inputs)
            if isinstance(outs, dict):
                outs = [outs]
            texts.extend((o.get("text", "") if isinstance(o, dict) else str(o)) for o in outs)
            _empty_cuda_cache()
        return texts, empties

    def close(self):
        try:
            del self.pipe
        except Exception:
            pass
        gc.collect()
        _empty_cuda_cache()


BASE_COLS = ["utterance_id", "region", "gender", "speaker_uid", "sentence_id", "reference"]


def _base_frame(manifest):
    df = manifest[["utterance_id", "region", "gender", "speaker_uid", "sentence_id"]].copy()
    df["reference"] = manifest["sentence"].values
    return df


def ensure_predictions(cfg, models=None, force=None):
    """Transcribe the given model(s) into the shared cache and return predictions.

    `models` defaults to all of cfg.models; pass a single-element list to run one
    model independently. Resumable; checkpoints every `inference.save_every`.
    """
    models = list(models) if models is not None else list(cfg.models)
    force = cfg.force_retranscribe if force is None else force
    cache = Path(cfg.predictions_cache)
    ensure_dir(cache.parent)

    manifest = build_manifest(cfg.data_root, cfg.bpac_subdir, cfg.sentences_csv, verbose=cfg.verbose,
                               verified_manifest_path=cfg.verified_manifest)
    manifest = manifest.dropna(subset=["sentence"]).reset_index(drop=True)
    if cfg.max_utterances:
        manifest = (manifest.sort_values(["speaker_uid", "sentence_id"])
                    .head(cfg.max_utterances).reset_index(drop=True))

    if cache.exists() and not force:
        pred = pd.read_csv(cache)
        if set(pred.get("utterance_id", [])) != set(manifest["utterance_id"]):
            if cfg.verbose:
                print("[asr] cache utterances differ from manifest — rebuilding base frame.")
            pred = _base_frame(manifest)
        else:
            # Same utterance_id set doesn't mean same labels: verified_manifest.csv
            # corrections (T4 shifts) can change sentence_id/sentence for an
            # utterance without changing which utterances exist. Refresh the
            # manifest-derived columns from the current manifest every time;
            # keep only the expensive-to-recompute ASR hypothesis columns from
            # the cache (audio itself never changes, so re-transcribing would
            # be wasted work).
            base = _base_frame(manifest).set_index("utterance_id")
            keep_cols = [c for c in pred.columns if c not in BASE_COLS and c != "utterance_id"]
            pred = base.join(pred.set_index("utterance_id")[keep_cols], how="left").reset_index()
            if cfg.verbose:
                print("[asr] cache utterances match manifest ID set — refreshed "
                      "region/gender/speaker_uid/sentence_id/reference from the "
                      "current manifest, reused cached ASR hypotheses.")
    else:
        pred = _base_frame(manifest)

    if "is_empty_audio" not in pred.columns:
        pred["is_empty_audio"] = pd.NA

    path_map = dict(zip(manifest["utterance_id"], manifest["filepath"]))
    pred = pred.set_index("utterance_id")
    save_every = max(1, int(getattr(cfg, "save_every", 200)))

    def _checkpoint():
        pred.reset_index().to_csv(cache, index=False, encoding="utf-8-sig")

    # Persist any manifest-refresh (region/gender/speaker_uid/sentence_id/
    # reference) immediately, even if every model below turns out fully
    # cached and the per-model loop never calls _checkpoint() itself --
    # otherwise a corrected verified_manifest.csv silently never reaches
    # disk and downstream readers (e.g. T3) keep seeing stale labels.
    if cache.exists() and not force:
        _checkpoint()

    device_index = resolve_device(cfg.device)
    if cfg.verbose:
        print(f"[asr] device index = {device_index} (>=0 cuda, -1 cpu); "
              f"batch_size={cfg.batch_size}, fp16={cfg.fp16}, "
              f"chunk_length_s={cfg.chunk_length_s}, save_every={save_every}")

    for m in models:
        col = f"hyp_{m['key']}"
        if col not in pred.columns:
            pred[col] = pd.NA

        todo_ids = [u for u in pred.index if force or pd.isna(pred.at[u, col])]
        if not todo_ids:
            if cfg.verbose:
                print(f"[asr] {m['key']}: cached, skipping.")
            continue

        if cfg.verbose:
            print(f"[asr] {m['key']}: transcribing {len(todo_ids)} utterances "
                  f"(of {len(pred)}) with {m['hf_id']} ...")
        engine = ASREngine(m, device_index, cfg.target_sr, cfg.batch_size,
                           cfg.whisper_beam_size, cfg.fp16, cfg.whisper_max_new_tokens,
                           cfg.chunk_length_s)
        try:
            for start in range(0, len(todo_ids), save_every):
                ids_chunk = todo_ids[start:start + save_every]
                paths_chunk = [path_map[u] for u in ids_chunk]
                hyps, empties = engine.transcribe_paths(paths_chunk)
                for uid, h, e in zip(ids_chunk, hyps, empties):
                    # store blank output as a space so it survives the CSV round-trip
                    # (empty cell -> NaN would be indistinguishable from "not transcribed")
                    pred.at[uid, col] = h if (isinstance(h, str) and h.strip()) else " "
                    pred.at[uid, "is_empty_audio"] = bool(e)
                _checkpoint()
                if cfg.verbose:
                    done = min(start + save_every, len(todo_ids))
                    print(f"[asr] {m['key']}: {done}/{len(todo_ids)} done -> checkpointed.")
        finally:
            engine.close()
            del engine
            gc.collect()
            _empty_cuda_cache()
        if cfg.verbose:
            print(f"[asr] {m['key']}: complete.")

    pred = pred.reset_index()
    return pred
