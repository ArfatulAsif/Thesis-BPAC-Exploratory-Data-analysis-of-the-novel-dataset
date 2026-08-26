"""Embedding extraction for Part C, cached and resumable.

Two embedding kinds, shared by C1 and C2 via a per-utterance cache:
  * "ssl"     -> WavLM-Large (or any HF SSL model): mean/max-pooled hidden state,
                 captures phonetic / accent detail. (C1 distance + C2 content axis)
  * "speaker" -> ECAPA-TDNN via SpeechBrain: speaker-identity embedding.
                 (C2 confound axis)

Empty / too-short / silent files are skipped (not embedded) and reported, so
dead recordings don't pollute the analysis.

Cache: cache/emb_<kind>.npz  (keys = utterance_id, values = vectors). Checkpointed
every `inference.save_every` utterances, so extraction resumes after a crash.

Heavy deps (torch / transformers / speechbrain / librosa) are imported lazily.
"""
import os
from pathlib import Path

import numpy as np

# quiet the Windows HF symlink warning (we copy model files; see SpeakerExtractor)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from .manifest import build_manifest
from .reporting import ensure_dir, maybe_tqdm


# --------------------------------------------------------------------- helpers
def _device(pref="auto"):
    try:
        import torch

        if pref == "cpu":
            return "cpu"
        if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
            return "cuda"
    except Exception:
        pass
    return "cpu"


_LOAD_DIAG = {"printed": 0, "load_fail": 0, "trim_fail": 0, "flagged_empty": 0, "ok": 0}


def _load_trimmed(path, sr, top_db, diag=True):
    """Load mono @sr, trim silence. Return (y, is_empty).

    Reads with soundfile first (fast, decodes WAV/FLAC/OGG directly, no audioread),
    resampling if needed; falls back to librosa only if soundfile can't open it.
    Surfaces the real reason for the first few failures.
    """
    y = None
    err_sf = err_lr = None

    # 1) soundfile first — handles standard WAV at any rate / channel count
    try:
        import soundfile as sf

        data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:                      # stereo -> mono
            data = data.mean(axis=1)
        if data.size and file_sr != sr:        # resample to target
            import librosa
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        y = np.asarray(data, dtype=np.float32)
    except Exception as e:
        err_sf = e

    # 2) fall back to librosa (which may use audioread)
    if y is None:
        try:
            import librosa
            y, _ = librosa.load(str(path), sr=sr, mono=True)
            y = np.asarray(y, dtype=np.float32)
        except Exception as e:
            err_lr = e

    if y is None:
        _LOAD_DIAG["load_fail"] += 1
        if diag and _LOAD_DIAG["printed"] < 8:
            _LOAD_DIAG["printed"] += 1
            print(f"[emb] LOAD FAILED: {path} -> soundfile: {type(err_sf).__name__}: "
                  f"{err_sf} | librosa: {type(err_lr).__name__}: {err_lr}")
        return np.zeros(0, dtype=np.float32), True

    try:
        if y.size:
            import librosa
            yt, _ = librosa.effects.trim(y, top_db=top_db)
            if yt.size:
                y = yt
    except Exception as e:
        _LOAD_DIAG["trim_fail"] += 1
        if diag and _LOAD_DIAG["printed"] < 8:
            _LOAD_DIAG["printed"] += 1
            print(f"[emb] TRIM FAILED: {path} -> {type(e).__name__}: {e}")

    min_samples = max(400, int(0.20 * sr))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    is_empty = (y.size == 0) or (y.size < min_samples) or (peak <= 1e-5)
    if is_empty:
        if y.size > 0 and diag and _LOAD_DIAG["printed"] < 8:
            _LOAD_DIAG["printed"] += 1
            print(f"[emb] FLAGGED EMPTY (samples={y.size}, peak={peak:.2e}): {path}")
        _LOAD_DIAG["flagged_empty"] += 1
    else:
        _LOAD_DIAG["ok"] += 1
    return y, is_empty


def _load_npz_dict(path):
    if not Path(path).exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        return {k: data[k] for k in data.files}
    except Exception:
        return {}


def _save_npz_dict(path, d):
    np.savez(path, **d)


def _probe_first_files(rows, cfg, n=3):
    """Loudly report what happens when loading the first few files (once)."""
    print(f"[emb] --- audio probe (first {n} files) ---")
    for r in list(rows)[:n]:
        path = r.filepath
        info_str = ""
        try:
            import soundfile as sf
            info = sf.info(str(path))
            info_str = (f"sr={info.samplerate} ch={info.channels} "
                        f"frames={info.frames} fmt={info.format}/{info.subtype}")
        except Exception as e:
            info_str = f"soundfile.info failed: {type(e).__name__}: {e}"
        y, is_empty = _load_trimmed(path, cfg.target_sr, cfg.top_db, diag=True)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        print(f"[emb]   {os.path.basename(str(path))}: {info_str} | "
              f"loaded_samples={y.size} peak={peak:.2e} is_empty={is_empty}")
    print(f"[emb] --- end probe ---")


# ----------------------------------------------------------------- extractors
class SSLExtractor:
    """Pooled hidden-state embedding from a HF SSL model (e.g. WavLM-Large)."""

    def __init__(self, cfg):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.device = _device(cfg.device)
        self.sr = cfg.target_sr
        self.top_db = cfg.top_db
        self.layer = cfg.ssl_layer
        self.pooling = cfg.ssl_pooling
        self.fe = AutoFeatureExtractor.from_pretrained(cfg.ssl_id)
        self.model = AutoModel.from_pretrained(cfg.ssl_id, output_hidden_states=True)
        self.model.to(self.device).eval()
        self.dtype = torch.float32
        if cfg.fp16 and self.device == "cuda":
            self.model.half()
            self.dtype = torch.float16
        self._torch = torch

    def __call__(self, path):
        y, is_empty = _load_trimmed(path, self.sr, self.top_db)
        if is_empty:
            return None
        torch = self._torch
        inputs = self.fe(y, sampling_rate=self.sr, return_tensors="pt")
        iv = inputs.input_values.to(self.device)
        if self.dtype == torch.float16:
            iv = iv.half()
        with torch.no_grad():
            out = self.model(iv)
        hs = out.hidden_states  # tuple(L+1) of (1, T, D)
        if isinstance(self.layer, str):
            lname = self.layer.lower()
            if lname == "mean":
                h = torch.stack(hs, dim=0).mean(dim=0)        # (1, T, D)
            elif lname == "middle":
                h = hs[len(hs) // 2]                          # most phonetic ~ middle
            else:
                h = hs[int(self.layer)]
        else:
            h = hs[int(self.layer)]
        if self.pooling == "max":
            vec = h.float().squeeze(0).max(dim=0).values
        else:
            vec = h.float().squeeze(0).mean(dim=0)
        return vec.cpu().numpy().astype(np.float32)

    def close(self):
        try:
            del self.model
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class SpeakerExtractor:
    """ECAPA-TDNN speaker embedding via SpeechBrain."""

    def __init__(self, cfg):
        self.sr = cfg.target_sr
        self.top_db = cfg.top_db
        self.device = _device(cfg.device)
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:
            try:
                from speechbrain.pretrained import EncoderClassifier
            except Exception as e:
                raise ImportError(
                    "speechbrain is required for speaker embeddings: "
                    "pip install speechbrain"
                ) from e

        kw = dict(
            source=cfg.speaker_source,
            run_opts={"device": self.device},
            savedir=str(Path(cfg.cache_dir) / "speechbrain_ecapa"),
        )
        # On Windows, SpeechBrain's default SYMLINK fetch strategy needs admin /
        # Developer Mode and fails with WinError 1314. Force COPY instead.
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kw["local_strategy"] = LocalStrategy.COPY
        except Exception:
            pass
        try:
            self.model = EncoderClassifier.from_hparams(**kw)
        except TypeError:
            # older speechbrain without the local_strategy argument
            kw.pop("local_strategy", None)
            self.model = EncoderClassifier.from_hparams(**kw)

    def __call__(self, path):
        import torch

        y, is_empty = _load_trimmed(path, self.sr, self.top_db)
        if is_empty:
            return None
        sig = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            emb = self.model.encode_batch(sig)  # (1, 1, D)
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)

    def close(self):
        try:
            del self.model
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _build_extractor(cfg, kind):
    if kind == "ssl":
        return SSLExtractor(cfg)
    if kind == "speaker":
        return SpeakerExtractor(cfg)
    raise ValueError(f"unknown embedding kind: {kind}")


def _cache_tag(cfg, kind):
    """Cache filename tag — includes model + (for ssl) layer + pooling, so
    different layer/pooling choices don't collide in the cache."""
    if kind == "ssl":
        m = str(cfg.ssl_id).split("/")[-1]
        return f"ssl__{m}__L{cfg.ssl_layer}__{cfg.ssl_pooling}"
    return f"speaker__{str(cfg.speaker_source).split('/')[-1]}"


# ----------------------------------------------------------------- public API
def extract_embeddings(cfg, kind):
    """Return (meta_df, X) for embeddings of `kind` over all (non-empty) utterances.

    meta_df rows align with X rows; columns include utterance_id, region, gender,
    speaker_uid, sentence_id, filepath. Resumable via a cached npz.
    """
    import pandas as pd

    manifest = build_manifest(cfg.data_root, cfg.bpac_subdir, cfg.sentences_csv, verbose=cfg.verbose,
                               verified_manifest_path=getattr(cfg, "verified_manifest", None))
    manifest = manifest.dropna(subset=["sentence"]).reset_index(drop=True)
    if cfg.max_utterances:
        manifest = (manifest.sort_values(["speaker_uid", "sentence_id"])
                    .head(cfg.max_utterances).reset_index(drop=True))

    cache_dir = ensure_dir(cfg.cache_dir)
    tag = _cache_tag(cfg, kind)
    cache_path = Path(cache_dir) / f"emb_{tag}.npz"
    excluded_path = Path(cache_dir) / f"excluded_{tag}.txt"
    store = _load_npz_dict(cache_path)
    excluded = set()
    if excluded_path.exists():
        excluded = set(excluded_path.read_text(encoding="utf-8").split())
    # If a previous run stored nothing but excluded everything, it almost
    # certainly failed wholesale (e.g. an audio-backend problem) rather than the
    # corpus being empty. Don't let that stale list skip the retry.
    if len(store) == 0 and len(excluded) > 0:
        if cfg.verbose:
            print(f"[emb:{kind}] ignoring stale excluded list ({len(excluded)} ids) "
                  f"because no embeddings were stored — re-attempting all.")
        excluded = set()

    todo = [r for r in manifest.itertuples(index=False)
            if (r.utterance_id not in store) and (r.utterance_id not in excluded)]

    if todo:
        if cfg.verbose:
            print(f"[emb:{kind}] loader: soundfile-first v2")
            print(f"[emb:{kind}] extracting {len(todo)} of {len(manifest)} "
                  f"(model: {cfg.ssl_id if kind=='ssl' else cfg.speaker_source})")
            _probe_first_files(todo, cfg, n=3)
        extractor = _build_extractor(cfg, kind)
        save_every = max(1, int(cfg.save_every))
        consecutive_empty = 0
        try:
            for i, r in enumerate(maybe_tqdm(todo, desc=f"emb[{kind}]")):
                vec = extractor(r.filepath)
                if vec is None:
                    excluded.add(r.utterance_id)
                    consecutive_empty += 1
                else:
                    store[r.utterance_id] = vec
                    consecutive_empty = 0
                # fail fast: if the first 50 are all empty, something is wrong with
                # this extractor/audio — don't grind through the whole corpus.
                if consecutive_empty >= 50 and len(store) == 0:
                    raise SystemExit(
                        f"[emb:{kind}] ABORTING: first {consecutive_empty} utterances "
                        f"all returned empty while nothing was stored. See the probe / "
                        f"'LOAD FAILED' / 'FLAGGED EMPTY' lines above for the real cause. "
                        f"(C1 working but this failing points to a problem specific to the "
                        f"'{kind}' extractor, not the audio.)"
                    )
                if (i + 1) % save_every == 0:
                    _save_npz_dict(cache_path, store)
                    excluded_path.write_text(" ".join(sorted(excluded)), encoding="utf-8")
                    if cfg.verbose:
                        print(f"[emb:{kind}] {i+1}/{len(todo)} -> checkpointed "
                              f"({len(store)} stored, {len(excluded)} empty).")
        finally:
            _save_npz_dict(cache_path, store)
            excluded_path.write_text(" ".join(sorted(excluded)), encoding="utf-8")
            extractor.close()
        if cfg.verbose:
            print(f"[emb:{kind}] done. stored={len(store)} excluded_empty={len(excluded)}")
        d = _LOAD_DIAG
        if d["load_fail"] or d["trim_fail"]:
            print(f"[emb:{kind}] audio diagnostics: ok={d['ok']} "
                  f"load_failures={d['load_fail']} trim_failures={d['trim_fail']} "
                  f"flagged_empty={d['flagged_empty']}")

    # Build aligned arrays for utterances that have an embedding.
    keep = manifest[manifest["utterance_id"].isin(store.keys())].reset_index(drop=True)
    if len(keep) == 0:
        one = manifest["filepath"].iloc[0] if len(manifest) else "<no files>"
        raise SystemExit(
            f"[emb:{kind}] No embeddings were produced — all {len(manifest)} utterances "
            f"were skipped. This almost always means the audio could not be LOADED "
            f"(not that it is truly empty); see any 'LOAD FAILED' lines above for the "
            f"real error.\n"
            f"  Quick check from your project root:\n"
            f"    python -c \"import librosa; y,sr=librosa.load(r'{one}', sr=16000); "
            f"print('samples',len(y),'peak',float(abs(y).max()))\"\n"
            f"  If that errors, the audio backend is the problem. Fix with:\n"
            f"    pip install --upgrade soundfile audioread\n"
            f"  (soundfile reads WAV directly; audioread is the fallback decoder.)"
        )
    X = np.stack([store[u] for u in keep["utterance_id"]]).astype(np.float32)
    if cfg.verbose:
        print(f"[emb:{kind}] matrix: {X.shape}; excluded_empty={len(excluded)}")
    return keep, X
