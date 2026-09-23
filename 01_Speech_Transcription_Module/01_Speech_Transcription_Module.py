"""Russian speech transcription with faster-whisper; run --help for options."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

BASE = Path(__file__).resolve().parent
FORMATS = {".mp3", ".wav", ".m4a", ".flac"}


class ModelError(RuntimeError):
    """A model could not be initialized; stop the whole batch."""


def safe_error(exc: Exception) -> str:
    """Never copy external exception text (which may contain credentials) to logs."""
    message = str(exc).lower()
    if isinstance(exc, PermissionError):
        return "Access denied: check input/output permissions and open files."
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "Dependency unavailable: install requirements.txt in a compatible Python environment."
    if "out of memory" in message or isinstance(exc, MemoryError):
        return "Insufficient memory: release GPU/RAM resources or select --device cpu."
    if any(s in message for s in ("cuda", "cudnn", "cublas", "float16")):
        return "CUDA initialization/inference failed: check NVIDIA driver, CUDA 12, cuDNN 9 and GPU memory."
    if isinstance(exc, (ValueError, EOFError)) or "invaliddata" in type(exc).__name__.lower():
        return "Invalid audio or timestamps: check that the audio is readable and not corrupted."
    if isinstance(exc, OSError):
        return "File or network operation failed: check files, disk space, connection and model cache."
    return "Processing failed: check audio, installed dependencies, model cache and device configuration."


def discover(source: Path) -> list[Path]:
    if not source.exists():
        raise ValueError(f"Input path does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in FORMATS:
            raise ValueError("Supported formats: .mp3, .wav, .m4a, .flac")
        return [source]
    if not source.is_dir():
        raise ValueError("Input must be a file or directory.")
    files = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in FORMATS),
                   key=lambda p: (p.name.casefold(), p.name))
    if not files:
        raise ValueError("No supported audio files found in the input directory.")
    return files


def output_names(files: list[Path]) -> list[str]:
    """Resolve extension, case and secondary name conflicts over the entire batch."""
    counts = Counter(p.stem.casefold() for p in files)
    candidates = [p.stem if counts[p.stem.casefold()] == 1
                  else f"{p.stem}_{p.suffix[1:].lower()}" for p in files]
    repeated = Counter(s.casefold() for s in candidates)
    reserved = {s.casefold() for s in candidates}
    used: set[str] = set()
    result = []
    for path, name in zip(files, candidates):
        if repeated[name.casefold()] > 1 or name.casefold() in used:
            digest = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:10]
            root = f"{name}_{digest}"
            name = root
            index = 2
            while name.casefold() in reserved or name.casefold() in used:
                name = f"{root}_{index}"
                index += 1
        used.add(name.casefold())
        result.append(name)
    return result


class Engine:
    def __init__(self, model_name: str, device: str):
        self.name = model_name
        self.automatic = device == "auto"
        self.model = None
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
            self.factory = WhisperModel
            if self.automatic:
                try:
                    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                except Exception as exc:
                    print(f"CUDA probe failed; using CPU. {safe_error(exc)}", file=sys.stderr)
                    device = "cpu"
            self.device = device
            try:
                self._load()
            except Exception as exc:
                if self.automatic and self.device == "cuda":
                    self._fallback(exc)
                else:
                    raise
        except Exception as exc:
            raise ModelError(safe_error(exc)) from None

    def _load(self):
        compute = "float16" if self.device == "cuda" else "int8"
        print(f"Model: {self.name}; device: {self.device}; compute_type: {compute}")
        self.model = self.factory(self.name, device=self.device, compute_type=compute)

    def _fallback(self, exc: Exception):
        print(f"GPU failed; switching to CPU. {safe_error(exc)}", file=sys.stderr)
        self.model = None
        gc.collect()
        self.device = "cpu"
        try:
            self._load()
        except Exception as error:
            raise ModelError(safe_error(error)) from None

    def transcribe(self, audio: Path, words: bool) -> dict:
        try:
            return self._transcribe(audio, words)
        except Exception as exc:
            # CUDA errors can occur only when the lazy segment iterator is consumed.
            gpu_error = any(s in str(exc).lower() for s in
                            ("cuda", "cudnn", "cublas", "out of memory", "float16"))
            if not (self.automatic and self.device == "cuda" and gpu_error):
                raise
            self._fallback(exc)
            return self._transcribe(audio, words)

    def _transcribe(self, audio: Path, words: bool) -> dict:
        segments, info = self.model.transcribe(
            str(audio), language="ru", task="transcribe", vad_filter=True,
            beam_size=5, word_timestamps=words,
        )
        duration = float(info.duration)  # Original decoded duration, before VAD.
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid duration")
        items = []
        for segment in segments:
            # faster-whisper restores VAD timestamps to the original audio timeline.
            check_times(segment.start, segment.end, duration)
            item = {"id": segment.id, "start": segment.start,
                    "end": segment.end, "text": segment.text}
            if words:
                item["words"] = []
                for word in segment.words or []:
                    check_times(word.start, word.end, duration)
                    item["words"].append({"word": word.word, "start": word.start, "end": word.end})
            items.append(item)
        text = "".join(s["text"] for s in items).strip()
        return {"audio_file": audio.name, "model": self.name, "language": "ru",
                "duration_seconds": duration, "text": text, "segments": items}


def check_times(start: float, end: float, duration: float):
    # Whisper timestamps have 20 ms resolution; tolerate one rounding tick.
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start <= end <= duration + 0.02:
        raise ValueError("Invalid timestamp in model output")


def stage(path: Path, data: bytes) -> Path:
    fd, name = tempfile.mkstemp(prefix=".transcription-", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_result(folder: Path, name: str, result: dict):
    """Stage both outputs before publishing; roll back on ordinary write errors."""
    payloads = {
        folder / f"{name}.txt": result["text"].encode("utf-8"),
        folder / f"{name}.json": json.dumps(result, ensure_ascii=False, indent=2,
                                           allow_nan=False).encode("utf-8"),
    }
    staged = {}
    backups = {}
    published = []
    try:
        for path, data in payloads.items():
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise PermissionError("Unsafe output target")
            backups[path] = stage(path, path.read_bytes()) if path.exists() else None
            staged[path] = stage(path, data)
        for path, tmp in staged.items():
            os.replace(tmp, path)
            published.append(path)
    except BaseException:
        for path in reversed(published):
            backup = backups[path]
            if backup is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(backup, path)
        raise
    finally:
        for tmp in list(staged.values()) + list(backups.values()):
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    error = folder / "error.log"
    if error.is_file() and not error.is_symlink():
        error.unlink()


def log_failure(folder: Path, message: str):
    print(f"ERROR: {folder.name}: {message}", file=sys.stderr)
    try:
        target = folder / "error.log"
        if folder.is_symlink() or target.is_symlink():
            raise PermissionError("Unsafe log target")
        tmp = stage(target, (message + "\n").encode("utf-8"))
        try:
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError:
        print("Cannot save error.log: check output permissions/disk space.", file=sys.stderr)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transcribe Russian audio with faster-whisper (no translation).")
    p.add_argument("--input", type=Path, default=BASE / "01_datasets_audio", help="Audio file or directory")
    p.add_argument("--output-dir", type=Path, default=BASE / "outputs", help="Results directory")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--model", default="large-v3", help="Model name or local model directory")
    p.add_argument("--word-timestamps", action="store_true", help="Include word start/end times")
    p.add_argument("--overwrite", action="store_true", help="Replace this audio's existing result files")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    ok = failed = skipped = 0
    try:
        files = discover(args.input)
    except (ValueError, OSError) as exc:
        print(str(exc) if isinstance(exc, ValueError) else safe_error(exc), file=sys.stderr)
        print("Summary: successful=0; failed=0; skipped=0; input_error=1")
        return 1
    pending = []
    for index, (audio, name) in enumerate(zip(files, output_names(files)), 1):
        folder = args.output_dir / name
        try:
            if folder.exists() and not args.overwrite:
                print(f"[{index}/{len(files)}] SKIP {audio.name}: output already exists: {folder}")
                skipped += 1
                continue
            if folder.is_symlink():
                raise PermissionError("Output directory is a symlink")
            folder.mkdir(parents=True, exist_ok=args.overwrite)
            pending.append((index, audio, name, folder))
        except OSError as exc:
            failed += 1
            log_failure(folder, safe_error(exc))
    engine = None
    fatal = None
    for index, audio, name, folder in pending:
        print(f"[{index}/{len(files)}] Processing: {audio.name}")
        try:
            if fatal:
                raise ModelError(fatal)
            if engine is None:
                engine = Engine(args.model, args.device)
            result = engine.transcribe(audio, args.word_timestamps)
            save_result(folder, name, result)
            if not result["segments"]:
                print("No speech detected; saved empty text and segments.")
            print(f"Saved: {folder.resolve()}")
            ok += 1
        except ModelError as exc:
            fatal = str(exc)  # Already sanitized by Engine.
            failed += 1
            log_failure(folder, fatal)
        except Exception as exc:
            failed += 1
            log_failure(folder, safe_error(exc))
    print(f"Summary: successful={ok}; failed={failed}; skipped={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

