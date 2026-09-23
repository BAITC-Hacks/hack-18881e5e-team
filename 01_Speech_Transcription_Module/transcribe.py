"""Transcribe Russian audio to per-file TXT/JSON folders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
FORMATS = {".mp3", ".wav", ".m4a", ".flac"}


def collect_audio(source):
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in FORMATS:
            raise ValueError("Unsupported audio format: " + source.suffix)
        return [source]
    if not source.is_dir():
        raise ValueError("Input path does not exist: " + str(source))
    files = sorted(
        (p for p in source.iterdir() if p.is_file() and p.suffix.lower() in FORMATS),
        key=lambda p: p.name.casefold(),
    )
    if not files:
        raise ValueError("No supported audio files found.")
    return files


def output_names(files):
    """Resolve duplicate stems, including case-insensitive collisions."""
    names, used = {}, set()
    counts = {}
    for path in files:
        key = path.stem.casefold()
        counts[key] = counts.get(key, 0) + 1
    for path in files:
        name = path.stem
        if counts[name.casefold()] > 1:
            name += "_" + path.suffix[1:].lower()
        if name.casefold() in used:
            digest = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:10]
            name += "_" + digest
        while name.casefold() in used:
            name += "_"
        names[path] = name
        used.add(name.casefold())
    return names


def save_result(folder, name, payload):
    # Stage both complete files before replacing destinations.
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=folder) as staging:
        staging = Path(staging)
        (staging / (name + ".txt")).write_text(payload["text"], encoding="utf-8")
        (staging / (name + ".json")).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for suffix in (".txt", ".json"):
            os.replace(staging / (name + suffix), folder / (name + suffix))
    (folder / "error.log").unlink(missing_ok=True)


def load_model(args):
    import ctranslate2
    from faster_whisper import WhisperModel

    device = args.device
    if device == "auto":
        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        except Exception:
            device = "cpu"
    print("Loading model:", args.model, "| device:", device, flush=True)
    try:
        return WhisperModel(
            args.model, device=device,
            compute_type="float16" if device == "cuda" else "int8",
            cpu_threads=max(1, os.cpu_count() or 1),
        )
    except Exception:
        if args.device != "auto" or device != "cuda":
            raise
        print("GPU model initialization failed; retrying on CPU.", flush=True)
        return WhisperModel(args.model, device="cpu", compute_type="int8",
                            cpu_threads=max(1, os.cpu_count() or 1))


def transcribe_audio(model, audio, model_name, word_timestamps):
    segments, info = model.transcribe(
        str(audio), language="ru", task="transcribe", beam_size=5,
        vad_filter=True, word_timestamps=word_timestamps,
    )
    rows = []
    for segment in segments:
        row = {"id": len(rows), "start": segment.start,
               "end": segment.end, "text": segment.text.strip()}
        if word_timestamps:
            row["words"] = [
                {"word": w.word.strip(), "start": w.start, "end": w.end}
                for w in (segment.words or [])
            ]
        rows.append(row)
        print(f"  Processed audio through {segment.end:.1f}s", flush=True)
    return {
        "audio_file": audio.name, "model": model_name, "language": "ru",
        "duration_seconds": info.duration,
        "text": " ".join(row["text"] for row in rows if row["text"]),
        "segments": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "01_datasets_audio")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        files = collect_audio(args.input)
        output = args.output_dir.resolve()
        # Protect input audio from accidental overlap with generated results.
        if any(p.is_relative_to(output) for p in files):
            raise ValueError("Output directory must not contain input audio.")
        names = output_names(files)
        output.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    model = None
    succeeded = failed = skipped = 0
    for index, audio in enumerate(files, 1):
        name = names[audio]
        folder = output / name
        if folder.exists() and not args.overwrite:
            print("Skipping existing output:", folder, flush=True)
            skipped += 1
            continue
        print(f"[{index}/{len(files)}] {audio.name}", flush=True)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if model is None:
                model = load_model(args)
            payload = transcribe_audio(model, audio, args.model, args.word_timestamps)
            save_result(folder, name, payload)
            if not payload["segments"]:
                print("  No speech detected.", flush=True)
            print("Saved:", folder, flush=True)
            succeeded += 1
        except Exception as error:
            failed += 1
            # Avoid printing exception strings that may include signed URLs/tokens.
            message = (
                f"{type(error).__name__}: transcription failed. "
                "Check audio validity, model download, available memory and device libraries.\n"
            )
            print(message, file=sys.stderr)
            try:
                (folder / "error.log").write_text(message, encoding="utf-8")
            except OSError:
                print("Could not write error.log.", file=sys.stderr)
    print(f"Done: successful={succeeded}, failed={failed}, skipped={skipped}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
