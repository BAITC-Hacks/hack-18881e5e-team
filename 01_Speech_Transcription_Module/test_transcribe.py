"""Offline checks: do not load or download an ASR model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import transcribe


class TranscriptionTests(unittest.TestCase):
    def test_duplicate_names(self):
        files = [Path("a.mp3"), Path("a.wav"), Path("a_mp3.flac")]
        names = transcribe.output_names(files)
        self.assertEqual(len({n.casefold() for n in names.values()}), 3)

    def test_outputs_and_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "input"
            audio.mkdir()
            (audio / "a.mp3").touch()
            output = root / "outputs"
            payload = {"audio_file": "a.mp3", "text": "Привет.", "segments": []}
            transcribe.save_result(output / "a", "a", payload)
            text = (output / "a" / "a.txt").read_text(encoding="utf-8")
            data = json.loads((output / "a" / "a.json").read_text(encoding="utf-8"))
            self.assertEqual(text, data["text"])
            with patch.object(transcribe, "load_model") as loader:
                self.assertEqual(transcribe.main(["--input", str(audio),
                    "--output-dir", str(output)]), 0)
                loader.assert_not_called()

    def test_failed_file_does_not_stop_next(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "input"
            audio.mkdir()
            for name in ("a.mp3", "b.mp3"):
                (audio / name).touch()
            output = root / "outputs"
            payload = {"text": "Здравствуйте.", "segments": [{"text": "Здравствуйте."}]}
            with patch.object(transcribe, "load_model", return_value=object()), \
                 patch.object(transcribe, "transcribe_audio",
                              side_effect=[ValueError("bad audio"), payload]):
                result = transcribe.main(["--input", str(audio),
                    "--output-dir", str(output)])
            self.assertEqual(result, 1)
            self.assertTrue((output / "a" / "error.log").exists())
            self.assertTrue((output / "b" / "b.json").exists())

    def test_missing_input(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                transcribe.collect_audio(Path(temp) / "missing")


if __name__ == "__main__":
    unittest.main()
