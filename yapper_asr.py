"""Run the Yapper model the way the Yapper app runs it.

Most of the speed comes from encoding only the audio that exists instead of a full
30-second window. The weights were adapted so accuracy survives that crop, so use
both together. Requires faster-whisper.
"""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio


def crop_frames(samples: int) -> int:
    """Encoder frames for an audio length at 16 kHz: at least 8 s, at most 30 s."""
    return min(1500, max(400, ((samples + 15999) // 16000 + 1) * 50))


class YapperModel:
    def __init__(self, folder: str, device: str = "auto", threads: int = 4):
        compute = "int8" if device == "cpu" else "default"
        self.model = WhisperModel(folder, device=device, compute_type=compute, cpu_threads=threads)
        self._encode = self.model.encode
        self._frames = 1500
        self.model.encode = self._cropped_encode

    def _cropped_encode(self, features):
        return self._encode(np.ascontiguousarray(features[..., :2 * self._frames]))

    def transcribe(self, audio, language: str | None = None, beam_size: int = 3) -> str:
        """Transcribe a file path or 16 kHz mono float32 samples. None detects the language."""
        samples = decode_audio(audio) if isinstance(audio, str) else audio
        self._frames = crop_frames(len(samples))
        segments, _ = self.model.transcribe(
            samples, language=language, beam_size=beam_size, best_of=1, temperature=0,
            condition_on_previous_text=False, without_timestamps=True)
        return " ".join(segment.text.strip() for segment in segments).strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("audio")
    parser.add_argument("--language")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(YapperModel(args.model, args.device).transcribe(args.audio, args.language))
