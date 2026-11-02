"""Compare Yapper with Whisper large-v3-turbo on the frozen FLEURS set.

Arms use the parameters of Yapper's final transcription pass:
  turbo         large-v3-turbo with standard faster-whisper
  turbo-crop    the same weights with Yapper's encoder crop
  yapper        Yapper weights with the crop, as shipped in the app
  yapper-plain  Yapper weights with standard faster-whisper
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ["turbo", "turbo-crop", "yapper", "yapper-plain"]


def words(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, flags=re.UNICODE)


def chars(text):
    return [c for c in unicodedata.normalize("NFKC", text).casefold() if unicodedata.category(c)[0] in "LNM"]


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def score(reference, hypothesis):
    w, c = words(reference), chars(reference)
    return dict(word_errors=distance(w, words(hypothesis)), words=len(w),
                char_errors=distance(c, chars(hypothesis)), chars=len(c))


def crop_frames(samples):
    """Same bound as FasterBackend.transcribe in the Yapper engine."""
    return min(1500, max(400, ((samples + 15999) // 16000 + 1) * 50))


class Engine:
    def __init__(self, folder, device, threads):
        import numpy as np
        from faster_whisper import WhisperModel
        self.np = np
        compute = "int8_float16" if device == "cuda" else "int8"
        self.model = WhisperModel(str(folder), device=device, compute_type=compute,
                                  cpu_threads=threads, num_workers=1, local_files_only=True)
        self.full_encode = self.model.encode
        self.frames = None
        self.model.encode = self.encode

    def encode(self, features):
        if self.frames is None:
            return self.full_encode(features)
        return self.full_encode(self.np.ascontiguousarray(features[..., :2 * self.frames]))

    def run(self, audio, language, cropped):
        self.frames = crop_frames(len(audio)) if cropped else None
        segments, info = self.model.transcribe(
            audio, language=language, task="transcribe", beam_size=3, best_of=1, temperature=0,
            condition_on_previous_text=False, vad_filter=False, no_speech_threshold=.6,
            log_prob_threshold=-1, compression_ratio_threshold=2.4,
            **({"without_timestamps": True} if cropped else {}))
        return " ".join(s.text.strip() for s in segments).strip(), info.language


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summarize(rows, key):
    result = {}
    for language in sorted({r["language"] for r in rows}) + ["all"]:
        subset = [r for r in rows if language == "all" or r["language"] == language]
        audio = sum(r["duration"] for r in subset)
        result[language] = {}
        for arm in ARMS:
            runs = [r[key][arm] for r in subset]
            errors, total = sum(x["word_errors"] for x in runs), sum(x["words"] for x in runs)
            cer, count = sum(x["char_errors"] for x in runs), sum(x["chars"] for x in runs)
            seconds = sum(statistics.median(x["seconds"]) for x in runs)
            result[language][arm] = dict(wer=errors / total, word_errors=errors, words=total, cer=cer / count,
                                         seconds=seconds, audio_seconds=audio, realtime=audio / seconds,
                                         wrong_language=sum(x["detected"] != r["language"] for x, r in zip(runs, subset)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yapper", type=Path, required=True)
    parser.add_argument("--turbo", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--languages", default="")
    parser.add_argument("--per-language", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--auto", action="store_true", help="also run with language detection")
    parser.add_argument("--cuda-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cuda_dir:
        os.add_dll_directory(str(args.cuda_dir))
        os.environ["PATH"] = str(args.cuda_dir) + os.pathsep + os.environ["PATH"]
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = ROOT / "fresh-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keep = set(args.languages.split(",")) if args.languages else None
    rows, used = [], {}
    for row in manifest:
        used[row["language"]] = used.get(row["language"], 0) + 1
        if keep and row["language"] not in keep:
            continue
        if args.per_language and used[row["language"]] > args.per_language:
            continue
        rows.append(row)

    from faster_whisper.audio import decode_audio
    import ctranslate2
    import faster_whisper
    engines = {"turbo": Engine(args.turbo, args.device, args.threads), "yapper": Engine(args.yapper, args.device, args.threads)}
    setup = {"turbo": ("turbo", False), "turbo-crop": ("turbo", True), "yapper": ("yapper", True), "yapper-plain": ("yapper", False)}
    import numpy as np
    for engine in engines.values():
        engine.run(np.zeros(16000, np.float32), "en", True)

    report = dict(complete=False, device=args.device, threads=args.threads, repetitions=args.repetitions,
                  manifest_sha256=digest(manifest_path), runtime=dict(faster_whisper=faster_whisper.__version__, ctranslate2=ctranslate2.__version__),
                  models={name: dict(sha256=digest(folder / "model.bin"), bytes=(folder / "model.bin").stat().st_size)
                          for name, folder in (("turbo", args.turbo), ("yapper", args.yapper))}, rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run_arms(audio, language, index, repetitions):
        runs = {arm: [] for arm in ARMS}
        for repetition in range(repetitions):
            shift = (index + repetition) % len(ARMS)
            order = ARMS[shift:] + ARMS[:shift]
            if repetition % 2:
                order.reverse()
            for arm in order:
                name, cropped = setup[arm]
                start = time.perf_counter()
                text, detected = engines[name].run(audio, language, cropped)
                runs[arm].append(dict(text=text, detected=detected, seconds=time.perf_counter() - start))
        return runs

    for index, row in enumerate(rows):
        path = ROOT / row["audio"]
        if digest(path) != row["audio_sha256"]:
            raise ValueError("Audio changed: " + row["id"])
        audio = decode_audio(str(path))
        record = {k: row[k] for k in ("id", "language", "duration", "reference")}
        for key, language, repetitions in [("known", row["language"], args.repetitions)] + ([("auto", None, 1)] if args.auto else []):
            runs = run_arms(audio, language, index, repetitions)
            record[key] = {arm: dict(text=values[0]["text"], detected=values[0]["detected"],
                                     seconds=[v["seconds"] for v in values], **score(row["reference"], values[0]["text"]))
                           for arm, values in runs.items()}
        report["rows"].append(record)
        if index % 10 == 9:
            print(index + 1, "/", len(rows), flush=True)
    report["summary"] = {key: summarize(report["rows"], key) for key in (["known", "auto"] if args.auto else ["known"])}
    report["complete"] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    all_known = report["summary"]["known"]["all"]
    print(json.dumps({arm: {k: round(v, 4) if isinstance(v, float) else v for k, v in all_known[arm].items()} for arm in ARMS}, indent=1))


if __name__ == "__main__":
    main()
