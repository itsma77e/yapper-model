"""Freeze a test set of FLEURS recordings the Yapper research never touched.

A row qualifies only if its prompt ID and its text appear in no research manifest
for any language. Selection is by hash, so it does not depend on model results.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

LANGUAGES = {"it_it": "it", "en_us": "en", "es_419": "es", "fr_fr": "fr", "de_de": "de", "ja_jp": "ja",
             "cmn_hans_cn": "zh", "ko_kr": "ko", "pt_br": "pt", "ru_ru": "ru", "ar_eg": "ar", "hi_in": "hi"}
LARGE = {"it_it": 100, "en_us": 100}
REVISION = "0b98f0538af50329c9b336824d7d697c594b9c26"


def seen(research: Path):
    prompts, texts = set(), set()
    for path in sorted((research / "data/plus").glob("*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("prompt_id") is not None:
                prompts.add(row["prompt_id"])
            if row.get("reference"):
                texts.add(row["reference"].strip().casefold())
    return prompts, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--small", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = root / "fresh-manifest.json"
    if out.exists():
        raise FileExistsError(out)
    prompts, texts = seen(args.research)
    rows, counts = [], {}
    for locale, language in LANGUAGES.items():
        archive = next((args.research / "data/final-v3" / locale).glob("test-*.parquet"))
        candidates = {}
        for batch in pq.ParquetFile(archive).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                text = row["transcription"].strip()
                if row["id"] in prompts or text.casefold() in texts or row["id"] in candidates:
                    continue
                candidates[row["id"]] = row
        order = sorted(candidates, key=lambda key: hashlib.sha256(f"{locale}:{key}".encode()).hexdigest())
        folder = root / "data" / locale
        folder.mkdir(parents=True, exist_ok=True)
        kept = 0
        for key in order:
            if kept == LARGE.get(locale, args.small):
                break
            row = candidates[key]
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if audio.ndim != 1 or not 3 <= len(audio) / rate <= 28:
                continue
            stream = io.BytesIO()
            sf.write(stream, audio, rate, format="WAV", subtype="PCM_16")
            payload = stream.getvalue()
            path = folder / f"prompt-{key}.wav"
            path.write_bytes(payload)
            rows.append(dict(id=f"{locale}-{key}", language=language, locale=locale, prompt_id=key,
                             audio=path.relative_to(root).as_posix(), audio_sha256=hashlib.sha256(payload).hexdigest(),
                             duration=len(audio) / rate, reference=row["transcription"].strip(),
                             source=f"google/fleurs@{REVISION}/{locale}/test", license="CC-BY-4.0"))
            kept += 1
        counts[locale] = dict(unused_prompts=len(order), kept=kept)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(counts), len(rows), round(sum(r["duration"] for r in rows)), "seconds")


if __name__ == "__main__":
    main()
