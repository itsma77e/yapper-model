---
license: mit
base_model: openai/whisper-large-v3-turbo
pipeline_tag: automatic-speech-recognition
library_name: ctranslate2
tags:
  - whisper
  - faster-whisper
  - ctranslate2
  - speech-to-text
  - dictation
language:
  - it
  - en
  - es
  - fr
  - de
  - pt
  - ru
  - ja
  - zh
  - ko
  - ar
  - hi
---

# Yapper Turbo

The speech-to-text model inside [Yapper](https://itsyapper.com), a local dictation app. It is an INT8 CTranslate2 model derived from OpenAI Whisper large-v3-turbo, adapted to stay accurate when the encoder only processes the audio that exists rather than a full 30-second window.

On short clips it runs about twice as fast as large-v3-turbo with close accuracy. The gain depends on that crop: with standard faster-whisper it runs at about the same speed as large-v3-turbo, so use the included `yapper_asr.py`.

## Usage

```bash
pip install -r requirements.txt
```

Use these versions: with PyAV 19, faster-whisper 1.2.1 fails to open audio.

```python
from yapper_asr import YapperModel

model = YapperModel("path/to/model")
print(model.transcribe("audio.wav", language="it"))
```

Keep all model files together. On GPU, faster-whisper needs CUDA 12 libraries.

## Results

Tested on 400 recordings never used in the research behind this model: FLEURS official test split, 12 languages, 3.96 to 27.84 s, median 12.06 s. The set and the code were committed before the full run. All runs use INT8 weights (`int8_float16` on GPU), language given, beam 3, on an RTX 3060 Laptop GPU and on a CPU with 4 threads.

| Configuration | CER, 400 clips | WER, IT | WER, EN | Speed, GPU, all | Speed, CPU, IT+EN (60 clips) |
| --- | ---: | ---: | ---: | ---: | ---: |
| large-v3-turbo | 2.81% | 2.96% | 5.02% | 22.8x realtime | 1.51x realtime |
| large-v3-turbo with crop | 8.24% | 14.63% | 5.86% | 44.3x | 4.04x |
| **Yapper Turbo with crop** | **3.04%** | **3.08%** | **5.20%** | **48.1x** | **4.40x** |
| Yapper Turbo, standard faster-whisper | 2.86% | 2.96% | 5.33% | 23.8x | 1.63x |

Paired with large-v3-turbo, with 95% bootstrap intervals over clips:

| Scope | Speed | WER difference |
| --- | ---: | ---: |
| GPU, all languages, 400 clips | 2.11x [2.05, 2.17] | +0.53 points [+0.10, +0.98] |
| GPU, Italian, 100 clips | 1.84x [1.76, 1.93] | +0.12 points [-0.40, +0.64] |
| GPU, English, 100 clips | 2.67x [2.53, 2.81] | +0.18 points [-0.59, +0.94] |
| CPU, Italian + English, 60 clips | 2.91x [2.65, 3.22] | +0.00 points [-0.80, +0.72] |

Size: 774.6 MB against 1,617.9 MB for the float16 large-v3-turbo conversion used here. That difference mostly reflects INT8 storage; an INT8 export of large-v3-turbo was not measured and should be of similar size.

Accuracy is not uniform. On the 20 clips per language, French, Spanish and Hindi are worse than large-v3-turbo (CER 3.12% vs 2.48%, 1.93% vs 0.70%, 14.60% vs 12.61%). With language detection, Hindi degrades further (WER 44.38% vs 37.12%). WER is not meaningful for Chinese and Japanese here; use CER. Per-language results are in `eval/results/`.

## Limitations

- FLEURS is clean read speech. These numbers do not cover spontaneous speech, noise, accents or poor microphones.
- The speed gain shrinks with duration and is gone from about 29 s, where the crop reaches the full window.
- The small per-language samples, apart from Italian and English, only indicate direction.
- Timings come from one laptop that was not isolated, and they are only comparable within this table.
- Do not use it where transcription errors carry serious consequences without human review.

## How it was made

The encoder comes from an earlier Yapper model derived from large-v3-turbo. The decoder was adapted on 192 FLEURS Italian and English training recordings in three acoustic conditions, for 576 optimization steps, then merged and exported to INT8. All recordings in the research manifests are from FLEURS, and no personal audio was used.

Reproduce the evaluation with `eval/prepare_fresh.py`, `eval/benchmark.py` and `eval/summarize.py`. This release includes weights, usage code and evaluation. It does not include training code.

## Related work

Shortening Whisper's encoder context is not new: whisper.cpp exposes it as `audio_ctx`, and [FUTO's whisper-acft](https://github.com/futo-org/whisper-acft) fine-tunes Whisper models so they tolerate a dynamic context. This release applies the same general idea to large-v3-turbo with an INT8 runtime. It has not been compared with whisper-acft models.

## License

Weights and code are MIT. See `LICENSE` and `NOTICE`. The Yapper name and logo are not licensed.

## Citation

Please credit Whisper (Radford et al., 2022) and FLEURS (Conneau et al., 2022).
