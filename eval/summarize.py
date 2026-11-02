"""Print the benchmark results as tables, with paired statistics between arms."""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ["turbo", "turbo-crop", "yapper", "yapper-plain"]


def table(summary, languages):
    print("lang | arm | WER | CER | wrong language | speed x realtime")
    for language in languages:
        for arm in ARMS:
            v = summary[language][arm]
            print(f"{language} | {arm} | {v['wer']:.2%} ({v['word_errors']}/{v['words']}) | {v['cer']:.2%} | {v['wrong_language']} | {v['realtime']:.2f}x")


def paired(rows, key, a, b):
    """Bootstrap the WER difference and the speed ratio of a over b, resampling clips."""
    random.seed(20261102)
    def stats(sample):
        words = sum(r[key][a]["words"] for r in sample)
        difference = (sum(r[key][a]["word_errors"] for r in sample) - sum(r[key][b]["word_errors"] for r in sample)) / words
        ratio = sum(statistics.median(r[key][b]["seconds"]) for r in sample) / sum(statistics.median(r[key][a]["seconds"]) for r in sample)
        return difference, ratio
    samples = [stats(random.choices(rows, k=len(rows))) for _ in range(2000)]
    interval = lambda values: (sorted(values)[50], sorted(values)[1949])
    point = stats(rows)
    low, high = interval([s[0] for s in samples])
    slow, fast = interval([s[1] for s in samples])
    print(f"{a} vs {b}: WER difference {point[0]:+.2%} [{low:+.2%}, {high:+.2%}], speed {point[1]:.2f}x [{slow:.2f}, {fast:.2f}]")


def main():
    for name in sys.argv[1:]:
        report = json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))
        print(f"\n=== {name}: {report['device']}, {len(report['rows'])} clips, {report['models']}")
        for key, summary in report["summary"].items():
            print(f"\n--- language {key}")
            table(summary, sorted(summary, key=lambda x: (x != "all", x)))
            for subset in ["all", "it", "en"]:
                rows = [r for r in report["rows"] if subset == "all" or r["language"] == subset]
                print(f"paired, {subset}, {len(rows)} clips")
                paired(rows, key, "yapper", "turbo")
                paired(rows, key, "yapper", "turbo-crop")
                paired(rows, key, "turbo-crop", "turbo")
                paired(rows, key, "yapper-plain", "turbo")
        durations = [r["duration"] for r in report["rows"]]
        print("durations", min(durations), statistics.median(durations), max(durations))


if __name__ == "__main__":
    main()
