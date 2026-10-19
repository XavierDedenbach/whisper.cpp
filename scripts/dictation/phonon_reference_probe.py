#!/usr/bin/env python3
"""Score both local engines against frozen LibriSpeech human-speech references."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import soundfile as sf

from phonon_probe import digest, phonon_hotwords, request
from temporal_audio import prepared_padded_wav
from vocab_prompt import build_whisper_prompt, load_all_vocabulary, load_config


def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower())


def score(reference, hypothesis):
    ref, hyp = tokens(reference), tokens(hypothesis)
    # Tuples retain an explicit alignment, with deterministic substitution ties.
    table = [[None] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    table[0][0] = (0, [])
    for i in range(1, len(ref) + 1):
        table[i][0] = (i, table[i - 1][0][1] + [("delete", ref[i - 1], "")])
    for j in range(1, len(hyp) + 1):
        table[0][j] = (j, table[0][j - 1][1] + [("insert", "", hyp[j - 1])])
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                candidates = [
                    (
                        table[i - 1][j - 1][0] + 1,
                        table[i - 1][j - 1][1]
                        + [("substitute", ref[i - 1], hyp[j - 1])],
                    ),
                    (
                        table[i - 1][j][0] + 1,
                        table[i - 1][j][1] + [("delete", ref[i - 1], "")],
                    ),
                    (
                        table[i][j - 1][0] + 1,
                        table[i][j - 1][1] + [("insert", "", hyp[j - 1])],
                    ),
                ]
                table[i][j] = min(candidates, key=lambda item: item[0])
    errors, edits = table[-1][-1]
    return {
        "reference_words": len(ref),
        "errors": errors,
        "wer": errors / max(1, len(ref)),
        "edits": edits,
    }


def self_test():
    assert score("Hello, WORLD!", "hello world")["errors"] == 0
    assert score("one two three", "one five three")["edits"] == [
        ("substitute", "two", "five")
    ]
    assert score("one two three", "one three")["edits"] == [("delete", "two", "")]
    assert score("one three", "one two three")["edits"] == [("insert", "", "two")]
    assert score("one two", "")["errors"] == 2


def select(dataset):
    candidates = {}
    for transcript in sorted(dataset.rglob("*.trans.txt")):
        for line in transcript.read_text().splitlines():
            identity, text = line.split(" ", 1)
            if not re.fullmatch(r"\d+-\d+-\d+", identity):
                raise ValueError(f"unexpected utterance ID: {identity}")
            candidates[identity] = (transcript.with_name(identity + ".flac"), text)
    chosen, speakers = [], set()
    for identity in sorted(candidates):
        speaker = identity.split("-")[0]
        if speaker in speakers:
            continue
        path, reference = candidates[identity]
        audio = sf.info(path)
        if audio.samplerate != 16000 or audio.channels != 1:
            raise ValueError("expected unmodified 16 kHz mono corpus audio")
        if 4 <= audio.duration <= 9:
            chosen.append(
                {
                    "id": identity,
                    "speaker": speaker,
                    "source": str(path),
                    "reference": reference,
                    "duration": audio.duration,
                    "sha256": digest(path),
                }
            )
            speakers.add(speaker)
            if len(chosen) == 30:
                return chosen
    raise RuntimeError(f"only {len(chosen)} qualifying speakers; no inference started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    self_test()
    selected = select(args.dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    # Freeze IDs, references and hashes before any model request.
    (args.output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    cfg = load_config()
    prompt = build_whisper_prompt(cfg)
    words, (_, replacements) = phonon_hotwords(cfg), load_all_vocabulary(cfg)
    report = {
        "dataset": "OpenSLR12 LibriSpeech test-clean",
        "self_test": "pass",
        "hotwords": words,
        "padding_seconds": 0.5,
        "clips": [],
        "summary": {},
        "load_before": Path("/proc/loadavg").read_text().strip(),
    }
    for index, item in enumerate(selected):
        audio = args.output / (item["id"] + ".wav")
        samples, rate = sf.read(item["source"], dtype="int16")
        sf.write(audio, samples, rate, subtype="PCM_16")
        with prepared_padded_wav(audio, 0.5) as padded:
            if index == 0:
                report["warmup"] = {
                    engine: request(
                        engine,
                        padded,
                        prompt,
                        hotwords=words,
                        replacements=replacements,
                    )
                    for engine in ("phonon", "whisper")
                }
            row = dict(item, wav=str(audio))
            for engine in (
                ("phonon", "whisper") if index % 2 == 0 else ("whisper", "phonon")
            ):
                result = request(
                    engine, padded, prompt, hotwords=words, replacements=replacements
                )
                result["raw_score"] = score(item["reference"], result["text"])
                result["corrected_score"] = score(
                    item["reference"], result["corrected_text"]
                )
                row[engine] = result
        if digest(Path(item["source"])) != item["sha256"]:
            raise RuntimeError("source audio changed")
        report["clips"].append(row)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "clip": index + 1,
                    "id": item["id"],
                    **{
                        engine: {
                            "seconds": round(row[engine]["seconds"], 3),
                            "errors": row[engine]["raw_score"]["errors"],
                        }
                        for engine in ("phonon", "whisper")
                    },
                }
            ),
            flush=True,
        )
    for engine in ("phonon", "whisper"):
        times = sorted(row[engine]["seconds"] for row in report["clips"])
        summary = {
            "median_seconds": statistics.median(times),
            "p95_seconds": times[math.ceil(len(times) * 0.95) - 1],
        }
        for kind in ("raw", "corrected"):
            scored = [row[engine][kind + "_score"] for row in report["clips"]]
            ref_words = sum(item["reference_words"] for item in scored)
            errors = sum(item["errors"] for item in scored)
            summary[kind] = {
                "reference_words": ref_words,
                "errors": errors,
                "wer": errors / ref_words,
                "exact_utterances": sum(item["errors"] == 0 for item in scored),
                "substitutions": sum(
                    edit[0] == "substitute" for item in scored for edit in item["edits"]
                ),
                "deletions": sum(
                    edit[0] == "delete" for item in scored for edit in item["edits"]
                ),
                "insertions": sum(
                    edit[0] == "insert" for item in scored for edit in item["edits"]
                ),
            }
        report["summary"][engine] = summary
    report["load_after"] = Path("/proc/loadavg").read_text().strip()
    report["audio_seconds"] = sum(item["duration"] for item in selected)
    report["acceptance"] = {
        "phonon_raw_wer_no_higher": report["summary"]["phonon"]["raw"]["wer"]
        <= report["summary"]["whisper"]["raw"]["wer"],
        "phonon_p95_below_two_seconds": report["summary"]["phonon"]["p95_seconds"] < 2,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"summary": report["summary"], "acceptance": report["acceptance"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
