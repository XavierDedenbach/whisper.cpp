#!/usr/bin/env python3
"""Local-only two-phase experiment; third-model drafts are never ground truth."""

import argparse
import json
import math
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from num2words import num2words

from phonon_probe import digest, phonon_hotwords, request
from phonon_reference_probe import score, self_test, tokens
from temporal_audio import prepared_padded_wav
from vocab_prompt import build_whisper_prompt, load_all_vocabulary, load_config


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def normalized(text):
    def expand(match):
        value = match.group().replace(",", "")
        ordinal = bool(re.search(r"(st|nd|rd|th)$", value))
        return num2words(
            int(value[:-2]) if ordinal else value,
            to="ordinal" if ordinal else "cardinal",
        )

    text = text.lower().replace("’", "'")
    text = re.sub(r"\b\d+(?:,\d{3})*(?:\.\d+)?(?:st|nd|rd|th)?\b", expand, text)
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text))


def test():
    self_test()
    assert normalized("15th, 3, 2.5") == normalized("fifteenth three two point five")
    assert normalized("RP2040 I2C") == "rp2040 i2c"
    assert normalized("Um, it's 15.") == "um it's fifteen"
    assert score(normalized("Hawkeye"), normalized("Hockey"))["errors"] == 1


def measured_score(reference, hypothesis, normalize=False):
    if normalize:
        reference, hypothesis = normalized(reference), normalized(hypothesis)
    return score(reference, hypothesis)


def draft(selection_path, model, output, reuse=None):
    import sherpa_onnx

    selection = json.loads(selection_path.read_text())
    if len(selection["session_ids"]) != 10:
        raise ValueError("expected the frozen ten-session selection")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows = []
    # Inspect all selected audio before any inference. Never use manifest text.
    for session_id in selection["session_ids"]:
        source = Path(selection["source_root"]) / session_id
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest["recording"]:
            raise ValueError("selected source is recording")
        pieces, chunks, offset = [], [], 0.0
        for chunk in sorted(manifest["chunks"], key=lambda item: item["index"]):
            path = source / chunk["wav"]
            samples, rate = sf.read(path, dtype="float32")
            if rate != 16000 or samples.ndim != 1 or not len(samples):
                raise ValueError(f"expected nonempty 16 kHz mono source: {path}")
            pieces.append(samples)
            chunks.append({"path": str(path), "sha256": digest(path), "start": offset})
            offset += len(samples) / rate
        if not pieces:
            raise ValueError("session has no retained audio")
        whole = output / (session_id + ".wav")
        sf.write(whole, np.concatenate(pieces), 16000, subtype="PCM_16")
        # Preserve current source-owned boundaries. Split only old >9s chunks,
        # which exceed the current context-512 inference window, into <=8s parts.
        parts = []
        for chunk_index, samples in enumerate(pieces):
            step = len(samples) if len(samples) <= 9 * 16000 else 8 * 16000
            for start in range(0, len(samples), step):
                path = output / f"{session_id}-{len(parts):03d}.wav"
                sf.write(path, samples[start : start + step], 16000, subtype="PCM_16")
                parts.append(
                    {
                        "wav": str(path),
                        "source_chunk": chunk_index,
                        "seconds": min(step, len(samples) - start) / 16000,
                        "sha256": digest(path),
                    }
                )
        rows.append(
            {
                "session_id": session_id,
                "wav": str(whole),
                "sha256": digest(whole),
                "sources": chunks,
                "parts": parts,
                "seconds": offset,
            }
        )
    report = {
        "reference_status": "provisional_machine_draft_not_ground_truth",
        "selection_sha256": digest(selection_path),
        "reference_model": "nvidia/canary-180m-flash, sherpa INT8",
        "sherpa_version": sherpa_onnx.__version__,
        "model_hashes": {
            name: digest(model / name)
            for name in ("encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt")
        },
        "reference_policy": "whole session, greedy CPU 4 threads, English, no hotwords or corrections",
        "sessions": rows,
    }
    prior_rows = {}
    if reuse:
        prior = json.loads(reuse.read_text())
        if prior["model_hashes"] != report["model_hashes"]:
            raise ValueError("cannot reuse drafts from another reference model")
        prior_rows = {row["session_id"]: row for row in prior["sessions"]}
        report["reused_from_sha256"] = digest(reuse)
    save(output / "inputs.json", report)
    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_canary(
        encoder=str(model / "encoder.int8.onnx"),
        decoder=str(model / "decoder.int8.onnx"),
        tokens=str(model / "tokens.txt"),
        src_lang="en",
        tgt_lang="en",
        num_threads=4,
        provider="cpu",
    )
    for index, row in enumerate(rows):
        previous = prior_rows.get(row["session_id"])
        if previous and previous["sha256"] == row["sha256"]:
            row["draft_reference"] = previous["draft_reference"]
            row["reference_seconds"] = previous["reference_seconds"]
            row["reused_unchanged"] = True
            save(output / "draft-progress.json", report)
            continue
        samples, rate = sf.read(row["wav"], dtype="float32")
        stream = recognizer.create_stream()
        started = time.monotonic()
        stream.accept_waveform(rate, samples)
        recognizer.decode_stream(stream)
        row["draft_reference"] = stream.result.text.strip()
        row["reference_seconds"] = time.monotonic() - started
        if not row["draft_reference"]:
            raise RuntimeError("empty third-model draft; no candidate run allowed")
        save(output / "draft-progress.json", report)
        print(
            json.dumps(
                {
                    "draft_session": index + 1,
                    "words": len(tokens(row["draft_reference"])),
                    "seconds": row["reference_seconds"],
                }
            ),
            flush=True,
        )
    report["frozen_at"] = datetime.now(timezone.utc).isoformat()
    baseline = output / "baseline.json"
    save(baseline, report)
    save(
        output / "baseline-seal.json",
        {"sha256": digest(baseline), "candidate_requests_started": False},
    )
    print("All ten provisional drafts frozen before candidate inference", flush=True)


def compare(output):
    baseline_path = output / "baseline.json"
    sealed = json.loads((output / "baseline-seal.json").read_text())["sha256"]
    if digest(baseline_path) != sealed:
        raise RuntimeError("baseline no longer matches its pre-comparison seal")
    baseline = json.loads(baseline_path.read_text())
    if (output / "comparison.json").exists():
        raise RuntimeError("comparison already exists; do not overwrite evidence")
    for row in baseline["sessions"]:
        if not tokens(row["draft_reference"]):
            raise ValueError(
                "unusable reference has no lexical words; no candidate run allowed"
            )
        for item in row["sources"] + row["parts"]:
            if digest(Path(item.get("path", item.get("wav")))) != item["sha256"]:
                raise RuntimeError("frozen source or inference input changed")
    cfg = load_config()
    words = phonon_hotwords(cfg)
    prompt = build_whisper_prompt(cfg)
    _, replacements = load_all_vocabulary(cfg)
    result = {
        "reference_status": baseline["reference_status"],
        "reference_sha256": sealed,
        "hotwords": words,
        "whisper_prompt": prompt,
        "replacements": replacements,
        "sessions": [],
        "summary": {},
        "load_before": Path("/proc/loadavg").read_text().strip(),
    }
    with prepared_padded_wav(
        Path(baseline["sessions"][0]["parts"][0]["wav"]), 0.5
    ) as audio:
        result["warmup"] = {
            engine: request(
                engine, audio, prompt, hotwords=words, replacements=replacements
            )
            for engine in ("phonon", "whisper")
        }
    part_count = 0
    for index, row in enumerate(baseline["sessions"]):
        measured = {
            "session_id": row["session_id"],
            "wav": row["wav"],
            "draft_reference": row["draft_reference"],
            "parts": [],
        }
        for part in row["parts"]:
            part_result = dict(part)
            with prepared_padded_wav(Path(part["wav"]), 0.5) as audio:
                engines = (
                    ("phonon", "whisper")
                    if part_count % 2 == 0
                    else ("whisper", "phonon")
                )
                for engine in engines:
                    part_result[engine] = request(
                        engine, audio, prompt, hotwords=words, replacements=replacements
                    )
            measured["parts"].append(part_result)
            part_count += 1
        for engine in ("phonon", "whisper"):
            measured[engine] = {}
            for kind, field in (("raw", "text"), ("corrected", "corrected_text")):
                text = " ".join(part[engine][field] for part in measured["parts"])
                measured[engine][field] = text
                for norm in (False, True):
                    key = kind + ("_normalized" if norm else "_strict")
                    measured[engine][key] = measured_score(
                        row["draft_reference"], text, norm
                    )
        measured["listening_review_required"] = True
        measured["candidate_disagreement"] = measured_score(
            measured["phonon"]["text"], measured["whisper"]["text"], True
        )
        result["sessions"].append(measured)
        save(output / "comparison.json", result)
        print(
            json.dumps(
                {
                    "compared_session": index + 1,
                    "chunks": len(measured["parts"]),
                    "phonon_edits": measured["phonon"]["raw_normalized"]["errors"],
                    "whisper_edits": measured["whisper"]["raw_normalized"]["errors"],
                }
            ),
            flush=True,
        )
    for engine in ("phonon", "whisper"):
        times = sorted(
            part[engine]["seconds"]
            for row in result["sessions"]
            for part in row["parts"]
        )
        summary = {
            "requests": len(times),
            "median_seconds": statistics.median(times),
            "p95_seconds": times[math.ceil(len(times) * 0.95) - 1],
        }
        for kind in (
            "raw_strict",
            "raw_normalized",
            "corrected_strict",
            "corrected_normalized",
        ):
            scores = [row[engine][kind] for row in result["sessions"]]
            errors = sum(item["errors"] for item in scores)
            count = sum(item["reference_words"] for item in scores)
            summary[kind] = {
                "edits": errors,
                "draft_words": count,
                "disagreement_rate": errors / count,
            }
        result["summary"][engine] = summary
    result["load_after"] = Path("/proc/loadavg").read_text().strip()
    result["audio_seconds"] = sum(row["seconds"] for row in baseline["sessions"])
    result["baseline_unchanged"] = digest(baseline_path) == sealed
    result["source_hashes_unchanged"] = all(
        digest(Path(item["path"])) == item["sha256"]
        for row in baseline["sessions"]
        for item in row["sources"]
    )
    save(output / "comparison.json", result)
    print(json.dumps(result["summary"], indent=2))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("draft", "compare", "self-test"))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    test()
    if args.phase == "draft":
        draft(args.selection, args.model, args.output, args.reuse)
    elif args.phase == "compare":
        compare(args.output)
    else:
        print("scoring self-tests passed")


if __name__ == "__main__":
    main()
