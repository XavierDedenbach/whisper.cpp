#!/usr/bin/env python3
"""Offline development-only comparison; never records or pastes desktop input."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
import wave
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from numeric_continuity_probe import score_transcript, self_test
from temporal_audio import prepared_padded_wav
from vocab_prompt import (
    apply_replacements,
    build_phonon_hotwords,
    build_whisper_prompt,
    load_all_vocabulary,
    load_config,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phonon_hotwords(cfg: dict[str, str] | None = None) -> list[str]:
    return build_phonon_hotwords(cfg)


def request(
    engine: str,
    audio: Path,
    prompt: str,
    *,
    hotwords: list[str] | None = None,
    replacements: list[tuple[str, str]] | None = None,
) -> dict:
    port, route = (
        (8180, "/v1/audio/transcriptions")
        if engine == "phonon"
        else (8178, "/inference")
    )
    command = [
        "curl",
        "--fail-with-body",
        "-sS",
        "--connect-timeout",
        "2",
        "--max-time",
        "60",
        f"http://127.0.0.1:{port}{route}",
        "-F",
        f"file=@{audio}",
        "-F",
        "response_format=verbose_json"
        if engine == "phonon"
        else "response_format=json",
    ]
    if engine == "phonon":
        words = phonon_hotwords() if hotwords is None else hotwords
        if len(words) > 25:
            raise ValueError("Phonon-2 supports at most 25 active hotwords")
        command += ["-F", "model=phonon-2", "-F", f"hotwords={json.dumps(words)}"]
    else:
        for field in (
            "temperature=0.0",
            "no_timestamps=true",
            "token_timestamps=false",
            "language=en",
            "beam_size=5",
            "audio_ctx=512",
            "suppress_nst=true",
            "carry_initial_prompt=true",
            f"prompt={prompt}",
        ):
            command += ["-F", field]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=65)
    seconds = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(f"{engine} failed: {result.stderr} {result.stdout[:1000]}")
    payload = json.loads(result.stdout)
    if not isinstance(payload.get("text"), str):
        raise RuntimeError(f"invalid {engine} response: {payload}")
    if replacements is None:
        _, replacements = load_all_vocabulary()
    raw_text = payload["text"].strip()
    return {
        "seconds": seconds,
        "text": raw_text,
        "corrected_text": apply_replacements(raw_text, replacements),
        "response": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The predeclared final holdout must never become a development input.
    if "final-500" in str(args.session) or "holdout" in str(args.session).lower():
        parser.error("sealed holdout is excluded")
    self_test()
    manifest = json.loads((args.session / "manifest.json").read_text())
    chunks = [
        args.session / chunk["wav"]
        for chunk in sorted(manifest["chunks"], key=lambda item: item["index"])
        if chunk["status"] == "complete"
    ]
    if len(chunks) != 19:
        raise RuntimeError("expected the declared 19-chunk development fixture")
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = build_whisper_prompt(load_config())
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "session": str(args.session),
        "padding_seconds": 0.5,
        "self_test": "exact/mixed pass; missing/duplicate/substitution/reordered/unrelated fail",
        "phonon_policy": "official default CPU threads, greedy, merged dictionary hotwords",
        "phonon_hotwords": phonon_hotwords(),
        "whisper_policy": "small.en SYCL, beam5, context512, configured prompt",
        "cold_or_warmup": {},
        "chunks": [],
        "summary": {},
        "load_before": Path("/proc/loadavg").read_text().strip(),
    }
    report_path = args.output / "report.json"

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        # First request is separated from warm measurements for both engines.
        with prepared_padded_wav(chunks[0], 0.5) as padded:
            for engine in ("phonon", "whisper"):
                report["cold_or_warmup"][engine] = request(engine, padded, prompt)
        for index, audio in enumerate(chunks):
            before = digest(audio)
            with wave.open(str(audio), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            row = {
                "index": index,
                "wav": str(audio),
                "sha256": before,
                "audio_seconds": duration,
            }
            with prepared_padded_wav(audio, 0.5) as padded:
                engines = (
                    ("phonon", "whisper") if index % 2 == 0 else ("whisper", "phonon")
                )
                for engine in engines:
                    row[engine] = request(engine, padded, prompt)
            row["source_unchanged"] = digest(audio) == before
            report["chunks"].append(row)
            save()
            print(
                json.dumps(
                    {
                        "chunk": index,
                        **{
                            engine: round(row[engine]["seconds"], 3)
                            for engine in engines
                        },
                    }
                ),
                flush=True,
            )
        for engine in ("phonon", "whisper"):
            latencies = sorted(row[engine]["seconds"] for row in report["chunks"])
            text = " ".join(row[engine]["text"] for row in report["chunks"])
            report["summary"][engine] = {
                "median_seconds": statistics.median(latencies),
                "mean_seconds": statistics.mean(latencies),
                "p95_seconds": latencies[math.ceil(len(latencies) * 0.95) - 1],
                "maximum_seconds": max(latencies),
                "score": asdict(score_transcript(text, 150)),
                "text": text,
            }
        phonon, whisper = (
            report["summary"][engine] for engine in ("phonon", "whisper")
        )
        report["acceptance"] = {
            "exact_150": phonon["score"]["passed"],
            "p95_below_two_seconds": phonon["p95_seconds"] < 2,
            "median_no_slower_than_whisper": phonon["median_seconds"]
            <= whisper["median_seconds"],
            "sources_unchanged": all(
                row["source_unchanged"] for row in report["chunks"]
            ),
            "no_failed_requests": True,
        }
        report["load_after"] = Path("/proc/loadavg").read_text().strip()
        save()
        print(
            json.dumps(
                {"summary": report["summary"], "acceptance": report["acceptance"]},
                indent=2,
            )
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
