#!/usr/bin/env python3
"""Generate and score a deterministic long-dictation continuity fixture.

The development fixture is 1..150.  The 1..500 holdout is gated so it cannot
be generated or run until a development report has passed, and it can only be
started once per artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from array import array
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from collections.abc import Iterator
from pathlib import Path

from vocab_prompt import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = (
    Path.home() / ".local/share/whisper-dictation/prototypes/numeric-continuity"
)
TOKEN_RE = re.compile(r"[a-z]+|\d+", re.IGNORECASE)
ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)


@dataclass(frozen=True)
class Score:
    passed: bool
    expected_count: int
    matched_prefix_count: int
    consumed_tokens: int
    total_tokens: int
    first_expected_mismatch: int | None
    unexpected_tail: list[str]
    numeric_tokens: list[int]


def _number_words(value: int, *, conjunction: bool) -> list[str]:
    if not 0 <= value <= 999:
        raise ValueError("numeric continuity oracle supports 0..999")
    if value < 20:
        return [ONES[value]]
    if value < 100:
        tens, ones = divmod(value, 10)
        return [TENS[tens]] + ([ONES[ones]] if ones else [])
    hundreds, remainder = divmod(value, 100)
    result = [ONES[hundreds], "hundred"]
    if remainder:
        if conjunction:
            result.append("and")
        result.extend(_number_words(remainder, conjunction=conjunction))
    return result


def _number_candidates(value: int) -> tuple[tuple[str, ...], ...]:
    candidates = {
        (str(value),),
        tuple(_number_words(value, conjunction=False)),
        tuple(_number_words(value, conjunction=True)),
    }
    return tuple(sorted(candidates, key=lambda item: (len(item), item)))


def score_transcript(text: str, expected_count: int) -> Score:
    """Require an exact 1..N sequence, accepting digits or English number words."""
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    positions = {0}
    matched = 0
    best_position = 0
    for expected in range(1, expected_count + 1):
        next_positions: set[int] = set()
        for position in positions:
            for candidate in _number_candidates(expected):
                end = position + len(candidate)
                if tuple(tokens[position:end]) == candidate:
                    next_positions.add(end)
        if not next_positions:
            break
        positions = next_positions
        matched = expected
        best_position = max(positions)

    passed = matched == expected_count and len(tokens) in positions
    if passed:
        best_position = len(tokens)
    numeric_tokens = [int(token) for token in tokens if token.isdigit()]
    mismatch = None if passed else matched + 1
    return Score(
        passed=passed,
        expected_count=expected_count,
        matched_prefix_count=matched,
        consumed_tokens=best_position,
        total_tokens=len(tokens),
        first_expected_mismatch=mismatch,
        unexpected_tail=tokens[best_position : best_position + 16],
        numeric_tokens=numeric_tokens,
    )


def self_test() -> None:
    exact_words = "one, two, three, four, five, six, seven, eight, nine, ten"
    exact_mixed = "1 2 three 4 five 6 seven 8 nine 10"
    assert score_transcript(exact_words, 10).passed
    assert score_transcript(exact_mixed, 10).passed
    cases = {
        "missing": "1 2 3 5 6 7 8 9 10",
        "duplicate": "1 2 3 4 4 5 6 7 8 9 10",
        "substitution": "1 2 3 4 15 6 7 8 9 10",
        "reordered": "1 2 3 5 4 6 7 8 9 10",
        "unrelated": "1 2 3 banana 4 5 6 7 8 9 10",
    }
    for name, text in cases.items():
        assert not score_transcript(text, 10).passed, name


def _trim_s16(frames: bytes, threshold: int = 120, pad_samples: int = 220) -> bytes:
    samples = array("h")
    samples.frombytes(frames)
    first = 0
    while first < len(samples) and abs(samples[first]) < threshold:
        first += 1
    last = len(samples)
    while last > first and abs(samples[last - 1]) < threshold:
        last -= 1
    first = max(0, first - pad_samples)
    last = min(len(samples), last + pad_samples)
    return samples[first:last].tobytes()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / source.getframerate()


def wav_pcm(path: Path) -> bytes:
    """Return only PCM payload bytes so independently wrapped WAVs can be compared."""
    with wave.open(str(path), "rb") as source:
        return source.readframes(source.getnframes())


def wav_pcm_rms(path: Path) -> float:
    """Return RMS for signed-16 mono PCM, matching the daemon's silence gate."""
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError("RMS probe requires mono signed-16 PCM")
        samples = array("h")
        samples.frombytes(source.readframes(source.getnframes()))
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def generate_audio(
    count: int,
    output: Path,
    *,
    seed: int,
    cadence: float,
    jitter: float,
    voice: str,
    rate: int,
) -> dict[str, object]:
    espeak = shutil.which("espeak")
    ffmpeg = shutil.which("ffmpeg")
    if not espeak or not ffmpeg:
        raise RuntimeError("espeak and ffmpeg are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="whisper-numeric-tts-") as tmp:
        temp = Path(tmp)
        raw_output = temp / "combined.wav"
        chunks: list[bytes] = []
        sample_rate = 0
        sample_width = 0
        channels = 0
        intervals: list[float] = []
        for value in range(1, count + 1):
            phrase_path = temp / f"{value:04d}.wav"
            subprocess.run(
                [
                    espeak,
                    "-v",
                    voice,
                    "-s",
                    str(rate),
                    "-z",
                    "-w",
                    str(phrase_path),
                    str(value),
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
            with wave.open(str(phrase_path), "rb") as source:
                params = (
                    source.getframerate(),
                    source.getsampwidth(),
                    source.getnchannels(),
                )
                if sample_rate == 0:
                    sample_rate, sample_width, channels = params
                if params != (sample_rate, sample_width, channels):
                    raise RuntimeError("espeak emitted inconsistent WAV formats")
                if sample_width != 2 or channels != 1:
                    raise RuntimeError("expected mono signed-16 espeak output")
                phrase = _trim_s16(source.readframes(source.getnframes()))
            phrase_seconds = len(phrase) / (sample_rate * sample_width)
            interval = max(
                phrase_seconds + 0.08,
                cadence + rng.uniform(-jitter, jitter),
            )
            intervals.append(interval)
            silence_samples = max(0, round((interval - phrase_seconds) * sample_rate))
            chunks.extend((phrase, b"\x00\x00" * silence_samples))

        leading_silence = b"\x00\x00" * round(sample_rate * 0.25)
        trailing_silence = b"\x00\x00" * round(sample_rate * 1.0)
        with wave.open(str(raw_output), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(sample_rate)
            target.writeframes(leading_silence + b"".join(chunks) + trailing_silence)
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(raw_output),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            check=True,
            timeout=120,
        )
    with wave.open(str(output), "rb") as rendered:
        duration = rendered.getnframes() / rendered.getframerate()
    return {
        "count": count,
        "voice": voice,
        "rate": rate,
        "seed": seed,
        "requested_cadence_seconds": cadence,
        "jitter_seconds": jitter,
        "minimum_interval_seconds": min(intervals),
        "maximum_interval_seconds": max(intervals),
        "mean_interval_seconds": sum(intervals) / len(intervals),
        "duration_seconds": duration,
        "sample_rate": 16000,
        "channels": 1,
    }


def _capture_command(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    output = (result.stdout or result.stderr or "").strip()
    return output if result.returncode == 0 else f"exit {result.returncode}: {output}"


def collect_provenance(cfg: dict[str, str]) -> dict[str, object]:
    repo = SCRIPT_DIR.parents[1]
    raw_home = cfg.get("WHISPER_HOME", "").strip()
    whisper_home = (
        Path(os.path.expanduser(os.path.expandvars(raw_home))) if raw_home else repo
    )
    raw_build = cfg.get("WHISPER_BUILD_DIR", "build").strip() or "build"
    build_dir = Path(os.path.expanduser(os.path.expandvars(raw_build)))
    if not build_dir.is_absolute():
        build_dir = whisper_home / build_dir
    model_name = cfg.get("WHISPER_MODEL", "small.en").strip() or "small.en"
    model_path = whisper_home / "models" / f"ggml-{model_name}.bin"
    server_binary = build_dir / "bin/whisper-server"
    cache = build_dir / "CMakeCache.txt"
    script_path = Path(__file__).resolve()
    repository_diff = _capture_command(["git", "diff", "--binary"], cwd=repo)
    return {
        "exact_command": [str(script_path), *sys.argv[1:]],
        "repository_head": _capture_command(["git", "rev-parse", "HEAD"], cwd=repo),
        "repository_describe": _capture_command(
            ["git", "describe", "--always", "--dirty"], cwd=repo
        ),
        "repository_status_porcelain_v2": _capture_command(
            ["git", "status", "--porcelain=v2", "--untracked-files=all"],
            cwd=repo,
        ),
        "tracked_diff_sha256": hashlib.sha256(repository_diff.encode()).hexdigest(),
        "harness": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
        "model": {
            "name": model_name,
            "path": str(model_path),
            "sha256": sha256_file(model_path) if model_path.is_file() else "missing",
        },
        "server": {
            "url": "configured per request",
            "systemd": _capture_command(
                [
                    "systemctl",
                    "--user",
                    "show",
                    "whisper-dictation-server.service",
                    "-p",
                    "ActiveState",
                    "-p",
                    "MainPID",
                    "-p",
                    "StatusText",
                ]
            ),
            "build_dir": str(build_dir),
            "binary_sha256": (
                sha256_file(server_binary) if server_binary.is_file() else "missing"
            ),
            "cmake_cache_sha256": sha256_file(cache) if cache.is_file() else "missing",
            "accelerator": cfg.get("WHISPER_ACCELERATOR", "auto"),
            "oneapi_device_selector": cfg.get("WHISPER_ONEAPI_DEVICE_SELECTOR", ""),
            "sycl_device": cfg.get("WHISPER_SYCL_DEVICE", ""),
            "expected_device": cfg.get("WHISPER_SYCL_EXPECTED_DEVICE", ""),
        },
        "tools": {
            "espeak": _capture_command(["espeak", "--version"]).splitlines()[0],
            "ffmpeg": _capture_command(["ffmpeg", "-version"]).splitlines()[0],
            "python": sys.version,
        },
    }


@contextmanager
def pulse_test_sink(run_id: str) -> Iterator[tuple[str, str]]:
    """Create an inaudible Pulse sink and restore the user's source afterward."""
    sink = "whisper_continuity_" + re.sub(r"[^a-zA-Z0-9_]", "", run_id)[-24:]
    original_source_result = subprocess.run(
        ["pactl", "get-default-source"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    original_source = original_source_result.stdout.strip()
    loaded = subprocess.run(
        [
            "pactl",
            "load-module",
            "module-null-sink",
            f"sink_name={sink}",
            "rate=16000",
            "channels=1",
            "format=s16le",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if loaded.returncode != 0:
        raise RuntimeError((loaded.stderr or loaded.stdout).strip())
    module_id = loaded.stdout.strip()
    try:
        source = f"{sink}.monitor"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            listed = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if source in listed.stdout:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f"Pulse monitor source did not appear: {source}")
        yield sink, source
    finally:
        if original_source:
            subprocess.run(
                ["pactl", "set-default-source", original_source],
                capture_output=True,
                timeout=5,
            )
        subprocess.run(
            ["pactl", "unload-module", module_id],
            capture_output=True,
            timeout=10,
        )


def _stop_probe_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    if process.stderr is not None and not process.stderr.closed:
        process.stderr.close()


def _stop_probe_app(app: object) -> None:
    if getattr(app, "_record_proc", None) is not None:
        app._finish_recording()
    worker = getattr(app, "_worker", None)
    if worker is not None and worker.is_alive():
        app._chunk_queue.put(None)
        worker.join(timeout=5)


def run_implemented_capture(
    args: argparse.Namespace,
    *,
    run_id: str,
    run_dir: Path,
    audio_path: Path,
    audio_metadata: dict[str, object],
    count: int,
) -> int:
    """Exercise actual Dictation capture/storage and derive its attestation."""
    from dictation import Dictation, wav_rms

    cfg = load_config()
    cfg.update(
        {
            "WHISPER_SESSION_DIR": str(run_dir / "sessions"),
            "CONTINUOUS_CAPTURE": "1",
            "STREAM_SEGMENT_TARGET_SEC": str(args.target_seconds),
            "STREAM_SEGMENT_MIN_SEC": str(args.minimum_seconds),
            "STREAM_SEGMENT_MAX_SEC": str(args.maximum_seconds),
            "TRANSCRIPTION_TRAILING_SILENCE_SEC": str(args.trailing_silence_seconds),
            "WHISPER_BEAM_SIZE": str(args.beam_size or 5),
            "WHISPER_AUDIO_CTX": str(args.audio_ctx),
            "MIN_AUDIO_RMS": str(args.minimum_audio_rms),
            "MAX_RECORD_SEC": "0",
            "TRAY_INDICATOR": "0",
        }
    )
    inserted: list[str] = []
    notifications: list[str] = []
    max_queue_depth = 0
    playback_error = ""
    app: Dictation | None = None
    session: Path | None = None
    recorder_pid: int | None = None
    started = time.monotonic()
    with pulse_test_sink(run_id) as (sink, source), ExitStack() as cleanup:
        cfg["AUDIO_SOURCE"] = source
        app = Dictation(cfg)
        cleanup.callback(_stop_probe_app, app)
        app._notify = notifications.append

        def capture_insert(text: str) -> bool:
            inserted.append(text)
            return True

        app._insert = capture_insert
        app._start_recording()
        if not app._recording or app._record_proc is None:
            raise RuntimeError(
                notifications[-1]
                if notifications
                else "continuous recorder did not start"
            )
        recorder_pid = app._record_proc.pid
        session = app._active_session
        playback = subprocess.Popen(
            ["paplay", f"--device={sink}", str(audio_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        cleanup.callback(_stop_probe_process, playback)
        playback_deadline = (
            time.monotonic() + float(audio_metadata["duration_seconds"]) + 30
        )
        while playback.poll() is None and time.monotonic() < playback_deadline:
            max_queue_depth = max(max_queue_depth, app._chunk_queue.qsize())
            time.sleep(0.1)
        if playback.poll() is None:
            playback.terminate()
            try:
                playback.wait(timeout=2)
            except subprocess.TimeoutExpired:
                playback.kill()
                playback.wait(timeout=2)
            raise RuntimeError("test playback exceeded its source duration budget")
        if playback.stderr is not None:
            playback_error = (
                playback.stderr.read().decode("utf-8", errors="replace").strip()
            )
            playback.stderr.close()
        if playback.returncode != 0:
            raise RuntimeError(playback_error or f"paplay exit {playback.returncode}")
        time.sleep(0.25)
        app._finish_recording()
        if not app._stream_done.wait(10):
            raise RuntimeError("continuous recorder did not finish draining")
        queue_deadline = time.monotonic() + max(120.0, args.timeout * 2)
        while app._chunk_queue.unfinished_tasks and time.monotonic() < queue_deadline:
            max_queue_depth = max(max_queue_depth, app._chunk_queue.qsize())
            time.sleep(0.05)
        if app._chunk_queue.unfinished_tasks:
            raise RuntimeError("transcription queue did not drain")
        app._chunk_queue.put(None)
        app._worker.join(timeout=5)

    if app is None or session is None:
        raise RuntimeError("implemented capture did not create a session")
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_chunks = sorted(manifest["chunks"], key=lambda item: int(item["index"]))
    wav_paths = [session / str(item["wav"]) for item in manifest_chunks]
    captured_pcm = b"".join(wav_pcm(path) for path in wav_paths if path.is_file())
    captured_pcm_sha256 = hashlib.sha256(captured_pcm).hexdigest()
    completed_text = app._store.completed_text(session)
    delivered_text = inserted[0] if len(inserted) == 1 else ""
    score = score_transcript(delivered_text, count)
    bad_statuses = [
        {"index": item["index"], "status": item["status"]}
        for item in manifest_chunks
        if item["status"] not in {"complete", "silent"}
    ]
    policy = manifest.get("policy", {})
    attestation_checks = {
        "one_recorder_process": app._stream_recorder_pids == [recorder_pid],
        "capture_mode_persisted": policy.get("capture_mode") == "continuous",
        "beam_size_persisted": int(policy.get("beam_size", 0)) == (args.beam_size or 5),
        "audio_ctx_persisted": int(policy.get("audio_ctx", 0)) == args.audio_ctx,
        "padding_persisted": float(policy.get("trailing_silence_seconds", -1))
        == args.trailing_silence_seconds,
        "silence_threshold_persisted": float(policy.get("minimum_audio_rms", -1))
        == args.minimum_audio_rms,
        "session_closed": manifest.get("recording") is False,
        "all_manifest_wavs_exist": all(path.is_file() for path in wav_paths),
        "all_recorder_bytes_partitioned": (
            len(captured_pcm) == app._stream_input_bytes == app._stream_output_bytes
            and captured_pcm_sha256 == app._stream_input_sha256
        ),
        "no_failed_chunks": not bad_statuses,
        "single_atomic_paste": len(inserted) == 1,
        "paste_matches_store": delivered_text == completed_text,
    }
    attestation_passed = all(attestation_checks.values())
    chunk_evidence = []
    for item, wav_path in zip(manifest_chunks, wav_paths):
        fragment = session / f"chunk-{int(item['index']):04d}.txt"
        chunk_evidence.append(
            {
                "index": int(item["index"]),
                "status": item["status"],
                "duration_seconds": float(item["duration"]),
                "wav": str(wav_path),
                "wav_sha256": sha256_file(wav_path) if wav_path.is_file() else None,
                "pcm_bytes": len(wav_pcm(wav_path)) if wav_path.is_file() else 0,
                "rms": wav_rms(str(wav_path)) if wav_path.is_file() else None,
                "text": (
                    fragment.read_text(encoding="utf-8").strip()
                    if fragment.is_file()
                    else ""
                ),
            }
        )
    timing_values = app._inference_timings
    report = {
        "run_id": run_id,
        "kind": "final-holdout" if args.final else "development",
        "requested_implementation_path": True,
        "implementation_path": attestation_passed,
        "production_equivalent_shape": True,
        "audio": audio_metadata,
        "provenance": collect_provenance(cfg),
        "production_attestation": {
            "passed": attestation_passed,
            "checks": attestation_checks,
            "dictation_module_sha256": sha256_file(SCRIPT_DIR / "dictation.py"),
            "session_store_module_sha256": sha256_file(SCRIPT_DIR / "session_store.py"),
            "temporal_audio_module_sha256": sha256_file(
                SCRIPT_DIR / "temporal_audio.py"
            ),
            "recorder_pid": recorder_pid,
            "recorder_pids": app._stream_recorder_pids,
            "captured_input_bytes": app._stream_input_bytes,
            "durable_output_bytes": app._stream_output_bytes,
            "captured_pcm_sha256": app._stream_input_sha256,
            "concatenated_chunk_pcm_sha256": captured_pcm_sha256,
            "manifest": str(manifest_path),
            "paste_count": len(inserted),
            "bad_statuses": bad_statuses,
        },
        "request": {
            "server_url": app.server_url,
            "target_seconds": args.target_seconds,
            "minimum_seconds": args.minimum_seconds,
            "maximum_seconds": args.maximum_seconds,
            "trailing_silence_seconds": args.trailing_silence_seconds,
            "minimum_audio_rms": args.minimum_audio_rms,
            "beam_size": args.beam_size or 5,
            "audio_ctx": args.audio_ctx,
            "text_overlap_or_deduplication": False,
        },
        "chunks": chunk_evidence,
        "score": asdict(score),
        "transcript": delivered_text,
        "notifications": notifications,
        "playback_error": playback_error,
        "maximum_queue_depth": max_queue_depth,
        "elapsed_seconds": time.monotonic() - started,
        "inference_timing_seconds": {
            "count": len(timing_values),
            "minimum": min(timing_values, default=0.0),
            "median": statistics.median(timing_values) if timing_values else 0.0,
            "mean": statistics.mean(timing_values) if timing_values else 0.0,
            "maximum": max(timing_values, default=0.0),
            "total": sum(timing_values),
        },
    }
    (run_dir / "transcript.txt").write_text(delivered_text + "\n", encoding="utf-8")
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(asdict(score), indent=2))
    print(json.dumps(report["production_attestation"], indent=2))
    print(f"artifacts: {run_dir}")
    return 0 if attestation_passed else 1


def run_probe(args: argparse.Namespace) -> int:
    self_test()
    if not args.implemented:
        raise RuntimeError(
            "--implemented is required; superseded prototype paths were removed"
        )

    root = args.artifact_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.final:
        development_reports = sorted(root.glob("dev-implemented-150-*/report.json"))
        passed_development = any(
            (
                (report := json.loads(path.read_text(encoding="utf-8"))).get(
                    "implementation_path"
                )
                is True
                and report.get("production_attestation", {}).get("passed") is True
                and report["score"]["passed"]
            )
            for path in development_reports
        )
        if not passed_development:
            raise RuntimeError("the implemented 1..150 fixture has not passed")
        marker = root / "FINAL_HOLDOUT_STARTED.json"
        if marker.exists():
            raise RuntimeError(f"the final holdout was already started: {marker}")
        marker.write_text(
            json.dumps(
                {"started_at": datetime.now(timezone.utc).isoformat(), "count": 500},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        count = 500
        prefix = "final-500"
    else:
        count = 150
        prefix = "dev-implemented-150"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = root / f"{prefix}-{run_id}"
    run_dir.mkdir(parents=True)
    audio_path = run_dir / "counting.wav"
    audio_metadata = generate_audio(
        count,
        audio_path,
        seed=args.seed,
        cadence=args.cadence,
        jitter=args.jitter,
        voice=args.voice,
        rate=args.rate,
    )
    audio_metadata["sha256"] = sha256_file(audio_path)
    return run_implemented_capture(
        args,
        run_id=run_id,
        run_dir=run_dir,
        audio_path=audio_path,
        audio_metadata=audio_metadata,
        count=count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final", action="store_true", help="run sealed 1..500 holdout"
    )
    parser.add_argument(
        "--implemented",
        action="store_true",
        help="required guard confirming the wired production path is intended",
    )
    parser.add_argument("--target-seconds", type=float, default=8.0)
    parser.add_argument("--minimum-seconds", type=float, default=7.0)
    parser.add_argument("--maximum-seconds", type=float, default=9.0)
    parser.add_argument("--trailing-silence-seconds", type=float, default=0.5)
    parser.add_argument("--audio-ctx", type=int, default=512)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument(
        "--minimum-audio-rms",
        type=float,
        default=80.0,
        help="skip inference for quieter source-owned chunks",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=150_8178)
    parser.add_argument("--cadence", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=0.08)
    parser.add_argument("--voice", default="en-us")
    parser.add_argument("--rate", type=int, default=260)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_probe(parse_args()))
