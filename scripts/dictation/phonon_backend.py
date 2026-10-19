"""Phonon-2 HTTP adapter shared by the daemon, startup warmup, and health check."""

from __future__ import annotations

import argparse
import json
import subprocess
from urllib.parse import urlsplit

from vocab_prompt import build_phonon_hotwords, load_config


def local_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or not parsed.port
    ):
        raise ValueError("Phonon requires an http://127.0.0.1:PORT loopback server URL")
    return f"http://127.0.0.1:{parsed.port}"


def health_matches(payload: object, device: str) -> bool:
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return False
    model = str(payload.get("model", "")).lower()
    decode = payload.get("decode", {})
    actual_device = decode.get("engine") if isinstance(decode, dict) else None
    actual_device = actual_device or payload.get("engine") or payload.get("backend")
    if (
        not actual_device
        and payload.get("gpu")
        and "CUDA" in str(payload.get("path", ""))
    ):
        actual_device = "cuda"
    return model in {"phonon-2", "fermionresearch/phonon-2"} and actual_device == device


def health(url: str, device: str) -> bool:
    url = local_url(url)
    try:
        result = subprocess.run(
            [
                "curl",
                "-fsS",
                "--noproxy",
                "*",
                "--connect-timeout",
                "1",
                "--max-time",
                "2",
                url + "/health",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.returncode == 0 and health_matches(
            json.loads(result.stdout), device
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False


def transcribe(url: str, wav: str, timeout: float, hotwords: list[str]) -> str | None:
    url = local_url(url)
    if len(hotwords) > 25:
        raise ValueError("Phonon-2 supports at most 25 active hotwords")
    cmd = [
        "curl",
        "--fail-with-body",
        "-sS",
        "--noproxy",
        "*",
        "--connect-timeout",
        "2",
        "--max-time",
        str(max(5, int(timeout))),
        url + "/v1/audio/transcriptions",
        "-F",
        f"file=@{wav}",
        "-F",
        "model=phonon-2",
        "-F",
        "response_format=verbose_json",
        "-F",
        "hotwords=" + json.dumps(hotwords),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 10
        )
        if result.returncode:
            return None
        payload = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if not isinstance(payload, dict) or "error" in payload:
        return None
    meta = payload.get("x_fermion", {})
    if isinstance(meta, dict) and meta.get("truncated"):
        return None
    text = payload.get("text")
    return " ".join(text.split()) if isinstance(text, str) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("url", "health", "warmup"))
    parser.add_argument("url")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--audio")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.mode == "url":
        print(local_url(args.url))
        return
    if not health(args.url, args.device):
        raise SystemExit("Phonon health/model/device mismatch")
    if args.mode == "warmup":
        text = transcribe(
            args.url, args.audio, args.timeout, build_phonon_hotwords(load_config())
        )
        if not text or not any(character.isalnum() for character in text):
            raise SystemExit("Phonon warmup returned no usable text")
        print(text)


if __name__ == "__main__":
    main()
