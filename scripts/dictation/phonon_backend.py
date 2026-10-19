"""Phonon-2 HTTP adapter shared by the daemon, startup warmup, and health check."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import stat
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from vocab_prompt import build_phonon_hotwords, load_config


def read_key_file(path: str | Path) -> str:
    path = Path(os.path.expandvars(str(path))).expanduser()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r", encoding="ascii") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o077
            or info.st_uid != os.geteuid()
        ):
            raise ValueError("Phonon API key must be a private file owned by this user")
        if info.st_size > 257:
            raise ValueError("Invalid Phonon API key file")
        key = stream.read(258).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", key):
        raise ValueError("Invalid Phonon API key file")
    return key


def ensure_key_file(path: str | Path) -> None:
    path = Path(os.path.expandvars(str(path))).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        read_key_file(path)
        return
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write(secrets.token_urlsafe(32) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _auth_input(key_file: str | Path | None) -> str | None:
    if key_file is None:
        return None
    # curl reads this through stdin; the credential never appears in argv.
    return 'header = "Authorization: Bearer ' + read_key_file(key_file) + '"\n'


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


def health(url: str, device: str, *, key_file: str | Path | None = None) -> bool:
    url = local_url(url)
    try:
        auth_input = _auth_input(key_file)
        result = subprocess.run(
            [
                "curl",
                "--disable",
                "-fsS",
                "--noproxy",
                "*",
                "--connect-timeout",
                "1",
                "--max-time",
                "2",
                url + "/health",
                *(["--config", "-"] if key_file is not None else []),
            ],
            capture_output=True,
            text=True,
            timeout=3,
            input=auth_input,
        )
        return result.returncode == 0 and health_matches(
            json.loads(result.stdout), device
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False


def transcribe(
    url: str,
    wav: str,
    timeout: float,
    hotwords: list[str],
    *,
    key_file: str | Path | None = None,
) -> str | None:
    url = local_url(url)
    if len(hotwords) > 25:
        raise ValueError("Phonon-2 supports at most 25 active hotwords")
    cmd = [
        "curl",
        "--disable",
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
        auth_input = _auth_input(key_file)
        if key_file is not None:
            cmd.extend(["--config", "-"])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10,
            input=auth_input,
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
    parser.add_argument("mode", choices=("key", "url", "health", "warmup"))
    parser.add_argument("url")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--audio")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.mode == "key":
        ensure_key_file(args.url)
        # Private launcher capture only: never log or invoke this as a diagnostic.
        print(read_key_file(args.url))
        return
    if args.mode == "url":
        print(local_url(args.url))
        return
    cfg = load_config()
    key_file = (
        cfg.get(
            "PHONON_API_KEY_FILE",
            str(Path.home() / ".config/whisper-dictation/phonon-api-key"),
        )
        if args.device == "cuda"
        else None
    )
    if not health(args.url, args.device, key_file=key_file):
        raise SystemExit("Phonon health/model/device mismatch")
    if args.mode == "warmup":
        text = transcribe(
            args.url,
            args.audio,
            args.timeout,
            build_phonon_hotwords(cfg),
            key_file=key_file,
        )
        if not text or not any(character.isalnum() for character in text):
            raise SystemExit("Phonon warmup returned no usable text")
        print(text)


if __name__ == "__main__":
    main()
