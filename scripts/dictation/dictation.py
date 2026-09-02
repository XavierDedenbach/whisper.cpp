#!/usr/bin/env python3
"""Global dictation: record mic -> whisper.cpp -> type into focused field.

Toggle mode (default): Ctrl+Space to start, Ctrl+Space again to stop.
Long utterances are sliced every MAX_RECORD_SEC: the filled slice is
transcribed and pasted while the next slice is already recording.
Uses scripts/dictation/.venv if present (created by install.sh).
"""

from __future__ import annotations

import atexit
import fcntl
import json
import math
import os
import queue
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

_venv = Path(__file__).resolve().parent / ".venv" / "lib"
if _venv.is_dir():
    for _site in _venv.glob("python*/site-packages"):
        sys.path.insert(0, str(_site))
        break

try:
    from pynput import keyboard
except ImportError:
    print(
        "pynput not found. Run: bash scripts/dictation/install.sh",
        file=sys.stderr,
    )
    sys.exit(1)

from dictation_indicator import TrayIndicator, tray_indicator_available  # noqa: E402
from vocab_prompt import (  # noqa: E402
    apply_replacements,
    build_whisper_prompt,
    load_all_vocabulary,
)

MODIFIER_KEYS = {
    "alt": {keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
    "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
    "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
    "super": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
}

# notify-send -r requires an integer replace id (a string is ignored / errors).
NOTIFY_REPLACE_ID = "4242"
SERVER_UNIT = "whisper-dictation-server.service"
_CHUNK_NO_SPACE_PREFIX = set(".,!?;:")


def prefix_chunk_for_insert(need_space: bool, chunk: str) -> tuple[str, bool]:
    """Return (text to paste, whether the next chunk needs a leading space)."""
    chunk = " ".join(chunk.split())
    if not chunk:
        return "", need_space
    if need_space and chunk[0] not in _CHUNK_NO_SPACE_PREFIX:
        chunk = " " + chunk
    return chunk, True


def build_notify_cmd(title: str, msg: str, dwell_ms: int) -> list[str]:
    return [
        "notify-send",
        "-a",
        "whisper-dictation",
        "-r",
        NOTIFY_REPLACE_ID,
        "-t",
        str(dwell_ms),
        "-h",
        "int:transient:1",
        title,
        msg,
    ]


def build_recorder_cmd(
    audio_source: str, cfg: dict[str, str] | None = None
) -> list[str] | None:
    """Build argv to record 16 kHz mono WAV; prefer parecord (reliable WAV finalize)."""
    cfg = cfg or {}
    latency_msec = cfg.get("RECORDER_LATENCY_MSEC", "20").strip() or "20"
    source = audio_source.strip()
    if subprocess.run(["which", "parecord"], capture_output=True).returncode == 0:
        cmd = [
            "parecord",
            f"--latency-msec={latency_msec}",
            "--rate=16000",
            "--channels=1",
            "--file-format=wav",
        ]
        if source:
            cmd.extend(["-d", source])
        return cmd
    if subprocess.run(["which", "pw-record"], capture_output=True).returncode == 0:
        cmd = [
            "pw-record",
            f"--latency={latency_msec}ms",
            "--rate=16000",
            "--channels=1",
        ]
        if source:
            cmd.extend(["--target", source])
        return cmd
    if subprocess.run(["which", "arecord"], capture_output=True).returncode == 0:
        cmd = ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1"]
        if source:
            cmd.extend(["-D", source])
        return cmd
    return None


def wait_for_wav_stable(path: str, timeout: float = 1.0) -> None:
    """Wait until WAV file size stops growing (recorder flushed to disk)."""
    if not path or not os.path.exists(path):
        return
    deadline = time.monotonic() + timeout
    last_size = -1
    stable_since: float | None = None
    while time.monotonic() < deadline:
        size = os.path.getsize(path)
        if size == last_size:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= 0.05:
                return
        else:
            last_size = size
            stable_since = None
        time.sleep(0.02)


def graceful_stop_recorder(
    proc: subprocess.Popen | None,
    wav_path: str | None,
    flush_msec: float,
) -> None:
    """Keep recording briefly to capture buffered tail audio, then finalize WAV."""
    if flush_msec > 0 and proc and proc.poll() is None:
        time.sleep(flush_msec / 1000.0)
    if not proc or proc.poll() is not None:
        if wav_path:
            wait_for_wav_stable(wav_path)
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
        proc.wait(timeout=2)
    if wav_path:
        wait_for_wav_stable(wav_path)


def wake_audio_source(source: str) -> None:
    """Un-suspend PipeWire/Pulse source so the mic is not idle."""
    target = source.strip() or "@DEFAULT_SOURCE@"
    for args in (
        ["pactl", "set-default-source", target],
        ["pactl", "suspend-source", target, "0"],
        ["pactl", "set-source-mute", target, "0"],
    ):
        subprocess.run(args, capture_output=True)


def load_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    paths = [
        Path(__file__).with_name("config.env"),
        Path.home() / ".config/whisper-dictation/config.env",
    ]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            val = os.path.expandvars(val)
            cfg[key.strip()] = val
    return cfg


def parse_modifiers(spec: str) -> list[set]:
    """Each inner set is one modifier group (any key in the group counts)."""
    groups: list[set] = []
    for name in spec.lower().split(","):
        name = name.strip()
        if name in MODIFIER_KEYS:
            groups.append(MODIFIER_KEYS[name])
    return groups


def resolve_trigger_key(name: str):
    name = name.strip().lower()
    special = {
        "space": keyboard.Key.space,
        "enter": keyboard.Key.enter,
        "tab": keyboard.Key.tab,
        "f8": keyboard.Key.f8,
        "f9": keyboard.Key.f9,
        "f10": keyboard.Key.f10,
    }
    if name in special:
        return special[name]
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(f"Unknown HOTKEY_KEY: {name!r}")


def resolve_whisper_home(cfg: dict[str, str]) -> Path:
    default = Path(__file__).resolve().parents[2]
    raw = cfg.get("WHISPER_HOME", "").strip()
    if not raw or "$" in raw or "BASH_SOURCE" in raw:
        return default
    home = Path(os.path.expanduser(raw))
    return home if home.is_dir() else default


def resolve_whisper_build_dir(cfg: dict[str, str], home: Path) -> Path:
    """Resolve the configured side-by-side build directory."""
    raw = cfg.get("WHISPER_BUILD_DIR", "build").strip() or "build"
    build_dir = Path(os.path.expanduser(os.path.expandvars(raw)))
    return build_dir if build_dir.is_absolute() else home / build_dir


def acquire_singleton_lock() -> object:
    """Exit if another dictation daemon already holds the lock (avoids double paste)."""
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir()))
    lock_path = runtime / "whisper-dictation.lock"
    lock_file = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            "whisper-dictation: another instance is already running; exiting",
            file=sys.stderr,
        )
        lock_file.close()
        sys.exit(0)
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    atexit.register(lock_file.close)
    return lock_file


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


# Whisper often hallucinates these on silence / accidental short clips.
_HALLUCINATION_PHRASES = (
    "thank you",
    "thanks for watching",
    "thanks for listening",
    "thank you for watching",
    "please subscribe",
    "see you next time",
    "the end",
    "bye",
    "goodbye",
    "subtitle",
    "subtitles",
)


def wav_rms(path: str) -> float:
    """Root-mean-square amplitude of 16-bit mono WAV (0 = digital silence)."""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() < 1:
            return 0.0
        frames = wf.readframes(wf.getnframes())
    if len(frames) < 2:
        return 0.0
    samples = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    if not samples:
        return 0.0
    mean_sq = sum(s * s for s in samples) / len(samples)
    return math.sqrt(mean_sq)


def is_likely_hallucination(text: str) -> bool:
    normalized = " ".join(text.strip().lower().rstrip(".!?").split())
    if not normalized:
        return True
    if normalized in {"you", "the", "i", "it", "a", "and", "or"}:
        return True
    return any(
        normalized == phrase or normalized.startswith(phrase + " ")
        for phrase in _HALLUCINATION_PHRASES
    )


def parse_transcript_output(stdout: str) -> str:
    lines = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line in ("[BLANK_AUDIO]", "[ Silence ]", "[SILENCE]"):
            continue
        if line.startswith("[") and line.endswith("]") and "-->" not in line:
            continue
        if "]" in line and "-->" in line:
            line = line.split("]", 1)[-1].strip()
        lines.append(line)
    return " ".join(lines).strip()


class Dictation:
    def __init__(self, cfg: dict[str, str]) -> None:
        self.home = resolve_whisper_home(cfg)
        self.build_dir = resolve_whisper_build_dir(cfg, self.home)
        model_name = cfg.get("WHISPER_MODEL", "small.en")
        self.model = self.home / "models" / f"ggml-{model_name}.bin"
        self.cli = self.build_dir / "bin/whisper-cli"
        self.threads = cfg.get("WHISPER_THREADS", "4")
        self.min_record = float(cfg.get("MIN_RECORD_SEC", "0.4"))
        self.min_toggle_stop = float(cfg.get("MIN_TOGGLE_STOP_SEC", "0.75"))
        self.min_audio_rms = float(cfg.get("MIN_AUDIO_RMS", "80"))
        self.recorder_latency_msec = float(cfg.get("RECORDER_LATENCY_MSEC", "20"))
        self.recorder_stop_flush_msec = float(
            cfg.get("RECORDER_STOP_FLUSH_MSEC", "120")
        )
        self.insert_method = cfg.get("INSERT_METHOD", "clipboard")
        self.mod_groups = parse_modifiers(cfg.get("HOTKEY_MODIFIERS", "ctrl"))
        self.trigger_key = resolve_trigger_key(cfg.get("HOTKEY_KEY", "space"))
        self.hotkey_mode = cfg.get("HOTKEY_MODE", "toggle").strip().lower()
        self.audio_source = cfg.get("AUDIO_SOURCE", "").strip()
        self.backend = cfg.get("WHISPER_BACKEND", "cli").strip().lower()
        self.server_url = cfg.get("WHISPER_SERVER_URL", "http://127.0.0.1:8178").rstrip(
            "/"
        )
        self.prompt = build_whisper_prompt(cfg)
        _, self.replacements = load_all_vocabulary(cfg)
        self.carry_initial_prompt = _truthy(
            cfg.get("WHISPER_CARRY_INITIAL_PROMPT", "1")
        )
        self.language = cfg.get("WHISPER_LANGUAGE", "en").strip() or "en"
        self.suppress_nst = _truthy(cfg.get("WHISPER_SUPPRESS_NST", "1"))
        self.max_record = float(cfg.get("MAX_RECORD_SEC", "45"))
        self.server_timeout = float(cfg.get("WHISPER_SERVER_TIMEOUT", "90"))
        self.cli_timeout = float(cfg.get("WHISPER_CLI_TIMEOUT", "90"))

        self._pressed: set = set()
        self._recording = False
        self._wav_path: str | None = None
        self._record_proc: subprocess.Popen | None = None
        self._record_start = 0.0
        self._session_start = 0.0
        self._lock = threading.Lock()
        self._max_timer: threading.Timer | None = None
        self._recorder = build_recorder_cmd(self.audio_source, cfg)
        self._hotkey_chord_active = False  # ignore Space key-repeat until release
        self._hotkey_label_str = self._hotkey_label(cfg)
        self._notify_ms = int(
            cfg.get("NOTIFY_MS", cfg.get("NOTIFY_DEFAULT_MS", "1000"))
        )
        self._tray_enabled = _truthy(cfg.get("TRAY_INDICATOR", "1"))
        blink_s = float(cfg.get("INDICATOR_BLINK_SEC", "1.0"))
        self._tray = TrayIndicator(blink_interval_s=blink_s)
        self._chunk_seq = 0
        self._next_submit = 0
        self._staged: dict[int, tuple[str, float, int]] = {}
        self._chunk_queue: queue.Queue[tuple[str, float, int] | None] = queue.Queue()
        self._session_id = 0
        self._paste_session = -1
        self._chunk_needs_space = False
        self._worker = threading.Thread(
            target=self._transcribe_worker, name="whisper-chunk-worker", daemon=True
        )
        self._worker.start()

    def _mods_active(self) -> bool:
        if not self.mod_groups:
            return True
        return all(any(k in self._pressed for k in group) for group in self.mod_groups)

    def _key_id(self, key) -> object:
        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char.lower()
        return key

    def _is_trigger(self, key) -> bool:
        return key == self.trigger_key or self._key_id(key) == self._key_id(
            self.trigger_key
        )

    def _on_hotkey(self) -> None:
        if self.hotkey_mode == "hold":
            if not self._recording:
                threading.Thread(target=self._start_recording, daemon=True).start()
            return
        # toggle: start or stop on each Ctrl+Space press
        if self._recording:
            elapsed = time.monotonic() - self._session_start
            if elapsed < self.min_toggle_stop:
                self._notify(
                    f"Still recording ({elapsed:.1f}s) — speak, then {self._hotkey_label_str} to stop"
                )
                return
            threading.Thread(target=self._finish_recording, daemon=True).start()
        else:
            threading.Thread(target=self._start_recording, daemon=True).start()

    def on_press(self, key) -> None:
        self._pressed.add(key)
        if not self._is_trigger(key) or not self._mods_active():
            return
        # One toggle per Ctrl+Space press (ignore Space auto-repeat while held).
        if self._hotkey_chord_active:
            return
        self._hotkey_chord_active = True
        self._on_hotkey()

    def on_release(self, key) -> None:
        self._pressed.discard(key)
        if self._is_trigger(key) or not self._mods_active():
            self._hotkey_chord_active = False
        if self.hotkey_mode != "hold":
            return
        if self._is_trigger(key) and self._recording:
            threading.Thread(target=self._finish_recording, daemon=True).start()

    def _notify(self, msg: str) -> None:
        dwell_ms = self._notify_ms
        try:
            subprocess.run(
                build_notify_cmd("Dictation", msg, dwell_ms),
                check=False,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(msg, file=sys.stderr)

    def _cancel_max_timer(self) -> None:
        timer = self._max_timer
        self._max_timer = None
        if timer is not None:
            timer.cancel()

    def _arm_max_timer(self) -> None:
        self._cancel_max_timer()
        if self.max_record <= 0:
            return
        timer = threading.Timer(self.max_record, self._on_max_record)
        timer.daemon = True
        self._max_timer = timer
        timer.start()

    def _on_max_record(self) -> None:
        with self._lock:
            if not self._recording:
                return
        self._roll_recording()

    def _spawn_recorder(self) -> tuple[subprocess.Popen, str] | None:
        """Start parecord/pw-record into a new temp WAV. Caller holds no assumptions."""
        if not self._recorder:
            return None
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="whisper-dictation-")
        os.close(fd)
        wake_audio_source(self.audio_source)
        try:
            proc = subprocess.Popen(
                [*self._recorder, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            os.unlink(path)
            return None
        return proc, path

    def _stage_chunk(self, wav: str | None, duration: float, seq: int, session_id: int) -> None:
        """Finalize-order gate: chunks enter the worker FIFO in seq order."""
        if not wav:
            return
        with self._lock:
            self._staged[seq] = (wav, duration, session_id)
            while self._next_submit in self._staged:
                item = self._staged.pop(self._next_submit)
                self._chunk_queue.put(item)
                self._next_submit += 1

    def _transcribe_worker(self) -> None:
        while True:
            item = self._chunk_queue.get()
            try:
                if item is None:
                    return
                wav, duration, session_id = item
                self._deliver_chunk(wav, duration, session_id)
            finally:
                self._chunk_queue.task_done()

    def _insert_chunk(self, text: str, session_id: int) -> str:
        if session_id != self._paste_session:
            self._paste_session = session_id
            self._chunk_needs_space = False
        pasted, self._chunk_needs_space = prefix_chunk_for_insert(
            self._chunk_needs_space, text
        )
        if pasted:
            self._insert(pasted)
        return pasted

    def _start_recording(self) -> None:
        started = False
        hold = False
        with self._lock:
            if self._recording:
                return
            if not self.cli.is_file():
                self._notify(
                    f"Missing {self.cli.name}; run scripts/dictation/install.sh"
                )
                return
            if not self.model.is_file():
                self._notify(f"Missing model {self.model.name}")
                return
            if not self._recorder:
                self._notify("No recorder (parecord/pw-record/arecord); run install.sh")
                return
            spawned = self._spawn_recorder()
            if spawned is None:
                self._notify(f"{self._recorder[0]} not found")
                return
            self._record_proc, self._wav_path = spawned
            self._recording = True
            now = time.monotonic()
            self._record_start = now
            self._session_start = now
            self._session_id += 1
            self._tray.set_recording(True)
            self._arm_max_timer()
            started = True
            hold = self.hotkey_mode != "toggle"
        if started:
            self._notify("Listening…" if hold else "Recording… (Ctrl+Space to stop)")

    def _roll_recording(self) -> None:
        """Keep listening: start the next slice, then transcribe the one that just filled."""
        with self._lock:
            if not self._recording:
                return
            old_proc = self._record_proc
            old_wav = self._wav_path
            duration = time.monotonic() - self._record_start
            seq = self._chunk_seq
            self._chunk_seq += 1
            session_id = self._session_id
            spawned = self._spawn_recorder()
            if spawned is None:
                self._recording = False
                self._record_proc = None
                self._wav_path = None
                self._cancel_max_timer()
                self._tray.set_recording(False)
            else:
                self._record_proc, self._wav_path = spawned
                self._record_start = time.monotonic()
                self._arm_max_timer()
        graceful_stop_recorder(old_proc, old_wav, self.recorder_stop_flush_msec)
        self._stage_chunk(old_wav, duration, seq, session_id)
        if spawned is None:
            self._notify("Recorder died — finishing this utterance")

    def _finish_recording(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._cancel_max_timer()
            self._tray.set_recording(False)
            proc = self._record_proc
            wav = self._wav_path
            duration = time.monotonic() - self._record_start
            seq = self._chunk_seq
            self._chunk_seq += 1
            session_id = self._session_id
            self._record_proc = None
            self._wav_path = None
        graceful_stop_recorder(proc, wav, self.recorder_stop_flush_msec)
        self._stage_chunk(wav, duration, seq, session_id)

    def _deliver_chunk(self, wav: str, duration: float, session_id: int) -> None:
        listening = self._recording and session_id == self._session_id
        wav_bytes = os.path.getsize(wav) if wav and os.path.exists(wav) else 0
        if wav_bytes < 1000:
            err = ""
            if not listening:
                hint = "check mic in Settings → Sound → Input"
                if self.audio_source:
                    hint = f"mic: {self.audio_source[:40]}… — {hint}"
                self._notify(f"Recording empty ({wav_bytes} B). {hint}. {err}")
            if wav and os.path.exists(wav):
                os.unlink(wav)
            return

        if duration < self.min_record:
            if wav and os.path.exists(wav):
                os.unlink(wav)
            if not listening:
                self._notify("Too short — speak longer, then Ctrl+Space to stop")
            return

        rms = wav_rms(wav)
        if rms < self.min_audio_rms:
            if wav and os.path.exists(wav):
                os.unlink(wav)
            if not listening:
                self._notify(
                    f"No speech detected (mic level {rms:.0f}, need ≥{self.min_audio_rms:.0f}) — "
                    "check mic / wrong input device (run: bash scripts/dictation/test-mic.sh)"
                )
            return

        text = ""
        try:
            text = apply_replacements(self._transcribe(wav), self.replacements)
        except Exception as exc:
            print(f"whisper-dictation: transcription failed: {exc}", file=sys.stderr)
            self._notify(f"Transcription failed: {str(exc)[:80]}")
            return
        finally:
            if wav and os.path.exists(wav):
                os.unlink(wav)

        if not text:
            if not listening:
                self._notify(
                    "No speech in recording — check mic level / wrong input device "
                    "(run: bash scripts/dictation/test-mic.sh)"
                )
            return

        if is_likely_hallucination(text) and rms < self.min_audio_rms * 2:
            if not listening:
                self._notify(
                    f"Ignored likely silence hallucination ({text!r}) — try again with the mic closer"
                )
            return

        pasted = self._insert_chunk(text, session_id)
        preview = pasted.strip()
        if len(preview) > 60:
            preview = preview[:57] + "…"
        listening = self._recording and session_id == self._session_id
        if listening:
            self._notify(f"Typed: {preview} (still listening…)")
        else:
            self._notify(f"Typed: {preview}")

    def _cpu_fallback_cli(self) -> Path | None:
        """CPU whisper-cli from the portable build/, if it is not the active GPU binary."""
        cpu = self.home / "build" / "bin" / "whisper-cli"
        try:
            if cpu.is_file() and cpu.resolve() != self.cli.resolve():
                return cpu
        except OSError:
            return None
        return None

    def _server_health_ok(self) -> bool:
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-fsS",
                    "--connect-timeout",
                    "1",
                    "--max-time",
                    "2",
                    f"{self.server_url}/health",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        body = (result.stdout or "").lower()
        return result.returncode == 0 and "ok" in body

    def _restart_whisper_server(self) -> bool:
        """Restart the warm server unit and wait until /health is ok."""
        print(
            "whisper-dictation: restarting hung whisper-dictation-server",
            file=sys.stderr,
        )
        try:
            result = subprocess.run(
                ["systemctl", "--user", "restart", SERVER_UNIT],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            print(f"whisper-dictation: server restart failed: {exc}", file=sys.stderr)
            return False
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            print(f"whisper-dictation: server restart failed: {err}", file=sys.stderr)
            return False
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self._server_health_ok():
                return True
            time.sleep(0.2)
        return False

    def _transcribe(self, wav_path: str) -> str:
        if self.backend == "server":
            text = self._transcribe_server(wav_path)
            if text is not None:
                return text
            print(
                "whisper-dictation: server unavailable; recovering (no GPU CLI fallback)",
                file=sys.stderr,
            )
            self._notify("Dictation server stuck — restarting…")
            if self._restart_whisper_server():
                text = self._transcribe_server(wav_path)
                if text is not None:
                    return text
            cpu = self._cpu_fallback_cli()
            if cpu is not None:
                print(
                    f"whisper-dictation: falling back to CPU whisper-cli ({cpu})",
                    file=sys.stderr,
                )
                return self._transcribe_cli(wav_path, cli=cpu)
            self._notify("Dictation server stuck — try again in a moment")
            return ""
        return self._transcribe_cli(wav_path)

    def _transcribe_cli(self, wav_path: str, cli: Path | None = None) -> str:
        binary = cli or self.cli
        cmd = [
            str(binary),
            "-m",
            str(self.model),
            "-f",
            wav_path,
            "-nt",
            "-np",
            "-t",
            self.threads,
            "-l",
            self.language,
        ]
        if self.suppress_nst:
            cmd.append("-sns")
        if self.prompt:
            cmd.extend(["--prompt", self.prompt])
        try:
            result = subprocess.run(
                cmd,
                cwd=self.home,
                capture_output=True,
                text=True,
                timeout=self.cli_timeout,
            )
        except subprocess.TimeoutExpired:
            self._notify("Local whisper-cli timed out")
            return ""
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "transcription failed").strip()
            self._notify(err[:120])
            return ""
        return parse_transcript_output(result.stdout)

    def _transcribe_server(self, wav_path: str) -> str | None:
        """POST WAV to whisper-server. Return None to signal connection/HTTP failure."""
        if subprocess.run(["which", "curl"], capture_output=True).returncode != 0:
            return None
        max_time = str(max(5, int(self.server_timeout)))
        cmd = [
            "curl",
            "-sS",
            "--connect-timeout",
            "2",
            "--max-time",
            max_time,
            f"{self.server_url}/inference",
            "-F",
            f"file=@{wav_path}",
            "-F",
            "temperature=0.0",
            "-F",
            "response_format=json",
            "-F",
            f"language={self.language}",
        ]
        if self.prompt:
            cmd.extend(["-F", f"prompt={self.prompt}"])
        if self.carry_initial_prompt:
            cmd.extend(["-F", "carry_initial_prompt=true"])
        if self.suppress_nst:
            cmd.extend(["-F", "suppress_nst=true"])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.server_timeout + 10,
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        body = (result.stdout or "").strip()
        if not body:
            return None
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return parse_transcript_output(body)
        if isinstance(payload, dict) and "error" in payload:
            return None
        text = ""
        if isinstance(payload, dict):
            text = str(payload.get("text", "")).strip()
        return " ".join(text.split())

    def _insert(self, text: str) -> None:
        use_clipboard = (
            self.insert_method == "clipboard"
            and subprocess.run(["which", "xclip"], capture_output=True).returncode == 0
        )
        if use_clipboard:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=False,
            )
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                check=False,
            )
            return
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", text],
            check=False,
        )

    def _hotkey_label(self, cfg: dict[str, str]) -> str:
        mods = cfg.get("HOTKEY_MODIFIERS", "ctrl").replace(",", "+")
        key = cfg.get("HOTKEY_KEY", "space")
        return f"{mods}+{key}"

    def _start_tray_indicator(self) -> None:
        if not self._tray_enabled:
            return
        if self._tray.start():
            return
        hint = "re-run: bash scripts/dictation/install.sh"
        if not tray_indicator_available():
            print(
                f"whisper-dictation: tray indicator unavailable (missing pystray/Pillow); {hint}",
                file=sys.stderr,
            )
        else:
            print(
                f"whisper-dictation: tray indicator failed to start; {hint}",
                file=sys.stderr,
            )

    def run(self) -> None:
        self._start_tray_indicator()
        ok = self.cli.is_file() and self.model.is_file()
        cfg = load_config()
        label = self._hotkey_label(cfg)
        if self.hotkey_mode == "toggle":
            usage = f"  Press {label} to start, press again to stop and paste"
        else:
            usage = f"  Hold {label} to record, release to paste"
        backend_line = f"  Backend: {self.backend}"
        if self.backend == "server":
            backend_line += f" ({self.server_url})"
        print(
            f"whisper-dictation ready\n"
            f"{usage}\n"
            f"  Home:    {self.home}\n"
            f"  Model:   {self.model.name} ({'ok' if self.model.is_file() else 'MISSING'})\n"
            f"{backend_line}\n"
            f"  CLI:     {self.cli.name} ({'ok' if self.cli.is_file() else 'MISSING'})\n"
            f"  Mic:     {self.audio_source or '(system default)'}",
            flush=True,
        )
        if not ok:
            msg = (
                "whisper-cli or model missing — run: bash scripts/dictation/install.sh"
            )
            print(msg, file=sys.stderr)
            self._notify(msg)
        with keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        ) as listener:
            listener.join()


def main() -> None:
    acquire_singleton_lock()
    Dictation(load_config()).run()


if __name__ == "__main__":
    main()
