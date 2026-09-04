#!/usr/bin/env python3
"""Global dictation: record mic -> whisper.cpp -> type into focused field.

Toggle mode (default): Ctrl+Space to start, Ctrl+Space again to stop.
Long utterances are sliced every MAX_RECORD_SEC: the filled slice is
transcribed while the next slice is already recording. The complete ordered
session is pasted once when recording stops.
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
import unicodedata
import wave
from dataclasses import dataclass
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
from session_store import ChunkJob, SessionStore  # noqa: E402
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
MIN_USABLE_WAV_BYTES = 1000
RECORDER_START_ATTEMPTS = 3
RECORDER_START_TIMEOUT_SEC = 1.0
RECORDER_RETRY_DELAY_SEC = 0.05
RECORDER_WATCH_POLL_SEC = 0.05
SHUTDOWN_BUDGET_SEC = 13.0
SHUTDOWN_RESOURCE_RESERVE_SEC = 1.0
CLIPBOARD_READY_TIMEOUT_SEC = 0.5
CLIPBOARD_TERM_TIMEOUT_SEC = 0.25
CLIPBOARD_KILL_TIMEOUT_SEC = 0.25
XDOTOOL_TIMEOUT_SEC = 0.25
TRAY_STOP_TIMEOUT_SEC = 0.5
MAX_RECORDER_FLUSH_MSEC = 500.0


@dataclass(frozen=True)
class SessionPasteJob:
    """Paste all completed fragments after the final live chunk is processed."""

    session_path: Path
    session_id: int


def remaining_timeout(
    deadline: float | None,
    maximum: float,
    *,
    reserve: float = 0.0,
) -> float:
    """Return a non-negative wait bounded by an optional absolute deadline."""
    if deadline is None:
        return maximum
    return max(0.0, min(maximum, deadline - time.monotonic() - reserve))


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


def wait_for_wav_stable(
    path: str,
    timeout: float = 1.0,
    *,
    deadline: float | None = None,
) -> None:
    """Wait until WAV file size stops growing (recorder flushed to disk)."""
    if not path or not os.path.exists(path):
        return
    stable_deadline = time.monotonic() + remaining_timeout(deadline, timeout)
    last_size = -1
    stable_since: float | None = None
    while time.monotonic() < stable_deadline:
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
    wait_timeout: float = 5.0,
    *,
    deadline: float | None = None,
) -> bool:
    """Finalize the WAV and report whether the recorder child was reaped."""
    flush_msec = min(max(0.0, flush_msec), MAX_RECORDER_FLUSH_MSEC)
    if flush_msec > 0 and proc and proc.poll() is None:
        time.sleep(remaining_timeout(deadline, flush_msec / 1000.0, reserve=0.25))
    if proc is None or proc.poll() is not None:
        if wav_path:
            wait_for_wav_stable(wav_path, deadline=deadline)
        return True
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=remaining_timeout(deadline, wait_timeout, reserve=0.25))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=remaining_timeout(deadline, 0.25))
        except subprocess.TimeoutExpired:
            print(
                f"whisper-dictation: recorder {proc.pid} did not reap before shutdown deadline",
                file=sys.stderr,
            )
            return False
    if wav_path:
        wait_for_wav_stable(wav_path, deadline=deadline)
    return True


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
    "end of video",
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


def is_punctuation_only(text: str) -> bool:
    """Return whether Whisper produced punctuation without any spoken text."""
    compact = "".join(text.split())
    return bool(compact) and all(
        unicodedata.category(char).startswith("P") for char in compact
    )


def is_likely_hallucination(text: str) -> bool:
    if is_punctuation_only(text):
        return True
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
        configured_insert_method = cfg.get("INSERT_METHOD", "clipboard").strip().lower()
        if configured_insert_method not in {"", "clipboard"}:
            print(
                "whisper-dictation: direct typing is no longer used; "
                "INSERT_METHOD now uses atomic clipboard paste",
                file=sys.stderr,
            )
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
        session_root = cfg.get(
            "WHISPER_SESSION_DIR",
            "${HOME}/.local/share/whisper-dictation/sessions",
        )
        self._store = SessionStore(
            Path(os.path.expanduser(os.path.expandvars(session_root)))
        )

        self._pressed: set = set()
        self._recording = False
        self._wav_path: str | None = None
        self._record_proc: subprocess.Popen | None = None
        self._record_start = 0.0
        self._session_start = 0.0
        self._record_generation = 0
        self._last_recorder_error = ""
        self._rejected_recorder_lock = threading.Lock()
        self._unreaped_rejected_recorders = 0
        self._lock = threading.Lock()
        self._finalize_lock = threading.Lock()
        self._shutdown_deadline: float | None = None
        self._clipboard_lock = threading.Lock()
        self._clipboard_proc: subprocess.Popen | None = None
        self._clipboard_closed = threading.Event()
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
        self._staged: dict[int, ChunkJob | SessionPasteJob | None] = {}
        self._chunk_queue: queue.Queue[ChunkJob | SessionPasteJob | None] = (
            queue.Queue()
        )
        self._session_id = 0
        self._session_paths: dict[int, Path] = {}
        self._active_session: Path | None = None
        for job in self._store.recoverable(limit=32):
            self._chunk_queue.put(job)
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
        timer = threading.Timer(
            self.max_record,
            self._on_max_record,
            args=(self._record_generation,),
        )
        timer.daemon = True
        self._max_timer = timer
        timer.start()

    def _on_max_record(self, generation: int) -> None:
        with self._lock:
            if not self._recording or generation != self._record_generation:
                return
        self._roll_recording(expected_generation=generation)

    @staticmethod
    def _read_recorder_stderr(proc: subprocess.Popen) -> str:
        stream = getattr(proc, "stderr", None)
        if stream is None:
            return ""
        try:
            output = stream.read()
        except (OSError, ValueError):
            return ""
        finally:
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace").strip()
        return str(output or "").strip()

    def _recorder_exit_detail(self, proc: subprocess.Popen) -> str:
        code = proc.poll()
        detail = f"exit {code}" if code is not None else "no audio payload"
        stderr = self._read_recorder_stderr(proc)
        if stderr:
            detail += f": {' '.join(stderr.split())[:200]}"
        return detail

    def _wait_for_recorder_ready(
        self, proc: subprocess.Popen, path: str
    ) -> tuple[bool, str, bool]:
        """Return readiness, detail, and whether a rejected child is reaped."""
        deadline = time.monotonic() + RECORDER_START_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False, self._recorder_exit_detail(proc), True
            try:
                if os.path.getsize(path) >= MIN_USABLE_WAV_BYTES:
                    return True, "", False
            except FileNotFoundError:
                pass
            time.sleep(0.01)

        if proc.poll() is None:
            reaped = graceful_stop_recorder(proc, path, 0, wait_timeout=0.5)
            if not reaped:
                return (
                    False,
                    f"no audio payload within {RECORDER_START_TIMEOUT_SEC:.2f}s "
                    "(recorder did not reap)",
                    False,
                )
        detail = self._recorder_exit_detail(proc)
        return (
            False,
            f"no audio payload within {RECORDER_START_TIMEOUT_SEC:.2f}s ({detail})",
            True,
        )

    @staticmethod
    def _reap_rejected_recorder(proc: subprocess.Popen, path: str) -> None:
        """Retain an unreaped startup child and its WAV until wait completes."""
        try:
            proc.wait()
        except (ChildProcessError, OSError):
            return
        stream = getattr(proc, "stderr", None)
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def _retain_rejected_recorder(self, proc: subprocess.Popen, path: str) -> None:
        with self._rejected_recorder_lock:
            self._unreaped_rejected_recorders += 1

        def reap() -> None:
            try:
                self._reap_rejected_recorder(proc, path)
            finally:
                with self._rejected_recorder_lock:
                    self._unreaped_rejected_recorders -= 1

        threading.Thread(
            target=reap,
            name=f"whisper-recorder-reaper-{proc.pid}",
            daemon=True,
        ).start()

    def _rejected_recorder_pending(self) -> bool:
        with self._rejected_recorder_lock:
            return self._unreaped_rejected_recorders > 0

    def _spawn_recorder(self) -> tuple[subprocess.Popen, str] | None:
        """Start and validate a recorder, retrying bounded startup failures."""
        if not self._recorder:
            self._last_recorder_error = "no recorder command is available"
            return None
        wake_audio_source(self.audio_source)
        self._last_recorder_error = ""
        for attempt in range(1, RECORDER_START_ATTEMPTS + 1):
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="whisper-dictation-")
            os.close(fd)
            try:
                proc = subprocess.Popen(
                    [*self._recorder, path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
            except OSError as exc:
                self._last_recorder_error = str(exc)
                os.unlink(path)
                break

            ready, detail, reaped = self._wait_for_recorder_ready(proc, path)
            if ready:
                if attempt > 1:
                    print(
                        f"whisper-dictation: recorder recovered on startup attempt {attempt}",
                        file=sys.stderr,
                    )
                return proc, path

            self._last_recorder_error = detail
            print(
                f"whisper-dictation: recorder start attempt {attempt}/"
                f"{RECORDER_START_ATTEMPTS} failed: {detail}",
                file=sys.stderr,
            )
            if reaped:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            else:
                self._retain_rejected_recorder(proc, path)
                break
            if attempt < RECORDER_START_ATTEMPTS and RECORDER_RETRY_DELAY_SEC > 0:
                time.sleep(RECORDER_RETRY_DELAY_SEC)
        return None

    def _stage_chunk(
        self,
        wav: str | None,
        duration: float,
        seq: int,
        session_id: int,
        *,
        finalize: bool = False,
    ) -> None:
        """Finalize-order gate: chunks enter the worker FIFO in seq order."""
        session = self._session_paths.get(session_id)
        if session is None:
            print(
                f"whisper-dictation: no durable session for chunk {seq}; preserving {wav}",
                file=sys.stderr,
            )
            self._submit_in_order(seq, None)
            return
        if not wav:
            marker = SessionPasteJob(session, session_id) if finalize else None
            self._submit_in_order(seq, marker)
            return
        try:
            job = self._store.ingest(
                session,
                seq,
                Path(wav),
                duration,
                paste=True,
                paste_session=session_id,
                finalize=finalize,
            )
        except Exception as exc:
            print(
                f"whisper-dictation: could not persist chunk {seq}: {exc}",
                file=sys.stderr,
            )
            try:
                self._store.record_gap(session, seq, duration, str(exc))
            except Exception as marker_exc:
                print(
                    f"whisper-dictation: could not record chunk {seq} gap: {marker_exc}",
                    file=sys.stderr,
                )
            marker = SessionPasteJob(session, session_id) if finalize else None
            self._submit_in_order(seq, marker)
            return

        self._submit_in_order(seq, job)

    def _submit_in_order(
        self, seq: int, job: ChunkJob | SessionPasteJob | None
    ) -> None:
        with self._lock:
            self._staged[seq] = job
            while self._next_submit in self._staged:
                item = self._staged.pop(self._next_submit)
                if item is not None:
                    self._chunk_queue.put(item)
                self._next_submit += 1

    def _transcribe_worker(self) -> None:
        while True:
            item = self._chunk_queue.get()
            try:
                if item is None:
                    return
                if isinstance(item, SessionPasteJob):
                    self._deliver_completed_session_safely(
                        item.session_path, item.session_id
                    )
                    continue
                try:
                    self._deliver_chunk(item)
                except Exception as exc:
                    print(
                        f"whisper-dictation: chunk {item.chunk_index} failed: {exc}",
                        file=sys.stderr,
                    )
                    try:
                        self._store.fail(item, detail=str(exc))
                    except Exception as store_exc:
                        print(
                            f"whisper-dictation: could not record chunk failure: {store_exc}",
                            file=sys.stderr,
                        )
                finally:
                    if item.finalize:
                        self._deliver_completed_session_safely(
                            item.session_path, item.paste_session
                        )
            finally:
                self._chunk_queue.task_done()

    def _deliver_completed_session(self, session: Path, session_id: int) -> None:
        """Dispatch a live session once, after every preceding chunk is terminal."""
        text = self._store.completed_text(session)
        transcript = session / "transcript.txt"
        if not text:
            print(
                f"whisper-dictation: session={session.name} has no completed text; "
                f"saved={transcript}",
                file=sys.stderr,
            )
            self._notify(f"No usable speech — session saved at {transcript}")
            return

        if not self._insert(text):
            print(
                f"whisper-dictation: paste-failed session={session.name} "
                f"chars={len(text)} saved={transcript}",
                file=sys.stderr,
            )
            self._notify(f"Paste failed — transcript saved at {transcript}")
            return

        print(
            f"whisper-dictation: paste-dispatched session={session.name} "
            f"session_id={session_id} chars={len(text)}",
            file=sys.stderr,
        )
        preview = text if len(text) <= 60 else text[:57] + "…"
        self._notify(f"Typed: {preview}")

    def _deliver_completed_session_safely(self, session: Path, session_id: int) -> None:
        try:
            self._deliver_completed_session(session, session_id)
        except Exception as exc:
            print(
                f"whisper-dictation: session paste failed unexpectedly: {exc}; "
                f"saved={session / 'transcript.txt'}",
                file=sys.stderr,
            )
            self._notify(f"Paste failed — transcript saved at {session}")

    def _activate_recorder_locked(self, spawned: tuple[subprocess.Popen, str]) -> int:
        self._record_proc, self._wav_path = spawned
        self._record_start = time.monotonic()
        self._record_generation += 1
        self._arm_max_timer()
        return self._record_generation

    def _start_recorder_watch(
        self, proc: subprocess.Popen, wav: str, generation: int
    ) -> None:
        watcher = threading.Thread(
            target=self._watch_recorder,
            args=(proc, wav, generation),
            name=f"whisper-recorder-{generation}",
            daemon=True,
        )
        watcher.start()

    def _watch_recorder(
        self, proc: subprocess.Popen, wav: str, generation: int
    ) -> None:
        while proc.poll() is None:
            time.sleep(RECORDER_WATCH_POLL_SEC)
        self._handle_recorder_exit(
            proc,
            wav,
            generation,
            self._recorder_exit_detail(proc),
        )

    def _handle_recorder_exit(
        self,
        proc: subprocess.Popen,
        wav: str,
        generation: int,
        detail: str,
    ) -> None:
        """Replace a recorder only when it is still the active generation."""
        deadline = self._finalization_deadline(None)
        if not self._acquire_finalize_lock(deadline):
            print(
                "whisper-dictation: recorder recovery skipped at shutdown deadline",
                file=sys.stderr,
            )
            return
        try:
            self._handle_recorder_exit_locked(proc, wav, generation, detail)
        finally:
            self._finalize_lock.release()

    def _handle_recorder_exit_locked(
        self,
        proc: subprocess.Popen,
        wav: str,
        generation: int,
        detail: str,
    ) -> None:
        watch: tuple[subprocess.Popen, str, int] | None = None
        session: Path | None = None
        continue_recording = False
        with self._lock:
            if (
                self._record_proc is not proc
                or self._wav_path != wav
                or self._record_generation != generation
            ):
                return

            continue_recording = self._recording and self._shutdown_deadline is None
            duration = time.monotonic() - self._record_start
            seq = self._chunk_seq
            self._chunk_seq += 1
            session_id = self._session_id
            session = self._session_paths.get(session_id)
            self._record_proc = None
            self._wav_path = None
            self._cancel_max_timer()
            if self._recording and not continue_recording:
                self._recording = False
                self._tray.set_recording(False)
            action = "reconnecting" if continue_recording else "finalizing"
            print(
                f"whisper-dictation: recorder exited during chunk {seq} "
                f"after {duration:.2f}s ({detail}); {action}",
                file=sys.stderr,
            )

            if continue_recording:
                spawned = self._spawn_recorder()
                if spawned is None:
                    self._recording = False
                    self._tray.set_recording(False)
                else:
                    next_generation = self._activate_recorder_locked(spawned)
                    watch = (*spawned, next_generation)

        if watch is not None:
            self._start_recorder_watch(*watch)
        if watch is None:
            self._stage_chunk(wav, duration, seq, session_id, finalize=True)
        else:
            self._stage_chunk(wav, duration, seq, session_id)
        if watch is not None:
            return

        self._store.stop_session(session)
        if not continue_recording:
            return
        reason = self._last_recorder_error or detail
        self._notify(f"Recorder unavailable — saved prior audio ({reason[:80]})")

    def _start_recording(self) -> None:
        started = False
        hold = False
        watch: tuple[subprocess.Popen, str, int] | None = None
        with self._lock:
            if (
                self._shutdown_deadline is not None
                or self._recording
                or self._record_proc is not None
                or self._rejected_recorder_pending()
            ):
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
                reason = self._last_recorder_error or f"{self._recorder[0]} failed"
                self._notify(f"Recorder unavailable: {reason[:100]}")
                return
            self._recording = True
            now = time.monotonic()
            self._session_start = now
            self._session_id += 1
            self._active_session = self._store.start_session()
            self._session_paths[self._session_id] = self._active_session
            self._tray.set_recording(True)
            generation = self._activate_recorder_locked(spawned)
            watch = (*spawned, generation)
            started = True
            hold = self.hotkey_mode != "toggle"
        if watch is not None:
            self._start_recorder_watch(*watch)
        if started:
            self._notify("Listening…" if hold else "Recording… (Ctrl+Space to stop)")

    def _finalization_deadline(self, deadline: float | None) -> float:
        own_deadline = (
            time.monotonic() + SHUTDOWN_BUDGET_SEC - SHUTDOWN_RESOURCE_RESERVE_SEC
        )
        if deadline is not None:
            own_deadline = min(own_deadline, deadline)
        if self._shutdown_deadline is not None:
            own_deadline = min(
                own_deadline,
                self._shutdown_deadline - SHUTDOWN_RESOURCE_RESERVE_SEC,
            )
        return own_deadline

    def _acquire_finalize_lock(self, deadline: float) -> bool:
        return self._finalize_lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        )

    def _roll_recording(
        self,
        expected_generation: int | None = None,
        *,
        deadline: float | None = None,
    ) -> None:
        """Release the filled recorder, then acquire and validate its successor."""
        deadline = self._finalization_deadline(deadline)
        if not self._acquire_finalize_lock(deadline):
            print(
                "whisper-dictation: recording roll skipped at shutdown deadline",
                file=sys.stderr,
            )
            return
        try:
            self._roll_recording_locked(
                expected_generation=expected_generation,
                deadline=deadline,
            )
        finally:
            self._finalize_lock.release()

    def _roll_recording_locked(
        self,
        expected_generation: int | None,
        *,
        deadline: float,
    ) -> None:
        watch: tuple[subprocess.Popen, str, int] | None = None
        session: Path | None = None
        with self._lock:
            if (
                self._shutdown_deadline is not None
                or not self._recording
                or (
                    expected_generation is not None
                    and expected_generation != self._record_generation
                )
            ):
                return
            old_proc = self._record_proc
            old_wav = self._wav_path
            duration = time.monotonic() - self._record_start
            seq = self._chunk_seq
            session_id = self._session_id
            session = self._session_paths.get(session_id)
            self._cancel_max_timer()
            if not graceful_stop_recorder(
                old_proc,
                old_wav,
                0,
                deadline=deadline,
            ):
                print(
                    "whisper-dictation: recorder rollover retained until child exits",
                    file=sys.stderr,
                )
                return
            self._chunk_seq += 1
            self._record_proc = None
            self._wav_path = None
            spawned = self._spawn_recorder()
            if spawned is None:
                self._recording = False
                self._tray.set_recording(False)
            else:
                next_generation = self._activate_recorder_locked(spawned)
                watch = (*spawned, next_generation)

        if watch is not None:
            self._start_recorder_watch(*watch)
        if watch is None:
            self._stage_chunk(old_wav, duration, seq, session_id, finalize=True)
        else:
            self._stage_chunk(old_wav, duration, seq, session_id)
        if watch is not None:
            return

        self._store.stop_session(session)
        reason = self._last_recorder_error or "recorder start failed"
        self._notify(f"Recorder unavailable — saved prior audio ({reason[:80]})")

    def _finish_recording(self, *, deadline: float | None = None) -> None:
        deadline = self._finalization_deadline(deadline)
        if not self._acquire_finalize_lock(deadline):
            print(
                "whisper-dictation: recording finalizer exceeded shutdown deadline",
                file=sys.stderr,
            )
            return
        try:
            self._finish_recording_locked(deadline=deadline)
        finally:
            self._finalize_lock.release()

    def _finish_recording_locked(self, *, deadline: float | None = None) -> None:
        with self._lock:
            if not self._recording and self._record_proc is None:
                return
            if self._recording:
                self._recording = False
                self._cancel_max_timer()
                self._tray.set_recording(False)
            proc = self._record_proc
            wav = self._wav_path
            generation = self._record_generation
            duration = time.monotonic() - self._record_start
            seq = self._chunk_seq
            session_id = self._session_id
        if not graceful_stop_recorder(
            proc,
            wav,
            self.recorder_stop_flush_msec,
            deadline=deadline,
        ):
            print(
                "whisper-dictation: recorder finalization retained until child exits",
                file=sys.stderr,
            )
            return
        with self._lock:
            if (
                self._record_proc is not proc
                or self._wav_path != wav
                or self._record_generation != generation
            ):
                return
            self._record_proc = None
            self._wav_path = None
            self._chunk_seq += 1
        self._stage_chunk(wav, duration, seq, session_id, finalize=True)
        self._store.stop_session(self._session_paths.get(session_id))

    def _deliver_chunk(self, job: ChunkJob) -> None:
        wav = str(job.wav_path)
        duration = job.duration
        session_id = job.paste_session
        listening = job.paste and self._recording and session_id == self._session_id
        self._store.begin(job)
        if duration < self.min_record:
            self._store.terminal(job, "ignored", "recording was too short")
            if not listening:
                self._notify("Too short — speak longer, then Ctrl+Space to stop")
            return

        wav_bytes = os.path.getsize(wav) if os.path.exists(wav) else 0
        if wav_bytes < MIN_USABLE_WAV_BYTES:
            if not listening:
                hint = "check mic in Settings → Sound → Input"
                if self.audio_source:
                    hint = f"mic: {self.audio_source[:40]}… — {hint}"
                self._notify(f"Recording empty ({wav_bytes} B). {hint}.")
            status = "missing" if not os.path.exists(wav) else "no_text"
            self._store.terminal(job, status, f"audio file is {wav_bytes} bytes")
            return

        try:
            rms = wav_rms(wav)
        except (EOFError, wave.Error) as exc:
            self._store.fail(job, "corrupt", str(exc))
            return
        if rms < self.min_audio_rms:
            self._store.terminal(job, "silent", f"RMS {rms:.0f}")
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
            self._store.fail(job, detail=str(exc))
            return

        print(
            f"whisper-dictation: chunk={job.chunk_index} transcript_chars={len(text)} "
            f"rms={rms:.0f} duration={duration:.1f}s",
            file=sys.stderr,
        )
        if not text:
            self._store.terminal(job, "no_text", "transcription returned no text")
            if not listening:
                self._notify(
                    "No speech in recording — check mic level / wrong input device "
                    "(run: bash scripts/dictation/test-mic.sh)"
                )
            return

        # Punctuation-only is never real dictation. Phrase hallucinations like
        # "thank you" are only dropped on quiet clips so spoken "bye" still works.
        if is_punctuation_only(text) or (
            is_likely_hallucination(text) and rms < self.min_audio_rms * 2
        ):
            self._store.terminal(job, "ignored", f"likely hallucination: {text[:80]}")
            if not listening:
                self._notify(
                    f"Ignored likely silence hallucination ({text!r}) — try again with the mic closer"
                )
            return

        self._store.complete(job, text)
        listening = self._recording and session_id == self._session_id
        if job.paste and listening:
            self._notify(f"Transcribed chunk {job.chunk_index + 1} (still listening…)")

    def shutdown(self, drain_seconds: float = 5.0) -> None:
        """Finalize active audio and leave queued work recoverable for next start."""
        shutdown_deadline = time.monotonic() + SHUTDOWN_BUDGET_SEC
        with self._lock:
            if self._shutdown_deadline is None:
                self._shutdown_deadline = shutdown_deadline
            else:
                shutdown_deadline = self._shutdown_deadline
        work_deadline = shutdown_deadline - SHUTDOWN_RESOURCE_RESERVE_SEC
        try:
            # Calling unconditionally also waits for a hotkey finalizer that already
            # cleared _recording but still owns the recorder/WAV handoff.
            self._finish_recording(deadline=work_deadline)
            self._store.stop_session(self._active_session)
            drain_deadline = min(time.monotonic() + drain_seconds, work_deadline)
            while (
                self._chunk_queue.unfinished_tasks and time.monotonic() < drain_deadline
            ):
                time.sleep(0.05)
        finally:
            self._stop_clipboard_owner(deadline=shutdown_deadline)
            self._tray.stop(
                timeout=remaining_timeout(shutdown_deadline, TRAY_STOP_TIMEOUT_SEC)
            )

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
            if self._valid_server_text(text):
                return text

            restart_beam_size: int | None = None
            if text is None:
                print(
                    "whisper-dictation: server unavailable; recovering "
                    "(no accelerator CLI fallback)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"whisper-dictation: server returned unusable text {text!r}; "
                    "retrying with beam search",
                    file=sys.stderr,
                )
                text = self._transcribe_server(wav_path, beam_size=5)
                if self._valid_server_text(text):
                    return text
                restart_beam_size = 5

            self._notify("Dictation server stuck — restarting…")
            if self._restart_whisper_server():
                if restart_beam_size is None:
                    text = self._transcribe_server(wav_path)
                else:
                    text = self._transcribe_server(
                        wav_path, beam_size=restart_beam_size
                    )
                if self._valid_server_text(text):
                    return text
                if restart_beam_size is None and text is not None:
                    text = self._transcribe_server(wav_path, beam_size=5)
                    if self._valid_server_text(text):
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

    @staticmethod
    def _valid_server_text(text: str | None) -> bool:
        return bool(text and not is_punctuation_only(text))

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

    def _transcribe_server(
        self, wav_path: str, *, beam_size: int | None = None
    ) -> str | None:
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
            "no_timestamps=true",
            "-F",
            "token_timestamps=false",
            "-F",
            "response_format=json",
            "-F",
            f"language={self.language}",
        ]
        if beam_size is not None:
            cmd.extend(["-F", f"beam_size={beam_size}"])
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

    @staticmethod
    def _terminate_clipboard_process(
        proc: subprocess.Popen,
        *,
        deadline: float | None = None,
    ) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=remaining_timeout(deadline, CLIPBOARD_TERM_TIMEOUT_SEC))
            return
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            proc.wait(timeout=remaining_timeout(deadline, CLIPBOARD_KILL_TIMEOUT_SEC))
        except subprocess.TimeoutExpired:
            print(
                f"whisper-dictation: xclip {proc.pid} did not reap before shutdown deadline",
                file=sys.stderr,
            )

    def _stop_clipboard_owner(self, *, deadline: float | None = None) -> None:
        self._clipboard_closed.set()
        if deadline is None:
            acquired = self._clipboard_lock.acquire()
        else:
            acquired = self._clipboard_lock.acquire(
                timeout=max(0.0, deadline - time.monotonic())
            )
        if not acquired:
            return
        try:
            proc = self._clipboard_proc
            self._clipboard_proc = None
            if proc is not None:
                self._terminate_clipboard_process(proc, deadline=deadline)
        finally:
            self._clipboard_lock.release()

    def _start_clipboard_owner(self, text: str) -> bool:
        encoded = text.encode("utf-8")
        with self._clipboard_lock:
            if self._clipboard_closed.is_set():
                return False
            previous = self._clipboard_proc
            self._clipboard_proc = None
            if previous is not None:
                self._terminate_clipboard_process(previous)
            try:
                proc = subprocess.Popen(
                    ["xclip", "-quiet", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                return False
            try:
                if proc.stdin is None:
                    raise OSError("xclip stdin unavailable")
                proc.stdin.write(encoded)
                proc.stdin.close()
                ready_deadline = time.monotonic() + CLIPBOARD_READY_TIMEOUT_SEC
                while proc.poll() is None and time.monotonic() < ready_deadline:
                    try:
                        selection = subprocess.run(
                            ["xclip", "-selection", "clipboard", "-out"],
                            capture_output=True,
                            timeout=0.25,
                        )
                    except (OSError, subprocess.TimeoutExpired):
                        selection = None
                    if (
                        selection is not None
                        and selection.returncode == 0
                        and selection.stdout == encoded
                    ):
                        if self._clipboard_closed.is_set():
                            break
                        self._clipboard_proc = proc
                        return True
                    time.sleep(0.01)
            except (BrokenPipeError, OSError):
                pass
            self._terminate_clipboard_process(proc)
            return False

    def _run_xdotool_if_open(self, command: list[str]) -> bool:
        with self._clipboard_lock:
            if self._clipboard_closed.is_set():
                return False
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    timeout=XDOTOOL_TIMEOUT_SEC,
                )
            except (OSError, subprocess.TimeoutExpired):
                return False
            return result.returncode == 0

    def _insert(self, text: str) -> bool:
        xclip_available = (
            subprocess.run(["which", "xclip"], capture_output=True).returncode == 0
        )
        if not xclip_available or not self._start_clipboard_owner(text):
            # Direct typing cannot be rolled back: a timeout can leave an
            # arbitrary prefix in the target. Preserve the durable transcript
            # and report failure without sending any text events instead.
            return False
        return self._run_xdotool_if_open(
            ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
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
    app = Dictation(load_config())

    def stop(_signum: int, _frame: object) -> None:
        app.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    app.run()


if __name__ == "__main__":
    main()
