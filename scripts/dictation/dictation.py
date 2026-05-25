#!/usr/bin/env python3
"""Global dictation: record mic -> whisper.cpp -> type into focused field.

Toggle mode (default): Ctrl+Space to start, Ctrl+Space again to stop and paste.
Uses scripts/dictation/.venv if present (created by install.sh).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
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

MODIFIER_KEYS = {
    "alt": {keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
    "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
    "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
    "super": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
}


def build_recorder_cmd(audio_source: str) -> list[str] | None:
    """Build argv to record 16 kHz mono WAV; prefer parecord (reliable WAV finalize)."""
    source = audio_source.strip()
    if subprocess.run(["which", "parecord"], capture_output=True).returncode == 0:
        cmd = ["parecord", "--rate=16000", "--channels=1", "--file-format=wav"]
        if source:
            cmd.extend(["-d", source])
        return cmd
    if subprocess.run(["which", "pw-record"], capture_output=True).returncode == 0:
        cmd = ["pw-record", "--rate=16000", "--channels=1"]
        if source:
            cmd.extend(["--target", source])
        return cmd
    if subprocess.run(["which", "arecord"], capture_output=True).returncode == 0:
        cmd = ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1"]
        if source:
            cmd.extend(["-D", source])
        return cmd
    return None


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


class Dictation:
    def __init__(self, cfg: dict[str, str]) -> None:
        self.home = resolve_whisper_home(cfg)
        model_name = cfg.get("WHISPER_MODEL", "base.en-q5_1")
        self.model = self.home / "models" / f"ggml-{model_name}.bin"
        self.cli = self.home / "build/bin/whisper-cli"
        self.threads = cfg.get("WHISPER_THREADS", "4")
        self.min_record = float(cfg.get("MIN_RECORD_SEC", "0.4"))
        self.insert_method = cfg.get("INSERT_METHOD", "clipboard")
        self.mod_groups = parse_modifiers(cfg.get("HOTKEY_MODIFIERS", "ctrl"))
        self.trigger_key = resolve_trigger_key(cfg.get("HOTKEY_KEY", "space"))
        self.hotkey_mode = cfg.get("HOTKEY_MODE", "toggle").strip().lower()
        self.audio_source = cfg.get("AUDIO_SOURCE", "").strip()

        self._pressed: set = set()
        self._recording = False
        self._wav_path: str | None = None
        self._record_proc: subprocess.Popen | None = None
        self._record_start = 0.0
        self._lock = threading.Lock()
        self._busy = False
        self._recorder = build_recorder_cmd(self.audio_source)
        self._hotkey_chord_active = False  # ignore Space key-repeat until release

    def _mods_active(self) -> bool:
        if not self.mod_groups:
            return True
        return all(any(k in self._pressed for k in group) for group in self.mod_groups)

    def _key_id(self, key) -> object:
        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char.lower()
        return key

    def _is_trigger(self, key) -> bool:
        return key == self.trigger_key or self._key_id(key) == self._key_id(self.trigger_key)

    def _on_hotkey(self) -> None:
        if self.hotkey_mode == "hold":
            if not self._recording and not self._busy:
                threading.Thread(target=self._start_recording, daemon=True).start()
            return
        # toggle: start or stop on each Ctrl+Space press
        if self._recording:
            threading.Thread(target=self._finish_recording, daemon=True).start()
        elif not self._busy:
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
        try:
            subprocess.run(
                ["notify-send", "-a", "whisper-dictation", "-t", "2500", "Dictation", msg],
                check=False,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(msg, file=sys.stderr)

    def _start_recording(self) -> None:
        with self._lock:
            if self._recording or self._busy:
                return
            if not self.cli.is_file():
                self._notify(f"Missing {self.cli.name}; run scripts/dictation/install.sh")
                return
            if not self.model.is_file():
                self._notify(f"Missing model {self.model.name}")
                return
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="whisper-dictation-")
            os.close(fd)
            self._wav_path = path
            if not self._recorder:
                self._notify("No recorder (parecord/pw-record/arecord); run install.sh")
                os.unlink(path)
                self._wav_path = None
                return
            wake_audio_source(self.audio_source)
            try:
                self._record_proc = subprocess.Popen(
                    [*self._recorder, path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError:
                self._notify(f"{self._recorder[0]} not found")
                os.unlink(path)
                self._wav_path = None
                return
            self._recording = True
            self._record_start = time.monotonic()
            if self.hotkey_mode == "toggle":
                self._notify("Recording… (Ctrl+Space to stop)")
            else:
                self._notify("Listening…")

    def _finish_recording(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._busy = True
            proc = self._record_proc
            wav = self._wav_path
            duration = time.monotonic() - self._record_start

        if proc and proc.poll() is None:
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
        time.sleep(0.12)

        wav_bytes = os.path.getsize(wav) if wav and os.path.exists(wav) else 0
        if wav_bytes < 1000:
            if wav and os.path.exists(wav):
                os.unlink(wav)
            self._busy = False
            err = ""
            if proc and proc.stderr:
                try:
                    err = proc.stderr.read().decode(errors="replace")[:80]
                except Exception:
                    pass
            hint = "check mic in Settings → Sound → Input"
            if self.audio_source:
                hint = f"mic: {self.audio_source[:40]}… — {hint}"
            self._notify(f"Recording empty ({wav_bytes} B). {hint}. {err}")
            return

        if not wav or duration < self.min_record:
            if wav and os.path.exists(wav):
                os.unlink(wav)
            self._busy = False
            self._notify("Too short — speak longer, then Ctrl+Space to stop")
            return

        self._notify("Transcribing…")
        try:
            text = self._transcribe(wav)
        finally:
            if os.path.exists(wav):
                os.unlink(wav)
            self._busy = False

        if not text:
            self._notify(
                "No speech in recording — check mic level / wrong input device "
                "(run: bash scripts/dictation/test-mic.sh)"
            )
            return

        self._insert(text)
        preview = text if len(text) <= 60 else text[:57] + "…"
        self._notify(f"Typed: {preview}")

    def _transcribe(self, wav_path: str) -> str:
        result = subprocess.run(
            [
                str(self.cli),
                "-m",
                str(self.model),
                "-f",
                wav_path,
                "-nt",
                "-np",
                "-t",
                self.threads,
            ],
            cwd=self.home,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "transcription failed").strip()
            self._notify(err[:120])
            return ""
        lines = []
        for line in result.stdout.splitlines():
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

    def _insert(self, text: str) -> None:
        use_clipboard = self.insert_method == "clipboard" and subprocess.run(
            ["which", "xclip"], capture_output=True
        ).returncode == 0
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

    def run(self) -> None:
        ok = self.cli.is_file() and self.model.is_file()
        cfg = load_config()
        label = self._hotkey_label(cfg)
        if self.hotkey_mode == "toggle":
            usage = f"  Press {label} to start, press again to stop and paste"
        else:
            usage = f"  Hold {label} to record, release to paste"
        print(
            f"whisper-dictation ready\n"
            f"{usage}\n"
            f"  Home: {self.home}\n"
            f"  Model: {self.model.name} ({'ok' if self.model.is_file() else 'MISSING'})\n"
            f"  CLI:   {self.cli.name} ({'ok' if self.cli.is_file() else 'MISSING'})\n"
            f"  Mic:   {self.audio_source or '(system default)'}",
            flush=True,
        )
        if not ok:
            msg = "whisper-cli or model missing — run: bash scripts/dictation/install.sh"
            print(msg, file=sys.stderr)
            self._notify(msg)
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()


def main() -> None:
    Dictation(load_config()).run()


if __name__ == "__main__":
    main()
