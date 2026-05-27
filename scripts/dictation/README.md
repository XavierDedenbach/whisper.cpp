# Desktop dictation (Ctrl+Space)

Press **Ctrl+Space** to start recording, **Ctrl+Space** again to stop. [whisper.cpp](https://github.com/ggml-org/whisper.cpp) transcribes locally and pastes into the focused text field.

Works on **Raspberry Pi 5**, Ubuntu, and other Debian-based desktops with **X11**.

---

## Setup on another machine

**You need:** Debian/Ubuntu (arm64 or amd64), `git`, `sudo`, a working mic, and network access.

### 1. Clone

```bash
git clone https://github.com/XavierDedenbach/whisper.cpp.git
cd whisper.cpp
```

### 2. Install (build, model, autostart)

```bash
bash scripts/dictation/install.sh
```

This installs system packages, builds `whisper-cli`, downloads `base.en-q5_1`, and enables the dictation service at login.

### 3. Verify

```bash
bash scripts/dictation/test-mic.sh    # speak when prompted (3 seconds)
bash scripts/dictation/check.sh
```

### 4. Use

Focus any text field → **Ctrl+Space** → speak → **Ctrl+Space** → text is pasted.

---

## Install options

```bash
# Smaller / faster model (Pi):
WHISPER_MODEL=tiny.en-q5_1 bash scripts/dictation/install.sh

# No autostart:
bash scripts/dictation/install.sh --no-autostart

# Repo moved — refresh paths and service:
bash scripts/dictation/install.sh --autostart-only

# Remove autostart only:
bash scripts/dictation/uninstall.sh
```

---

## Service commands

```bash
systemctl --user status whisper-dictation
systemctl --user restart whisper-dictation
systemctl --user stop whisper-dictation
systemctl --user disable whisper-dictation
```

---

## Configuration

File: `~/.config/whisper-dictation/config.env` (created on first install)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base.en-q5_1` | Model name |
| `WHISPER_THREADS` | `4` | CPU threads |
| `HOTKEY_MODIFIERS` | `ctrl` | `alt`, `shift`, `ctrl`, `super` |
| `HOTKEY_KEY` | `space` | Trigger key |
| `AUDIO_SOURCE` | auto-detected | PulseAudio source — list with `pactl list sources short` |
| `INSERT_METHOD` | `clipboard` | `clipboard` or `type` |

Example — set a specific USB mic:

```bash
pactl list sources short
# Edit config:
nano ~/.config/whisper-dictation/config.env
# AUDIO_SOURCE="alsa_input.usb-Your_Mic-00.mono-fallback"
systemctl --user restart whisper-dictation
```

---

## Troubleshooting

| Problem | Command / fix |
|---------|----------------|
| No speech detected | `bash scripts/dictation/test-mic.sh` — set `AUDIO_SOURCE` in config |
| Service not running | `systemctl --user status whisper-dictation` |
| Wrong repo path | `bash scripts/dictation/install.sh --autostart-only` |
| Hotkey conflict | Change `HOTKEY_*` in config, restart service |

---

## Files in this directory

| File | Purpose |
|------|---------|
| `install.sh` | Full setup |
| `uninstall.sh` | Remove autostart |
| `dictation.py` | Hotkey daemon |
| `config.env` | Default settings template |
| `check.sh` | Health check |
| `test-mic.sh` | Mic + transcription test |
