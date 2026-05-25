# Whisper dictation (fork-friendly setup)

Press **Ctrl+Space** to start recording, **Ctrl+Space** again to stop. Transcription runs locally with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and pastes into the focused text field.

Works on **Raspberry Pi 5**, Ubuntu, and other Debian-based desktops with X11.

## Fork and install on a new machine

```bash
# 1. Fork github.com/ggml-org/whisper.cpp on GitHub, then:
git clone https://github.com/YOUR_USER/whisper.cpp.git
cd whisper.cpp

# 2. One-command setup (build, model, mic config, autostart on boot):
bash scripts/dictation/install.sh

# 3. Test microphone (speak during the 3-second recording):
bash scripts/dictation/test-mic.sh
```

After install, dictation starts automatically at login via **systemd user service** and a **GNOME autostart** entry.

### Options

```bash
# Faster model (less accurate):
WHISPER_MODEL=tiny.en-q5_1 bash scripts/dictation/install.sh

# Install without autostart:
bash scripts/dictation/install.sh --no-autostart

# Re-point autostart after moving the repo:
bash scripts/dictation/install.sh --autostart-only

# Remove autostart only (keep build + models):
bash scripts/dictation/uninstall.sh
```

## Usage

1. Focus any text field.
2. **Ctrl+Space** — notification: “Recording…”
3. Speak.
4. **Ctrl+Space** again — “Transcribing…”, then text is pasted.

## Files (for forks)

| Path | Purpose |
|------|---------|
| `scripts/dictation/install.sh` | Full setup script |
| `scripts/dictation/uninstall.sh` | Remove autostart / launcher |
| `scripts/dictation/dictation.py` | Hotkey daemon |
| `scripts/dictation/config.env` | Default settings template |
| `scripts/dictation/check.sh` | Health check |
| `scripts/dictation/test-mic.sh` | Mic + whisper test |
| `~/.config/whisper-dictation/config.env` | Per-machine config (created on install) |
| `~/.config/whisper-dictation/install.env` | Repo path (created on install) |

## Configuration

Edit `~/.config/whisper-dictation/config.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base.en-q5_1` | Model name (see `models/download-ggml-model.sh`) |
| `WHISPER_THREADS` | `4` | CPU threads |
| `HOTKEY_MODE` | `toggle` | `toggle` or `hold` |
| `HOTKEY_MODIFIERS` | `ctrl` | `alt`, `shift`, `ctrl`, `super` |
| `HOTKEY_KEY` | `space` | Trigger key |
| `AUDIO_SOURCE` | auto-detected | PulseAudio source (`pactl list sources short`) |
| `INSERT_METHOD` | `clipboard` | `clipboard` or `type` |

If you move the repo to a new path, run:

```bash
bash scripts/dictation/install.sh --autostart-only
```

## Autostart management

```bash
systemctl --user status whisper-dictation   # running?
systemctl --user restart whisper-dictation
systemctl --user stop whisper-dictation
systemctl --user disable whisper-dictation  # disable boot start
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No speech / empty recording | `bash scripts/dictation/test-mic.sh`, set `AUDIO_SOURCE` |
| Service not running after boot | `systemctl --user status whisper-dictation` |
| Wrong repo after move | `bash scripts/dictation/install.sh --autostart-only` |
| Ctrl+Space conflicts | Change `HOTKEY_*` in config |

## Requirements

- Debian/Ubuntu (arm64 or x86_64)
- X11 session (GNOME on Pi tested)
- Microphone via PipeWire/PulseAudio
