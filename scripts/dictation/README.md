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

This installs system packages, builds `whisper-cli` / `whisper-server`, downloads `small.en`, and enables the dictation service at login. The warm server unit is enabled only when `WHISPER_BACKEND=server` in config.

### 3. Verify

```bash
bash scripts/dictation/test-mic.sh    # speak when prompted (3 seconds)
bash scripts/dictation/check.sh
```

### 4. Use

Focus any text field → **Ctrl+Space** → speak → **Ctrl+Space** → text is pasted.

A **silver dot** in the desktop top-panel tray shows dictation is running; it **blinks bright red** while recording. Notifications are 1 s each and replace the previous one.

---

## Machine profiles: Spark workstation vs LG Gram laptop

`install.sh` builds **CPU + OpenBLAS**. Hotkey, tray LED, and paste are the same on both machines. Use **X11** (not Wayland) for global Ctrl+Space and `xdotool` paste.

### Tray LED — pin in Quick Settings (both machines)

The recording indicator is identical on Spark and the Gram. Pin it to the **top bar / Quick Settings** area so you always see whether dictation is recording (not buried in hidden tray icons).

| LED | Meaning |
|-----|---------|
| Silver dot | Dictation running, **not** recording |
| Red blink | **Recording** |

**Pin the icon (Ubuntu GNOME, X11):**

1. After install, confirm the service is up: `systemctl --user status whisper-dictation`
2. Open the system menu (top-right). If you see a silver dot only under **hidden icons** (chevron / overflow), open that list.
3. Pin **Whisper dictation** to the visible top bar:
   - Ubuntu 24.04+: **Settings → Ubuntu Desktop → Icons** — show tray icons and keep the dictation LED visible, or use the pin affordance in the tray overflow when offered.
4. Ensure `TRAY_INDICATOR="1"` in `~/.config/whisper-dictation/config.env` (default).

If no dot appears, install AppIndicator support (`bash scripts/dictation/install.sh`) and enable the **AppIndicator** / **KStatusNotifierItem** extension, then restart:

```bash
systemctl --user restart whisper-dictation
```

Once pinned, the LED is your always-on recording status — notifications are optional and can lag behind.

### Spark workstation

Typical setup: large turbo model, warm server, tuned thread count.

1. Install:

```bash
bash scripts/dictation/install.sh
```

2. In `~/.config/whisper-dictation/config.env`:

```bash
WHISPER_MODEL="large-v3-turbo-q8_0"
WHISPER_BACKEND="server"
WHISPER_THREADS="8"
```

3. Download model and enable the warm server:

```bash
./models/download-ggml-model.sh large-v3-turbo-q8_0
systemctl --user enable whisper-dictation-server
systemctl --user restart whisper-dictation-server whisper-dictation
```

4. Pin the tray LED (see above) so recording status stays visible in Quick Settings.

### LG Gram (Intel Evo i7, X11)

Same install path; bump threads to match the laptop CPU.

1. Install:

```bash
bash scripts/dictation/install.sh
```

2. In `~/.config/whisper-dictation/config.env`:

```bash
WHISPER_THREADS="8"    # match core count (8–12 on most Gram i7 configs)
WHISPER_MODEL="large-v3-turbo-q8_0"   # optional; better accuracy
WHISPER_BACKEND="server"              # optional; faster repeat dictation
```

3. If using the large model + server:

```bash
./models/download-ggml-model.sh large-v3-turbo-q8_0
systemctl --user enable whisper-dictation-server
systemctl --user restart whisper-dictation-server whisper-dictation
```

4. Pin the tray LED (see above) so recording status stays visible in Quick Settings.

`small.en` with `WHISPER_BACKEND=cli` is fine for a lighter first install on the Gram.

---

## Accuracy (recommended)

**Option 1 — better model + flags (default):** `WHISPER_MODEL=small.en` with `WHISPER_BACKEND=cli`, prompt, and non-speech suppression.

**Option 2 — best quality + warm model:** download turbo and use the server backend:

```bash
./models/download-ggml-model.sh large-v3-turbo-q8_0
# In ~/.config/whisper-dictation/config.env:
#   WHISPER_MODEL=large-v3-turbo-q8_0
#   WHISPER_BACKEND=server
systemctl --user restart whisper-dictation-server whisper-dictation
```

If the server is down, the daemon falls back to `whisper-cli` automatically.

---

## Install options

```bash
# Smaller / faster model (Pi):
WHISPER_MODEL=tiny.en-q5_1 bash scripts/dictation/install.sh

# Best quality model:
WHISPER_MODEL=large-v3-turbo-q8_0 bash scripts/dictation/install.sh

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
systemctl --user status whisper-dictation whisper-dictation-server
systemctl --user restart whisper-dictation whisper-dictation-server
systemctl --user stop whisper-dictation whisper-dictation-server
systemctl --user disable whisper-dictation whisper-dictation-server
```

---

## Configuration

File: `~/.config/whisper-dictation/config.env` (created on first install)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `small.en` | Model name (no `ggml-` prefix) |
| `WHISPER_BACKEND` | `cli` | `cli` or `server` (warm `whisper-server`) |
| `WHISPER_SERVER_URL` | `http://127.0.0.1:8178` | Server URL when backend=server |
| `WHISPER_LANGUAGE` | `en` | Language id |
| `WHISPER_SUPPRESS_NST` | `1` | Suppress non-speech tokens |
| `WHISPER_PROMPT` | (see config.env) | Initial prompt / vocabulary hints |
| `WHISPER_THREADS` | `4` | CPU threads |
| `HOTKEY_MODIFIERS` | `ctrl` | `alt`, `shift`, `ctrl`, `super` |
| `HOTKEY_KEY` | `space` | Trigger key |
| `AUDIO_SOURCE` | empty (system default) | PulseAudio source — list with `pactl list sources short` |
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
| Misheard words | Prefer `small.en` or `large-v3-turbo-q8_0`; set `WHISPER_PROMPT` with your jargon |
| Text pasted twice | Two daemons running — `rm ~/.config/autostart/whisper-dictation.desktop` then `systemctl --user restart whisper-dictation` |
| Server fallback | Check `systemctl --user status whisper-dictation-server` |
| Tray dot missing | Install AppIndicator support (`bash scripts/dictation/install.sh`), enable GNOME AppIndicator extension if tray icons are hidden, restart service |
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
| `dictation_indicator.py` | Top-panel tray LED (silver idle, red blink while recording) |
| `run-server.sh` | Warm `whisper-server` launcher |
| `whisper-dictation.service` | systemd user unit for the daemon |
| `whisper-dictation-server.service` | systemd user unit for the warm server |
| `config.env` | Default settings template |
| `check.sh` | Health check |
| `test-mic.sh` | Mic + transcription test |
