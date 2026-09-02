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

Keep talking past 45 seconds: the first slice pastes while the next slice is already recording. Press Ctrl+Space when you are done; the last partial slice pastes after that.

A **silver dot** in the desktop top-panel tray shows dictation is running; it **blinks bright red** while recording. Notifications are 1 s each and replace the previous one.

---

## Machine profiles: Spark workstation vs LG Gram laptop

`install.sh` always builds the portable **CPU + OpenBLAS** path in `build/`. Optional CUDA and SYCL builds live beside it, so pulling this fork never replaces another machine's selected backend. Existing machine-local config is preserved by `install.sh`.

| Backend | Build | Select with |
|---------|-------|-------------|
| CPU (portable default) | `bash scripts/dictation/install.sh` | `WHISPER_BUILD_DIR="build"` |
| NVIDIA CUDA | `bash scripts/dictation/build-cuda.sh` | `WHISPER_BUILD_DIR="build-cuda"` |
| Intel SYCL Level Zero | `bash scripts/dictation/build-sycl.sh` | `WHISPER_BUILD_DIR="build-sycl"` |

Keep `WHISPER_ACCELERATOR="auto"`; it detects CUDA/SYCL from the build name or CMake cache. Hotkey, tray LED, and paste behavior are shared across backends. Use **X11** (not Wayland) for global Ctrl+Space and `xdotool` paste.

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

### Spark workstation (NVIDIA CUDA)

Typical setup: large turbo model, warm server, tuned thread count.

1. Install:

```bash
bash scripts/dictation/install.sh
```

2. With the NVIDIA driver and CUDA toolkit installed, create the side-by-side CUDA build:

```bash
bash scripts/dictation/build-cuda.sh
```

Set `WHISPER_CUDA_ARCHITECTURES` only when the CUDA toolkit cannot detect the installed GPU architecture automatically.

3. In `~/.config/whisper-dictation/config.env`:

```bash
WHISPER_MODEL="large-v3-turbo-q8_0"
WHISPER_BACKEND="server"
WHISPER_THREADS="8"
WHISPER_BUILD_DIR="build-cuda"
WHISPER_ACCELERATOR="auto"
```

4. Download the model and refresh the services:

```bash
./models/download-ggml-model.sh large-v3-turbo-q8_0
WHISPER_MODEL=large-v3-turbo-q8_0 bash scripts/dictation/install.sh --autostart-only
systemctl --user restart whisper-dictation-server whisper-dictation
```

5. Pin the tray LED (see above) so recording status stays visible in Quick Settings.

### LG Gram (Intel Evo i7, X11)

Same install path; bump threads to match the laptop CPU.

1. Install:

```bash
bash scripts/dictation/install.sh
```

2. With the Intel oneAPI C++/SYCL compiler and Level Zero runtime installed, create or refresh the side-by-side GPU build:

```bash
bash scripts/dictation/build-sycl.sh
```

The helper requires the configured Intel Level Zero GPU to appear in `sycl-ls`. The LG profile pins the expected device name to Iris Xe with `WHISPER_SYCL_EXPECTED_DEVICE`; leave that setting empty on other Intel GPUs. It configures the supported `whisper-cli` and `whisper-server` targets in `build-sycl/` while preserving the CPU build in `build/`.

3. Select the persistent warm server in `~/.config/whisper-dictation/config.env`:

```bash
WHISPER_THREADS="8"
WHISPER_MODEL="small.en"
WHISPER_BUILD_DIR="build-sycl"
WHISPER_ACCELERATOR="auto"
WHISPER_ONEAPI_SETVARS="/opt/intel/oneapi/setvars.sh"
WHISPER_ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
WHISPER_SYCL_DEVICE="0"
WHISPER_SYCL_EXPECTED_DEVICE="Intel Iris Xe Graphics"
WHISPER_BACKEND="server"
WHISPER_SERVER_WARMUP="1"
```

4. Refresh and restart the units. The server performs one disposable inference before systemd marks it ready:

```bash
WHISPER_MODEL=small.en bash scripts/dictation/install.sh --autostart-only
systemctl --user enable whisper-dictation-server
systemctl --user restart whisper-dictation-server whisper-dictation
```

5. Pin the tray LED (see above) so recording status stays visible in Quick Settings.

Rollback does not rebuild or uninstall anything. With `WHISPER_ACCELERATOR="auto"`, change only `WHISPER_BUILD_DIR="build"`, then restart both services:

```bash
systemctl --user restart whisper-dictation-server whisper-dictation
```

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
| `WHISPER_BUILD_DIR` | `build` | Build directory under the repo, such as `build`, `build-cuda`, or `build-sycl` |
| `WHISPER_ACCELERATOR` | `auto` | Detect CUDA/SYCL from the build name or CMake cache; otherwise use CPU. Explicit `cuda` or `sycl` supports nonstandard build names |
| `WHISPER_ONEAPI_SETVARS` | `/opt/intel/oneapi/setvars.sh` | oneAPI environment script used for SYCL |
| `WHISPER_ONEAPI_DEVICE_SELECTOR` | `level_zero:gpu` | oneAPI device filter |
| `WHISPER_SYCL_DEVICE` | `0` | `GGML_SYCL_DEVICE` index |
| `WHISPER_SYCL_EXPECTED_DEVICE` | empty | Optional strict name substring for the selected Intel Level Zero GPU |
| `WHISPER_BACKEND` | `cli` | `cli` or `server` (warm `whisper-server`) |
| `WHISPER_SERVER_URL` | `http://127.0.0.1:8178` | Server URL when backend=server |
| `WHISPER_SERVER_WARMUP` | `1` | Run one inference before the server reports ready |
| `WHISPER_SERVER_WARMUP_AUDIO` | JFK sample | Optional warmup WAV override |
| `WHISPER_SERVER_WARMUP_TIMEOUT` | `120` | Startup/warmup timeout in seconds |
| `WHISPER_SERVER_TIMEOUT` | `90` | Client `/inference` curl budget in seconds |
| `WHISPER_INFERENCE_WATCHDOG_SEC` | `90` | Kill and restart the server if an inference socket stays open this long (`0` disables) |
| `WHISPER_SERVER_RECYCLE_SEC` | `21600` | Recycle a healthy idle server this often (6h) to shed Intel GPU hangs |
| `MAX_RECORD_SEC` | `45` | Slice length. Auto-rolls into the next recording so long speech keeps landing in 45s pastes (`0` = unlimited single take) |
| `WHISPER_LANGUAGE` | `en` | Language id |
| `WHISPER_SUPPRESS_NST` | `1` | Suppress non-speech tokens |
| `WHISPER_PROMPT_PREFIX` | `Technical dictation.` | Prefix for the vocabulary prompt |
| `WHISPER_VOCABULARY_FILE` | `~/.config/whisper-dictation/vocabulary.txt` | Optional machine-local overlay (shared terms are in-repo) |
| `WHISPER_PROMPT` | (see config.env) | Fallback prompt if no vocabulary file exists |
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
| Misheard words | Add the term to `scripts/dictation/vocabulary.txt` (see Terminology below), then restart |
| Text pasted twice | Two daemons running — `rm ~/.config/autostart/whisper-dictation.desktop` then `systemctl --user restart whisper-dictation` |
| Server fallback | Check `systemctl --user status whisper-dictation-server` |
| Recording LED works but no text | SYCL server hung — `/health` still returns ok. `systemctl --user restart whisper-dictation-server`. A watchdog now kills stuck inferences automatically. |
| First GPU request is slow | Keep `WHISPER_SERVER_WARMUP=1`; wait for the server unit to become `active` before dictating |
| CUDA backend missing | Run `nvidia-smi`, confirm `nvcc` is installed, rebuild with `build-cuda.sh`, then run `check.sh` |
| SYCL device missing | Source oneAPI and run `ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls` |
| Tray dot missing | Install AppIndicator support (`bash scripts/dictation/install.sh`), enable GNOME AppIndicator extension if tray icons are hidden, restart service |
| Service not running | `systemctl --user status whisper-dictation` |
| Wrong repo path | `bash scripts/dictation/install.sh --autostart-only` |
| Hotkey conflict | Change `HOTKEY_*` in config, restart service |

---

## Terminology (shared vocabulary)

`scripts/dictation/vocabulary.txt` is the committed term list used on every machine after `git pull`. Whisper sees the canonical spellings as an initial prompt; `canonical => spoken / mishearing` lines also rewrite those forms after transcription.

**Primary spoken form for `os_droid`:** say **OS underscore droid**. Saying “osdroid” as one word is heard as *oesteroid*, *oysteroid*, and similar.

```text
os_droid => OS underscore droid / oesteroid / oysteroid
```

Add a term, commit, pull on the other laptop, then restart:

```bash
systemctl --user restart whisper-dictation whisper-dictation-server
```

Optional machine-local extras (not in git): `~/.config/whisper-dictation/vocabulary.txt`.

---

## Files in this directory

| File | Purpose |
|------|---------|
| `install.sh` | Full setup |
| `uninstall.sh` | Remove autostart |
| `dictation.py` | Hotkey daemon |
| `dictation_indicator.py` | Top-panel tray LED (silver idle, red blink while recording) |
| `run-server.sh` | Warm `whisper-server` launcher |
| `build-cuda.sh` | Reproducible NVIDIA CUDA build helper |
| `build-sycl.sh` | Reproducible Intel Level Zero SYCL build helper |
| `runtime-env.sh` | Shared side-by-side build and accelerator environment resolver |
| `validate-server-response.py` | Shared strict JSON response validator for startup and health checks |
| `whisper-dictation.service` | systemd user unit for the daemon |
| `whisper-dictation-server.service` | systemd user unit for the warm server |
| `config.env` | Default settings template |
| `vocabulary.txt` | Shared terminology (committed; used on all machines) |
| `vocabulary.txt.example` | Template for optional local overlay terms |
| `vocab_prompt.py` | Builds the Whisper prompt and applies replacements |
| `check.sh` | Health check |
| `test-mic.sh` | Mic + transcription test |
