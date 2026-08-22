#!/usr/bin/env bash
# One-shot setup: build whisper.cpp, download model, install dictation + boot autostart.
#
# Usage (fresh clone on Debian/Ubuntu / Raspberry Pi OS):
#   git clone https://github.com/YOUR_USER/whisper.cpp.git
#   cd whisper.cpp
#   bash scripts/dictation/install.sh
#
# Options:
#   WHISPER_MODEL=tiny.en-q5_1 ./install.sh   # smaller/faster model
#   WHISPER_MODEL=large-v3-turbo-q8_0 ./install.sh  # best quality (pair with server backend)
#   ./install.sh --no-autostart               # skip boot service
#   ./install.sh --autostart-only             # re-register autostart only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_SRC="${SCRIPT_DIR}/config.env"
CONFIG_DST="${HOME}/.config/whisper-dictation/config.env"
INSTALL_ENV="${HOME}/.config/whisper-dictation/install.env"
LOCAL_BIN="${HOME}/.local/bin/whisper-dictation"
SERVER_BIN="${HOME}/.local/bin/whisper-dictation-server"
MODEL="${WHISPER_MODEL:-small.en}"
AUTOSTART=1
AUTOSTART_ONLY=0

usage() {
    sed -n '2,12p' "$0"
    exit 0
}

for arg in "$@"; do
    case "${arg}" in
        -h|--help) usage ;;
        --no-autostart) AUTOSTART=0 ;;
        --autostart-only) AUTOSTART_ONLY=1 ;;
        *) echo "Unknown option: ${arg}" >&2; exit 1 ;;
    esac
done

write_install_env() {
    mkdir -p "${HOME}/.config/whisper-dictation"
    cat > "${INSTALL_ENV}" <<EOF
# Written by scripts/dictation/install.sh — do not edit unless you moved the repo
WHISPER_REPO_ROOT="${ROOT}"
WHISPER_MODEL="${MODEL}"
EOF
}

write_launcher() {
    mkdir -p "${HOME}/.local/bin"
    cat > "${LOCAL_BIN}" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
INSTALL_ENV="${HOME}/.config/whisper-dictation/install.env"
if [[ ! -f "${INSTALL_ENV}" ]]; then
    echo "whisper-dictation: run scripts/dictation/install.sh first" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${INSTALL_ENV}"
PY="${WHISPER_REPO_ROOT}/scripts/dictation/.venv/bin/python"
APP="${WHISPER_REPO_ROOT}/scripts/dictation/dictation.py"
if [[ ! -x "${PY}" ]]; then
    echo "whisper-dictation: missing venv at ${PY}" >&2
    exit 1
fi
exec "${PY}" "${APP}" "$@"
LAUNCHER
    chmod +x "${LOCAL_BIN}"

    cat > "${SERVER_BIN}" <<'SERVER'
#!/usr/bin/env bash
set -euo pipefail
INSTALL_ENV="${HOME}/.config/whisper-dictation/install.env"
if [[ ! -f "${INSTALL_ENV}" ]]; then
    echo "whisper-dictation-server: run scripts/dictation/install.sh first" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${INSTALL_ENV}"
exec "${WHISPER_REPO_ROOT}/scripts/dictation/run-server.sh" "$@"
SERVER
    chmod +x "${SERVER_BIN}"
}

detect_audio_source() {
    if [[ -f "${CONFIG_DST}" ]] && grep -q '^AUDIO_SOURCE=.' "${CONFIG_DST}" 2>/dev/null; then
        return
    fi
    if ! command -v pactl >/dev/null; then
        return
    fi
    local src
    src="$(pactl get-default-source 2>/dev/null || true)"
    if [[ -z "${src}" ]]; then
        src="$(pactl list sources short 2>/dev/null | awk '!/monitor/ && !/SUSPENDED/ {print $2; exit}')"
    fi
    if [[ -n "${src}" ]]; then
        mkdir -p "$(dirname "${CONFIG_DST}")"
        if [[ -f "${CONFIG_DST}" ]]; then
            if ! grep -q '^AUDIO_SOURCE=' "${CONFIG_DST}"; then
                echo "AUDIO_SOURCE=\"${src}\"" >> "${CONFIG_DST}"
            fi
        else
            cp "${CONFIG_SRC}" "${CONFIG_DST}"
            echo "AUDIO_SOURCE=\"${src}\"" >> "${CONFIG_DST}"
        fi
        echo "    Detected microphone: ${src}"
    fi
}

install_system_packages() {
    echo "==> System packages (sudo)"
    if ! command -v sudo >/dev/null; then
        echo "    No sudo — install manually: build-essential cmake libopenblas-dev"
        echo "    pulseaudio-utils xdotool xclip curl python3-venv"
        return
    fi
    sudo apt-get update -qq
    sudo apt-get install -y \
        build-essential cmake \
        libopenblas-dev \
        pulseaudio-utils \
        xdotool xclip curl \
        python3-venv \
        libnotify-bin \
        python3-gi \
        gir1.2-ayatanaappindicator3-0.1
}

install_python() {
    echo "==> Python venv (pynput)"
    python3 -m venv "${SCRIPT_DIR}/.venv"
    "${SCRIPT_DIR}/.venv/bin/pip" install -q --upgrade pip
    "${SCRIPT_DIR}/.venv/bin/pip" install -q pynput pystray pillow
}

build_whisper() {
    echo "==> Build whisper-cli"
    local blas_flag=-DGGML_BLAS=OFF
    if pkg-config --exists openblas 2>/dev/null; then
        blas_flag=-DGGML_BLAS=ON
    else
        echo "    (OpenBLAS not found — building without BLAS)"
    fi
    cmake -B "${ROOT}/build" "${blas_flag}" -DWHISPER_SDL2=OFF
    cmake --build "${ROOT}/build" -j"$(nproc)" --config Release
}

download_model() {
    echo "==> Download model: ${MODEL}"
    "${ROOT}/models/download-ggml-model.sh" "${MODEL}"
}

install_config() {
    echo "==> Config"
    mkdir -p "${HOME}/.config/whisper-dictation"
    if [[ ! -f "${CONFIG_DST}" ]]; then
        cp "${CONFIG_SRC}" "${CONFIG_DST}"
        echo "    Created ${CONFIG_DST}"
    else
        echo "    Keeping existing ${CONFIG_DST}"
    fi
    detect_audio_source
}

install_autostart() {
    echo "==> Autostart on login / boot"
    write_launcher
    write_install_env

    # Only systemd — do not also install a .desktop autostart. Running both
    # starts two daemons and every Ctrl+Space paste is typed twice.
    rm -f "${HOME}/.config/autostart/whisper-dictation.desktop"

    local display="${DISPLAY:-:0}"
    local backend="cli"
    if [[ -f "${CONFIG_DST}" ]]; then
        # shellcheck disable=SC1090
        set -a
        # shellcheck disable=SC1090
        source "${CONFIG_DST}" 2>/dev/null || true
        set +a
        backend="${WHISPER_BACKEND:-cli}"
    fi

    mkdir -p "${HOME}/.config/systemd/user"
    sed "s|%SERVER_BIN%|${SERVER_BIN}|g; s|%t|${XDG_RUNTIME_DIR:-/run/user/$(id -u)}|g" \
        "${SCRIPT_DIR}/whisper-dictation-server.service" \
        > "${HOME}/.config/systemd/user/whisper-dictation-server.service"
    sed "s|%LOCAL_BIN%|${LOCAL_BIN}|g; s|%DISPLAY%|${display}|g; s|%t|${XDG_RUNTIME_DIR:-/run/user/$(id -u)}|g" \
        "${SCRIPT_DIR}/whisper-dictation.service" \
        > "${HOME}/.config/systemd/user/whisper-dictation.service"

    systemctl --user daemon-reload
    systemctl --user enable whisper-dictation.service
    systemctl --user restart whisper-dictation.service 2>/dev/null || true
    echo "    systemd: whisper-dictation.service (enabled, DISPLAY=${display})"

    if [[ "${backend}" == "server" ]]; then
        systemctl --user enable whisper-dictation-server.service
        systemctl --user restart whisper-dictation-server.service 2>/dev/null || true
        echo "    systemd: whisper-dictation-server.service (enabled; WHISPER_BACKEND=server)"
    else
        systemctl --user disable whisper-dictation-server.service 2>/dev/null || true
        systemctl --user stop whisper-dictation-server.service 2>/dev/null || true
        echo "    systemd: whisper-dictation-server.service (disabled; set WHISPER_BACKEND=server to enable)"
    fi
}

if [[ "${AUTOSTART_ONLY}" -eq 1 ]]; then
    install_autostart
    echo ""
    echo "Autostart updated. Service status:"
    systemctl --user --no-pager status whisper-dictation.service 2>/dev/null || true
    exit 0
fi

echo "=== whisper.cpp dictation install ==="
echo "Repo: ${ROOT}"
echo ""

install_system_packages
install_python
build_whisper
download_model
install_config
write_install_env
write_launcher

if [[ "${AUTOSTART}" -eq 1 ]]; then
    install_autostart
fi

echo ""
echo "=== Done ==="
echo "  CLI:     ${ROOT}/build/bin/whisper-cli"
echo "  Model:   ${ROOT}/models/ggml-${MODEL}.bin"
echo "  Config:  ${CONFIG_DST}"
echo "  Launcher: ${LOCAL_BIN}"
echo ""
echo "Hotkey: Ctrl+Space → record, Ctrl+Space → stop & paste"
echo ""
if [[ "${AUTOSTART}" -eq 1 ]]; then
    echo "Autostart is ON (starts after login)."
    echo "  Status:  systemctl --user status whisper-dictation whisper-dictation-server"
    echo "  Stop:    systemctl --user stop whisper-dictation whisper-dictation-server"
    echo "  Disable: systemctl --user disable whisper-dictation whisper-dictation-server"
    echo "           # if an old GNOME autostart remains:"
    echo "           rm -f ~/.config/autostart/whisper-dictation.desktop"
else
    echo "Start manually: whisper-dictation"
    echo "Optional warm server: whisper-dictation-server"
fi
echo ""
echo "Verify mic: bash ${SCRIPT_DIR}/test-mic.sh"
echo "Health check: bash ${SCRIPT_DIR}/check.sh"
