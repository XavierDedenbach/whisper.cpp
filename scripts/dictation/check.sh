#!/usr/bin/env bash
# Quick health check for whisper dictation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${HOME}/.config/whisper-dictation/config.env"
PY="${ROOT}/scripts/dictation/.venv/bin/python"
CLI="${ROOT}/build/bin/whisper-cli"
SERVER="${ROOT}/build/bin/whisper-server"
FAIL=0

ok() { echo "  OK   $*"; }
bad() { echo "  FAIL $*"; FAIL=1; }

# Resolve configured model / backend
MODEL_NAME="small.en"
BACKEND="cli"
SERVER_URL="http://127.0.0.1:8178"
if [[ -f "${CFG}" ]]; then
    # shellcheck disable=SC1090
    set -a
    # shellcheck disable=SC1090
    source "${CFG}" 2>/dev/null || true
    set +a
    MODEL_NAME="${WHISPER_MODEL:-$MODEL_NAME}"
    BACKEND="${WHISPER_BACKEND:-$BACKEND}"
    SERVER_URL="${WHISPER_SERVER_URL:-$SERVER_URL}"
fi
MODEL="${ROOT}/models/ggml-${MODEL_NAME}.bin"

echo "=== whisper-dictation check ==="
echo ""

[[ -x "${CLI}" ]] && ok "whisper-cli: ${CLI}" || bad "whisper-cli missing — run install.sh"
[[ -x "${SERVER}" ]] && ok "whisper-server: ${SERVER}" || echo "  WARN whisper-server missing (needed for WHISPER_BACKEND=server)"
[[ -f "${MODEL}" ]] && ok "model: ${MODEL}" || bad "model missing — ./models/download-ggml-model.sh ${MODEL_NAME}"

if [[ -f "${CFG}" ]] && grep -q '\$' "${CFG}" 2>/dev/null; then
    bad "config has bash syntax in WHISPER_HOME — remove or comment that line in ${CFG}"
else
    ok "config: ${CFG} (model=${MODEL_NAME}, backend=${BACKEND})"
fi

REC=0
for cmd in pw-record parecord arecord; do
    command -v "${cmd}" >/dev/null && ok "recorder: ${cmd}" && REC=1 && break
done
[[ "${REC}" -eq 1 ]] || bad "no audio recorder (install pulseaudio-utils)"

command -v xdotool >/dev/null && ok xdotool || bad "xdotool missing"
command -v xclip >/dev/null && ok xclip || echo "  WARN xclip missing (will use xdotool type)"
command -v curl >/dev/null && ok curl || echo "  WARN curl missing (needed for server backend)"

[[ -x "${PY}" ]] && "${PY}" -c "from pynput import keyboard" 2>/dev/null && ok pynput || bad "pynput — run install.sh"

# Match the python daemon only (not this script's own argv, which mentions dictation.py).
mapfile -t DICT_PIDS < <(pgrep -f "[p]ython.*/scripts/dictation/dictation\.py" || true)
if [[ "${#DICT_PIDS[@]}" -gt 1 ]]; then
    echo ""
    bad "multiple dictation daemons: ${DICT_PIDS[*]} (causes doubled paste)"
    echo "         fix: rm -f ~/.config/autostart/whisper-dictation.desktop"
    echo "              systemctl --user restart whisper-dictation"
elif systemctl --user is-active whisper-dictation.service &>/dev/null; then
    echo ""
    ok "systemd: whisper-dictation.service active"
elif [[ "${#DICT_PIDS[@]}" -eq 1 ]]; then
    echo ""
    ok "daemon running (PID ${DICT_PIDS[0]})"
else
    echo ""
    echo "  Daemon not running — enable: bash scripts/dictation/install.sh --autostart-only"
fi
if systemctl --user is-active whisper-dictation-server.service &>/dev/null; then
    ok "systemd: whisper-dictation-server.service active"
elif [[ "${BACKEND}" == "server" ]]; then
    echo "  WARN whisper-dictation-server inactive (backend=server; daemon will fall back to CLI)"
fi
if [[ -f "${HOME}/.config/autostart/whisper-dictation.desktop" ]]; then
    echo "  WARN desktop autostart present — remove it to avoid a second daemon alongside systemd"
fi
INSTALL_ENV="${HOME}/.config/whisper-dictation/install.env"
if [[ -f "${INSTALL_ENV}" ]]; then
    ok "install.env: ${INSTALL_ENV}"
else
    echo "  WARN install.env missing — run install.sh"
fi

echo ""
echo "=== transcribe test (samples/jfk.wav) ==="
if [[ -x "${CLI}" && -f "${MODEL}" ]]; then
    OUT=$("${CLI}" -m "${MODEL}" -f "${ROOT}/samples/jfk.wav" -nt -np -t 4 -l en -sns 2>/dev/null | tail -1)
    [[ -n "${OUT}" ]] && ok "cli transcription: ${OUT:0:60}..." || bad "transcription returned empty"
fi
if [[ "${BACKEND}" == "server" ]]; then
    if OUT=$(curl -sS --connect-timeout 2 --max-time 60 \
        "${SERVER_URL}/inference" \
        -F "file=@${ROOT}/samples/jfk.wav" \
        -F "temperature=0.0" \
        -F "response_format=json" 2>/dev/null); then
        echo "${OUT}" | grep -q '"text"' \
            && ok "server transcription reachable at ${SERVER_URL}" \
            || echo "  WARN server responded but no text field"
    else
        echo "  WARN server inference failed at ${SERVER_URL} (CLI fallback will be used)"
    fi
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
    echo "All checks passed. Ctrl+Space to start, Ctrl+Space again to stop and paste."
else
    echo "Fix failures above, then: systemctl --user restart whisper-dictation"
    exit 1
fi
