#!/usr/bin/env bash
# Quick health check for whisper dictation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${HOME}/.config/whisper-dictation/config.env"
PY="${ROOT}/scripts/dictation/.venv/bin/python"
CLI="${ROOT}/build/bin/whisper-cli"
MODEL="${ROOT}/models/ggml-base.en-q5_1.bin"
FAIL=0

ok() { echo "  OK   $*"; }
bad() { echo "  FAIL $*"; FAIL=1; }

echo "=== whisper-dictation check ==="
echo ""

[[ -x "${CLI}" ]] && ok "whisper-cli: ${CLI}" || bad "whisper-cli missing — run install.sh"
[[ -f "${MODEL}" ]] && ok "model: ${MODEL}" || bad "model missing — ./models/download-ggml-model.sh base.en-q5_1"

if [[ -f "${CFG}" ]] && grep -q '\$' "${CFG}" 2>/dev/null; then
    bad "config has bash syntax in WHISPER_HOME — remove or comment that line in ${CFG}"
else
    ok "config: ${CFG}"
fi

REC=0
for cmd in pw-record parecord arecord; do
    command -v "${cmd}" >/dev/null && ok "recorder: ${cmd}" && REC=1 && break
done
[[ "${REC}" -eq 1 ]] || bad "no audio recorder (install pulseaudio-utils)"

command -v xdotool >/dev/null && ok xdotool || bad "xdotool missing"
command -v xclip >/dev/null && ok xclip || echo "  WARN xclip missing (will use xdotool type)"

[[ -x "${PY}" ]] && "${PY}" -c "from pynput import keyboard" 2>/dev/null && ok pynput || bad "pynput — run install.sh"

if systemctl --user is-active whisper-dictation.service &>/dev/null; then
    echo ""
    ok "systemd: whisper-dictation.service active"
elif pgrep -f "dictation.py" >/dev/null; then
    echo ""
    ok "daemon running (PID $(pgrep -f 'dictation.py' | head -1))"
else
    echo ""
    echo "  Daemon not running — enable: bash scripts/dictation/install.sh --autostart-only"
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
    OUT=$("${CLI}" -m "${MODEL}" -f "${ROOT}/samples/jfk.wav" -nt -np -t 4 2>/dev/null | tail -1)
    [[ -n "${OUT}" ]] && ok "transcription: ${OUT:0:60}..." || bad "transcription returned empty"
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
    echo "All checks passed. Ctrl+Space to start, Ctrl+Space again to stop and paste."
else
    echo "Fix failures above, then: pkill -f dictation.py; whisper-dictation"
    exit 1
fi
