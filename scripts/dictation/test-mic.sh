#!/usr/bin/env bash
# Record 3 seconds from the configured mic and run whisper (sanity check).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runtime-env.sh"
whisper_dictation_load_runtime
ROOT="${WHISPER_REPO_ROOT}"
CFG="${HOME}/.config/whisper-dictation/config.env"
CLI="${WHISPER_BIN_DIR}/whisper-cli"
MODEL_NAME="${WHISPER_MODEL:-small.en}"
MODEL="${ROOT}/models/ggml-${MODEL_NAME}.bin"
WAV="/tmp/whisper-mic-test.wav"
SOURCE=""

SOURCE="${AUDIO_SOURCE:-}"

echo "=== Mic test (speak when recording starts) ==="
echo "Source: ${SOURCE:-system default}"
echo ""

rm -f "${WAV}"
if command -v parecord >/dev/null; then
    ARGS=(parecord --rate=16000 --channels=1 --file-format=wav)
    [[ -n "${SOURCE}" ]] && ARGS+=(-d "${SOURCE}")
    ARGS+=("${WAV}")
    echo "Recording 3s with parecord…"
    "${ARGS[@]}" &
    PID=$!
    sleep 3
    kill -INT "${PID}" 2>/dev/null || true
    wait "${PID}" 2>/dev/null || true
else
    echo "parecord not found"; exit 1
fi

SZ=$(stat -c%s "${WAV}" 2>/dev/null || echo 0)
echo "WAV size: ${SZ} bytes"
if [[ "${SZ}" -lt 1000 ]]; then
    echo "FAIL: recording is empty — pick the right mic in Settings → Sound → Input"
    echo "List sources: pactl list sources short"
    exit 1
fi

echo "Transcribing…"
OUT=$("${CLI}" -m "${MODEL}" -f "${WAV}" -nt -np -t "${WHISPER_THREADS:-4}" 2>/dev/null | grep -v '^\[' | tail -1)
rm -f "${WAV}"
if [[ -z "${OUT}" || "${OUT}" == *"BLANK_AUDIO"* ]]; then
    echo "FAIL: whisper heard silence. Check mic volume / mute / wrong device."
    echo "Set AUDIO_SOURCE in ${CFG} — run: pactl list sources short"
    exit 1
fi
echo "OK: ${OUT}"
