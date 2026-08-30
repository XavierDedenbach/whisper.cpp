#!/usr/bin/env bash
# Quick health check for whisper dictation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runtime-env.sh"
whisper_dictation_load_runtime
ROOT="${WHISPER_REPO_ROOT}"
CFG="${HOME}/.config/whisper-dictation/config.env"
PY="${ROOT}/scripts/dictation/.venv/bin/python"
CLI="${WHISPER_BIN_DIR}/whisper-cli"
SERVER="${WHISPER_BIN_DIR}/whisper-server"
FAIL=0

ok() { echo "  OK   $*"; }
bad() { echo "  FAIL $*"; FAIL=1; }

# Resolve configured model / backend
MODEL_NAME="small.en"
BACKEND="cli"
SERVER_URL="http://127.0.0.1:8178"
ACCELERATOR="${WHISPER_ACCELERATOR:-}"
SYCL_DEVICE="${WHISPER_SYCL_DEVICE:-0}"
SYCL_EXPECTED_DEVICE="${WHISPER_SYCL_EXPECTED_DEVICE:-}"
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
    ok "config: ${CFG} (model=${MODEL_NAME}, backend=${BACKEND}, build=${WHISPER_RESOLVED_BUILD_DIR})"
fi

if [[ "${ACCELERATOR,,}" == "sycl" ]]; then
    SYCL_DEVICE_TAG="[level_zero:gpu:${SYCL_DEVICE}]"
    SYCL_DISCOVERY_TAG="[level_zero:${SYCL_DEVICE}]"
    SYCL_DEVICE_LINE="$(sycl-ls --ignore-device-selectors 2>/dev/null | grep -F '[level_zero:gpu]' | grep -F "${SYCL_DISCOVERY_TAG}" | grep -F 'Intel' | head -1 || true)"
    SYCL_DEVICE_LINE_NORMALIZED="${SYCL_DEVICE_LINE//\(R\)/}"
    SYCL_DEVICE_LINE_NORMALIZED="${SYCL_DEVICE_LINE_NORMALIZED//\(TM\)/}"
    if [[ -z "${SYCL_DEVICE_LINE}" ]]; then
        bad "Intel Level Zero SYCL device ${SYCL_DEVICE} unavailable"
    elif [[ -n "${SYCL_EXPECTED_DEVICE}" && "${SYCL_DEVICE_LINE_NORMALIZED}" != *"${SYCL_EXPECTED_DEVICE}"* ]]; then
        bad "expected SYCL device name not found at ${SYCL_DEVICE_TAG}: ${SYCL_EXPECTED_DEVICE}"
    else
        ok "SYCL Level Zero: ${SYCL_DEVICE_LINE}"
    fi
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
SERVER_SERVICE_ACTIVE=0
if systemctl --user is-active whisper-dictation-server.service &>/dev/null; then
    SERVER_SERVICE_ACTIVE=1
    ok "systemd: whisper-dictation-server.service active"
elif [[ "${BACKEND}" == "server" ]]; then
    bad "whisper-dictation-server inactive while backend=server"
fi
if [[ "${BACKEND}" == "server" && "${SERVER_SERVICE_ACTIVE}" -eq 1 ]]; then
    SERVER_MAIN_PID="$(systemctl --user show whisper-dictation-server.service --property MainPID --value 2>/dev/null || true)"
    RUNNING_SERVER_MATCH=0
    if [[ "${SERVER_MAIN_PID}" =~ ^[1-9][0-9]*$ ]]; then
        while read -r child_pid; do
            [[ -n "${child_pid}" ]] || continue
            if [[ "$(readlink -f "/proc/${child_pid}/exe" 2>/dev/null || true)" == "${SERVER}" ]]; then
                RUNNING_SERVER_MATCH=1
                break
            fi
        done < <(pgrep -P "${SERVER_MAIN_PID}" 2>/dev/null || true)
    fi
    if [[ "${RUNNING_SERVER_MATCH}" -eq 1 ]]; then
        ok "running server matches selected binary: ${SERVER}"
    else
        bad "running server does not match selected binary: ${SERVER}"
    fi
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
    CLI_LOG="$(mktemp --tmpdir whisper-dictation-check.XXXXXX.log)"
    trap 'rm -f "${CLI_LOG:-}"' EXIT
    # Keep diagnostic prints enabled so the selected backend/device can be proved.
    # They go to stderr; stdout remains the transcript consumed below.
    OUT=$("${CLI}" -m "${MODEL}" -f "${ROOT}/samples/jfk.wav" -nt -t 4 -l en -sns 2>"${CLI_LOG}" | tail -1)
    [[ -n "${OUT}" ]] && ok "cli transcription: ${OUT:0:60}..." || bad "transcription returned empty"
    if [[ "${ACCELERATOR,,}" == "sycl" ]]; then
        SYCL_EXPECTED_MATCH=1
        if [[ -n "${SYCL_EXPECTED_DEVICE}" ]] \
            && ! sed 's/(R)//g; s/(TM)//g' "${CLI_LOG}" | grep -Fq "${SYCL_EXPECTED_DEVICE}"; then
            SYCL_EXPECTED_MATCH=0
        fi
        if grep -Fq "whisper_backend_init_gpu: using SYCL0 backend" "${CLI_LOG}" \
            && grep -Fq "${SYCL_DEVICE_TAG}" "${CLI_LOG}" \
            && [[ "${SYCL_EXPECTED_MATCH}" -eq 1 ]]; then
            ok "selected binary reports SYCL0 on ${SYCL_DEVICE_TAG}"
        else
            bad "selected binary did not report SYCL0 on ${SYCL_DEVICE_TAG}"
            sed -n '/whisper_backend_init_gpu\|level_zero\|Intel.*Graphics/p' "${CLI_LOG}" | sed 's/^/         /'
        fi
    elif [[ "${ACCELERATOR,,}" == "cuda" ]]; then
        if grep -Fq "whisper_backend_init_gpu: using CUDA0 backend" "${CLI_LOG}" \
            && grep -Eq 'ggml_cuda_init: found [1-9][0-9]* CUDA devices?' "${CLI_LOG}"; then
            ok "selected binary reports CUDA0"
        else
            bad "selected binary did not report CUDA0"
            sed -n '/whisper_backend_init_gpu\|ggml_cuda_init/p' "${CLI_LOG}" | sed 's/^/         /'
        fi
    fi
fi
if [[ "${BACKEND}" == "server" ]]; then
    if OUT=$(curl -sS --connect-timeout 2 --max-time 60 \
        "${SERVER_URL}/inference" \
        -F "file=@${ROOT}/samples/jfk.wav" \
        -F "temperature=0.0" \
        -F "response_format=json" 2>/dev/null); then
        if printf '%s' "${OUT}" | python3 "${SCRIPT_DIR}/validate-server-response.py"; then
            ok "server transcription reachable at ${SERVER_URL}"
        else
            bad "server returned invalid or empty JSON response at ${SERVER_URL}"
        fi
    else
        bad "server inference failed at ${SERVER_URL}"
    fi
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
    echo "All checks passed. Ctrl+Space to start, Ctrl+Space again to stop and paste."
else
    echo "Fix failures above, then: systemctl --user restart whisper-dictation"
    exit 1
fi
