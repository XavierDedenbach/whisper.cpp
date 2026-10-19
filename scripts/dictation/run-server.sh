#!/usr/bin/env bash
# Launch and warm whisper-server using ~/.config/whisper-dictation/config.env.
# Always binds to loopback (127.0.0.1). WHISPER_SERVER_URL supplies the port only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runtime-env.sh"
whisper_dictation_load_runtime

ROOT="${WHISPER_REPO_ROOT}"
MODEL="${WHISPER_MODEL:-small.en}"
THREADS="${WHISPER_THREADS:-4}"
LANGUAGE="${WHISPER_LANGUAGE:-en}"
SUPPRESS_NST="${WHISPER_SUPPRESS_NST:-1}"
CARRY_INITIAL_PROMPT="${WHISPER_CARRY_INITIAL_PROMPT:-1}"
PROMPT=""
HOST="127.0.0.1"
PORT="8178"
SERVER_URL="${WHISPER_SERVER_URL:-http://127.0.0.1:8178}"
WARMUP="${WHISPER_SERVER_WARMUP:-1}"
WARMUP_AUDIO="${WHISPER_SERVER_WARMUP_AUDIO:-${ROOT}/samples/jfk.wav}"
WARMUP_TIMEOUT="${WHISPER_SERVER_WARMUP_TIMEOUT:-120}"
INFERENCE_WATCHDOG_SEC="${WHISPER_INFERENCE_WATCHDOG_SEC:-90}"
SERVER_RECYCLE_SEC="${WHISPER_SERVER_RECYCLE_SEC:-21600}"
WATCHDOG_POLL_SEC="${WHISPER_WATCHDOG_POLL_SEC:-1}"
PY="${ROOT}/scripts/dictation/.venv/bin/python"
[[ -x "${PY}" ]] || PY="python3"
ENGINE="${DICTATION_ENGINE:-whisper}"
PHONON_HELPER="${SCRIPT_DIR}/phonon_backend.py"
VOCAB_PY="${ROOT}/scripts/dictation/vocab_prompt.py"
RESPONSE_VALIDATOR="${SCRIPT_DIR}/validate-server-response.py"

if [[ -f "${VOCAB_PY}" ]]; then
    PROMPT="$("${PY}" "${VOCAB_PY}")"
else
    PROMPT="${WHISPER_PROMPT:-}"
fi

if [[ "${ENGINE}" == "phonon2" ]]; then
    SERVER_URL="$("${PY}" "${PHONON_HELPER}" url "${SERVER_URL}")"
fi
url="${SERVER_URL#http://}"
url="${url#https://}"
cfg_host="${url%%:*}"
rest="${url#*:}"
PORT="${rest%%/*}"
if [[ "${cfg_host}" != "127.0.0.1" && "${cfg_host}" != "localhost" ]]; then
    echo "whisper-dictation-server: refusing non-loopback WHISPER_SERVER_URL host '${cfg_host}' (authless server; bind stays 127.0.0.1)" >&2
fi
SERVER_URL="http://${HOST}:${PORT}"

container_dir=""
BIN="${WHISPER_BIN_DIR}/whisper-server"
MODEL_PATH="${ROOT}/models/ggml-${MODEL}.bin"

if [[ "${ENGINE}" == "phonon2" ]]; then
    case "${PHONON_DEVICE:-cpu}" in
        cpu)
            BIN="${PHONON_VENV:-${HOME}/.local/share/whisper-dictation/phonon-venv}/bin/fermion"
            args=(serve phonon-2 --host "${HOST}" --port "${PORT}")
            if [[ "${PHONON_THREADS:-0}" != "0" ]]; then
                args+=(--threads "${PHONON_THREADS}")
            fi
            [[ -x "${BIN}" ]] || { echo "Missing Phonon runtime; run install-phonon.sh cpu" >&2; exit 1; }
            ;;
        cuda)
            BIN="$(command -v docker)"
            container_dir="$(mktemp -d -t whisper-phonon-container.XXXXXX)"
            args=(run --rm --init --gpus all --stop-timeout 5
                --cidfile "${container_dir}/cid"
                -p "127.0.0.1:${PORT}:8000"
                --mount "type=volume,source=whisper-dictation-phonon2-cache,target=/home/phonon/.cache"
                "${PHONON_CUDA_IMAGE}" serve --model phonon-2 --host 0.0.0.0 --port 8000)
            ;;
    esac
else
if [[ ! -x "${BIN}" ]]; then
    echo "whisper-dictation-server: missing ${BIN}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "whisper-dictation-server: missing model ${MODEL_PATH}" >&2
    echo "  Download: bash ${ROOT}/models/download-ggml-model.sh ${MODEL}" >&2
    exit 1
fi

args=(
    -m "${MODEL_PATH}"
    -l "${LANGUAGE}"
    -t "${THREADS}"
    --host "${HOST}"
    --port "${PORT}"
)
case "${SUPPRESS_NST,,}" in
    1|true|yes|on) args+=(-sns) ;;
esac
if [[ -n "${PROMPT}" ]]; then
    args+=(--prompt "${PROMPT}")
fi
case "${CARRY_INITIAL_PROMPT,,}" in
    1|true|yes|on) args+=(--carry-initial-prompt) ;;
esac
fi

server_pid=""
warmup_response=""
stopping=0
cleanup() {
    local status=$?
    trap - EXIT
    [[ -n "${warmup_response}" ]] && rm -f "${warmup_response}"
    if [[ -n "${container_dir}" && -s "${container_dir}/cid" ]]; then
        timeout 8 docker stop --time 5 "$(<"${container_dir}/cid")" >/dev/null 2>&1 || true
    fi
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    if [[ -n "${container_dir}" ]]; then
        rm -f "${container_dir}/cid"
        rmdir "${container_dir}" 2>/dev/null || true
    fi
    exit "${status}"
}
forward_stop() {
    stopping=1
    if [[ -n "${container_dir}" && -s "${container_dir}/cid" ]]; then
        timeout 8 docker stop --time 5 "$(<"${container_dir}/cid")" >/dev/null 2>&1 || true
    fi
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
whisper_inference_sockets_busy() {
    local port="$1" out
    command -v ss >/dev/null 2>&1 || return 1
    out="$(ss -tn "sport = :${port}" 2>/dev/null || true)"
    [[ "${out}" == *ESTAB* || "${out}" == *CLOSE-WAIT* ]]
}

watch_server() {
    local poll="${WATCHDOG_POLL_SEC}"
    local busy_since=""
    while kill -0 "${server_pid}" 2>/dev/null; do
        if whisper_inference_sockets_busy "${PORT}"; then
            busy_since="${busy_since:-${SECONDS}}"
            if [[ "${INFERENCE_WATCHDOG_SEC}" != "0" ]] \
                && (( SECONDS - busy_since >= INFERENCE_WATCHDOG_SEC )); then
                echo "whisper-dictation-server: inference watchdog: sockets stuck $((SECONDS - busy_since))s on :${PORT}; killing hung server" >&2
                kill -TERM "${server_pid}" 2>/dev/null || true
                sleep 2
                kill -KILL "${server_pid}" 2>/dev/null || true
                wait "${server_pid}" 2>/dev/null || true
                server_pid=""
                exit 1
            fi
        else
            busy_since=""
            if [[ "${SERVER_RECYCLE_SEC}" != "0" ]] \
                && (( SECONDS >= SERVER_RECYCLE_SEC )); then
                echo "whisper-dictation-server: idle recycle after ${SECONDS}s uptime" >&2
                kill -TERM "${server_pid}" 2>/dev/null || true
                wait "${server_pid}" 2>/dev/null || true
                server_pid=""
                exit 1
            fi
        fi
        if [[ -n "${NOTIFY_SOCKET:-}" ]] && command -v systemd-notify >/dev/null; then
            systemd-notify WATCHDOG=1 >/dev/null 2>&1 || true
        fi
        sleep "${poll}"
    done
}

trap cleanup EXIT
trap forward_stop INT TERM

if command -v stdbuf >/dev/null; then
    stdbuf -oL -eL "${BIN}" "${args[@]}" &
else
    "${BIN}" "${args[@]}" &
fi
server_pid=$!

deadline=$((SECONDS + WARMUP_TIMEOUT))
while ! curl -fsS --noproxy '*' --connect-timeout 1 --max-time 2 "${SERVER_URL}/" >/dev/null 2>&1; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        wait "${server_pid}" || true
        echo "whisper-dictation-server: server exited before becoming reachable" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "whisper-dictation-server: timed out waiting for ${SERVER_URL}" >&2
        exit 1
    fi
    sleep 0.1
done

if [[ "${ENGINE}" == "phonon2" ]]; then
    # A Phonon service is ready only after selected-device inference succeeds.
    # WHISPER_SERVER_WARMUP remains a Whisper-only compatibility setting.
    "${PY}" "${PHONON_HELPER}" health "${SERVER_URL}" --device "${PHONON_DEVICE}"
    "${PY}" "${PHONON_HELPER}" warmup "${SERVER_URL}" --device "${PHONON_DEVICE}" \
        --audio "${WARMUP_AUDIO}" --timeout "${WARMUP_TIMEOUT}"
    echo "whisper-dictation-server: Phonon-2 startup warmup complete"
else
case "${WARMUP,,}" in
    1|true|yes|on)
        if [[ ! -f "${WARMUP_AUDIO}" ]]; then
            echo "whisper-dictation-server: warmup audio missing: ${WARMUP_AUDIO}" >&2
            exit 1
        fi
        warmup_response="$(mktemp -t whisper-dictation-warmup.XXXXXX.json)"
        warmup_args=(
            -fsS
            --connect-timeout 2
            --max-time "${WARMUP_TIMEOUT}"
            -o "${warmup_response}"
            "${SERVER_URL}/inference"
            -F "file=@${WARMUP_AUDIO}"
            -F "temperature=0.0"
            -F "response_format=json"
            -F "language=${LANGUAGE}"
        )
        if [[ -n "${PROMPT}" ]]; then
            warmup_args+=(-F "prompt=${PROMPT}")
        fi
        case "${CARRY_INITIAL_PROMPT,,}" in
            1|true|yes|on) warmup_args+=(-F "carry_initial_prompt=true") ;;
        esac
        case "${SUPPRESS_NST,,}" in
            1|true|yes|on) warmup_args+=(-F "suppress_nst=true") ;;
        esac
        if ! curl "${warmup_args[@]}" || ! python3 "${RESPONSE_VALIDATOR}" "${warmup_response}"; then
            echo "whisper-dictation-server: startup warmup failed" >&2
            exit 1
        fi
        echo "whisper-dictation-server: startup warmup complete"
        ;;
esac
fi

if [[ -n "${NOTIFY_SOCKET:-}" ]] && command -v systemd-notify >/dev/null; then
    status_text="Whisper model warm on ${WHISPER_ACCELERATOR:-cpu} (${WHISPER_RESOLVED_BUILD_DIR})"
    if [[ "${ENGINE}" == "phonon2" ]]; then
        status_text="Phonon-2 model warm on ${PHONON_DEVICE}"
    fi
    systemd-notify --ready --status="${status_text}"
fi
echo "whisper-dictation-server: ready at ${SERVER_URL}"

if [[ "${INFERENCE_WATCHDOG_SEC}" == "0" && "${SERVER_RECYCLE_SEC}" == "0" ]]; then
    if wait "${server_pid}"; then
        server_status=0
    else
        server_status=$?
    fi
else
    watch_server
    if wait "${server_pid}"; then
        server_status=0
    else
        server_status=$?
    fi
fi
server_pid=""
if [[ "${stopping}" -eq 1 ]]; then
    server_status=0
fi
exit "${server_status}"
