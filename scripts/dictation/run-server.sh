#!/usr/bin/env bash
# Launch whisper-server using ~/.config/whisper-dictation/config.env
# Always binds to loopback (127.0.0.1). WHISPER_SERVER_URL supplies the port only.
set -euo pipefail

INSTALL_ENV="${HOME}/.config/whisper-dictation/install.env"
CONFIG="${HOME}/.config/whisper-dictation/config.env"

if [[ -f "${INSTALL_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${INSTALL_ENV}"
fi

ROOT="${WHISPER_REPO_ROOT:-}"
if [[ -z "${ROOT}" ]]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

MODEL="${WHISPER_MODEL:-small.en}"
THREADS="4"
LANGUAGE="en"
SUPPRESS_NST="1"
CARRY_INITIAL_PROMPT="1"
PROMPT=""
HOST="127.0.0.1"
PORT="8178"
PY="${ROOT}/scripts/dictation/.venv/bin/python"
VOCAB_PY="${ROOT}/scripts/dictation/vocab_prompt.py"

if [[ -f "${CONFIG}" ]]; then
    # shellcheck disable=SC1090
    set -a
    # shellcheck disable=SC1090
    source "${CONFIG}"
    set +a
    MODEL="${WHISPER_MODEL:-$MODEL}"
    THREADS="${WHISPER_THREADS:-$THREADS}"
    LANGUAGE="${WHISPER_LANGUAGE:-$LANGUAGE}"
    SUPPRESS_NST="${WHISPER_SUPPRESS_NST:-$SUPPRESS_NST}"
    CARRY_INITIAL_PROMPT="${WHISPER_CARRY_INITIAL_PROMPT:-$CARRY_INITIAL_PROMPT}"
    if [[ -x "${PY}" && -f "${VOCAB_PY}" ]]; then
        PROMPT="$("${PY}" "${VOCAB_PY}")"
    else
        PROMPT="${WHISPER_PROMPT:-}"
    fi
    if [[ -n "${WHISPER_SERVER_URL:-}" ]]; then
        url="${WHISPER_SERVER_URL#http://}"
        url="${url#https://}"
        cfg_host="${url%%:*}"
        rest="${url#*:}"
        PORT="${rest%%/*}"
        if [[ "${cfg_host}" != "127.0.0.1" && "${cfg_host}" != "localhost" ]]; then
            echo "whisper-dictation-server: refusing non-loopback WHISPER_SERVER_URL host '${cfg_host}' (authless server; bind stays 127.0.0.1)" >&2
        fi
    fi
fi

BIN="${ROOT}/build/bin/whisper-server"
MODEL_PATH="${ROOT}/models/ggml-${MODEL}.bin"

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

exec "${BIN}" "${args[@]}"
