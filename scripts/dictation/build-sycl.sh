#!/usr/bin/env bash
# Reproducibly configure and build the Intel Level Zero SYCL dictation binaries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${ROOT}/build-sycl"

if [[ "${1:-}" == "--build-dir" ]]; then
    [[ -n "${2:-}" ]] || { echo "--build-dir requires a path" >&2; exit 2; }
    BUILD_DIR="${2}"
    shift 2
fi
[[ "$#" -eq 0 ]] || { echo "usage: $0 [--build-dir PATH]" >&2; exit 2; }

SETVARS="${WHISPER_ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
[[ -f "${SETVARS}" ]] || { echo "missing oneAPI environment script: ${SETVARS}" >&2; exit 1; }

restore_nounset=0
case "$-" in
    *u*) restore_nounset=1; set +u ;;
esac
# shellcheck disable=SC1090
source "${SETVARS}" >/dev/null
[[ "${restore_nounset}" -eq 1 ]] && set -u

export ONEAPI_DEVICE_SELECTOR="${WHISPER_ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export GGML_SYCL_DEVICE="${WHISPER_SYCL_DEVICE:-0}"
EXPECTED_DEVICE="${WHISPER_SYCL_EXPECTED_DEVICE:-}"

DEVICE_TAG="[level_zero:gpu:${GGML_SYCL_DEVICE}]"
DISCOVERY_TAG="[level_zero:${GGML_SYCL_DEVICE}]"
DEVICE_LINE="$(sycl-ls --ignore-device-selectors 2>/dev/null | grep -F '[level_zero:gpu]' | grep -F "${DISCOVERY_TAG}" | grep -F 'Intel' | head -1 || true)"
DEVICE_LINE_NORMALIZED="${DEVICE_LINE//\(R\)/}"
DEVICE_LINE_NORMALIZED="${DEVICE_LINE_NORMALIZED//\(TM\)/}"
if [[ -z "${DEVICE_LINE}" ]]; then
    echo "Intel Level Zero SYCL device ${GGML_SYCL_DEVICE} was not found" >&2
    exit 1
fi
if [[ -n "${EXPECTED_DEVICE}" && "${DEVICE_LINE_NORMALIZED}" != *"${EXPECTED_DEVICE}"* ]]; then
    echo "expected SYCL device name '${EXPECTED_DEVICE}' not found at ${DEVICE_TAG}: ${DEVICE_LINE}" >&2
    exit 1
fi
echo "Using ${DEVICE_LINE}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_SUPPORT_LEVEL_ZERO=ON \
    -DGGML_SYCL_F16=OFF \
    -DGGML_BLAS=ON \
    -DWHISPER_SDL2=OFF \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx

cmake --build "${BUILD_DIR}" -j"${WHISPER_BUILD_JOBS:-$(nproc)}" \
    --target whisper-cli whisper-server

echo "SYCL binaries built in ${BUILD_DIR}/bin"
echo "Verify with: WHISPER_BUILD_DIR=\"${BUILD_DIR}\" WHISPER_ACCELERATOR=sycl bash ${SCRIPT_DIR}/check.sh"
