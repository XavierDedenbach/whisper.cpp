#!/usr/bin/env bash
# Configure and build side-by-side NVIDIA CUDA dictation binaries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${ROOT}/build-cuda"

if [[ "${1:-}" == "--build-dir" ]]; then
    [[ -n "${2:-}" ]] || { echo "--build-dir requires a path" >&2; exit 2; }
    BUILD_DIR="${2}"
    shift 2
fi
[[ "$#" -eq 0 ]] || { echo "usage: $0 [--build-dir PATH]" >&2; exit 2; }

if ! command -v nvcc >/dev/null; then
    echo "CUDA compiler not found; install the NVIDIA CUDA toolkit first" >&2
    exit 1
fi

cmake_args=(
    -S "${ROOT}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_CUDA=ON
    -DWHISPER_SDL2=OFF
)
if [[ -n "${WHISPER_CUDA_ARCHITECTURES:-}" ]]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${WHISPER_CUDA_ARCHITECTURES}")
fi

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" -j"${WHISPER_BUILD_JOBS:-$(nproc)}" \
    --target whisper-cli whisper-server

echo "CUDA binaries built in ${BUILD_DIR}/bin"
echo "Verify with: WHISPER_BUILD_DIR=\"${BUILD_DIR}\" WHISPER_ACCELERATOR=cuda bash ${SCRIPT_DIR}/check.sh"
