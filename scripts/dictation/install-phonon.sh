#!/usr/bin/env bash
# Install only the selected inference runtime; machine config/services are untouched.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/runtime-env.sh"
whisper_dictation_load_runtime
device="${1:-${PHONON_DEVICE:-cpu}}"
case "${device}" in
    cpu)
        py="${PHONON_PYTHON:-python3.12}"
        runtime="${PHONON_VENV}"
        if [[ ! -x "${runtime}/bin/python" ]]; then
            "${py}" -m venv "${runtime}"
        fi
        if command -v uv >/dev/null; then
            uv pip install --python "${runtime}/bin/python" 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cpu
            uv pip install --python "${runtime}/bin/python" 'fermion-research==0.2.11'
        else
            "${runtime}/bin/python" -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cpu
            "${runtime}/bin/python" -m pip install 'fermion-research==0.2.11'
        fi
        "${runtime}/bin/fermion" describe
        ;;
    cuda)
        command -v docker >/dev/null || { echo "CUDA requires Docker and NVIDIA Container Toolkit" >&2; exit 1; }
        # Build natively on the target host. No x86 image or CPU fallback on Spark.
        docker build --pull -f "${SCRIPT_DIR}/phonon-cuda.Dockerfile" \
            -t "${PHONON_CUDA_IMAGE}" "${SCRIPT_DIR}"
        docker run --rm --gpus all --entrypoint python "${PHONON_CUDA_IMAGE}" \
            -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
        ;;
    *) echo "Usage: $0 [cpu|cuda]" >&2; exit 2 ;;
esac
