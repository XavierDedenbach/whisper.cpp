#!/usr/bin/env bash
# Shared build/runtime selection for dictation launchers and checks.

whisper_dictation_load_runtime() {
    local script_dir install_env config root build_spec accelerator cache setvars
    local restore_nounset=0
    local override_root_set="${WHISPER_REPO_ROOT+x}" override_root="${WHISPER_REPO_ROOT-}"
    local override_build_set="${WHISPER_BUILD_DIR+x}" override_build="${WHISPER_BUILD_DIR-}"
    local override_accel_set="${WHISPER_ACCELERATOR+x}" override_accel="${WHISPER_ACCELERATOR-}"
    local override_setvars_set="${WHISPER_ONEAPI_SETVARS+x}" override_setvars="${WHISPER_ONEAPI_SETVARS-}"
    local override_selector_set="${WHISPER_ONEAPI_DEVICE_SELECTOR+x}" override_selector="${WHISPER_ONEAPI_DEVICE_SELECTOR-}"
    local override_device_set="${WHISPER_SYCL_DEVICE+x}" override_device="${WHISPER_SYCL_DEVICE-}"
    local override_expected_set="${WHISPER_SYCL_EXPECTED_DEVICE+x}" override_expected="${WHISPER_SYCL_EXPECTED_DEVICE-}"

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    install_env="${HOME}/.config/whisper-dictation/install.env"
    config="${HOME}/.config/whisper-dictation/config.env"

    if [[ -f "${install_env}" ]]; then
        # shellcheck disable=SC1090
        source "${install_env}"
    fi
    if [[ -f "${config}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${config}"
        set +a
    fi

    [[ -n "${override_root_set}" ]] && WHISPER_REPO_ROOT="${override_root}"
    [[ -n "${override_build_set}" ]] && WHISPER_BUILD_DIR="${override_build}"
    [[ -n "${override_accel_set}" ]] && WHISPER_ACCELERATOR="${override_accel}"
    [[ -n "${override_setvars_set}" ]] && WHISPER_ONEAPI_SETVARS="${override_setvars}"
    [[ -n "${override_selector_set}" ]] && WHISPER_ONEAPI_DEVICE_SELECTOR="${override_selector}"
    [[ -n "${override_device_set}" ]] && WHISPER_SYCL_DEVICE="${override_device}"
    [[ -n "${override_expected_set}" ]] && WHISPER_SYCL_EXPECTED_DEVICE="${override_expected}"

    root="${WHISPER_REPO_ROOT:-}"
    if [[ -z "${root}" ]]; then
        root="$(cd "${script_dir}/../.." && pwd)"
    fi
    build_spec="${WHISPER_BUILD_DIR:-build}"
    if [[ "${build_spec}" = /* ]]; then
        WHISPER_RESOLVED_BUILD_DIR="${build_spec}"
    else
        WHISPER_RESOLVED_BUILD_DIR="${root}/${build_spec}"
    fi

    export WHISPER_REPO_ROOT="${root}"
    export WHISPER_RESOLVED_BUILD_DIR
    export WHISPER_BIN_DIR="${WHISPER_RESOLVED_BUILD_DIR}/bin"

    accelerator="${WHISPER_ACCELERATOR:-auto}"
    if [[ "${accelerator,,}" == "auto" ]]; then
        cache="${WHISPER_RESOLVED_BUILD_DIR}/CMakeCache.txt"
        if [[ "${build_spec,,}" == *sycl* ]] \
            || grep -q '^GGML_SYCL:BOOL=ON$' "${cache}" 2>/dev/null; then
            accelerator="sycl"
        elif [[ "${build_spec,,}" == *cuda* ]] \
            || grep -q '^GGML_CUDA:BOOL=ON$' "${cache}" 2>/dev/null; then
            accelerator="cuda"
        else
            accelerator=""
        fi
    fi
    export WHISPER_ACCELERATOR="${accelerator}"

    if [[ "${accelerator,,}" != "sycl" ]]; then
        return 0
    fi

    setvars="${WHISPER_ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
    if [[ ! -f "${setvars}" ]]; then
        echo "whisper-dictation: missing oneAPI environment script ${setvars}" >&2
        return 1
    fi

    case "$-" in
        *u*) restore_nounset=1; set +u ;;
    esac
    # shellcheck disable=SC1090
    if ! source "${setvars}" >/dev/null 2>&1; then
        [[ "${restore_nounset}" -eq 1 ]] && set -u
        echo "whisper-dictation: failed to load oneAPI environment from ${setvars}" >&2
        return 1
    fi
    [[ "${restore_nounset}" -eq 1 ]] && set -u

    export ONEAPI_DEVICE_SELECTOR="${WHISPER_ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
    export GGML_SYCL_DEVICE="${WHISPER_SYCL_DEVICE:-0}"
}
