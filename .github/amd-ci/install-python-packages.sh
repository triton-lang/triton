#!/usr/bin/env bash

set -euxo pipefail

if [ "${ROCM_RELEASE_TYPE}" = "pre-therock" ]; then
    python3 -m pip install --no-cache-dir --find-links "${PYTORCH_INDEX_URL}" "torch==${PYTORCH_VERSION}"
else
    if [ "${PYTORCH_GPU_TARGETS}" = "all" ]; then
        pytorch_extras="device-all"
    else
        IFS=',' read -r -a gpu_targets <<< "${PYTORCH_GPU_TARGETS}"
        device_extras=()
        for gpu_target in "${gpu_targets[@]}"; do
            # Reject malformed target names before turning them into pip
            # extras, which would otherwise fail with a less direct error.
            if [[ ! "${gpu_target}" =~ ^gfx[0-9][[:alnum:]-]*$ ]]; then
                echo "Invalid PyTorch GPU target: ${gpu_target}" >&2
                exit 1
            fi
            device_extras+=("device-${gpu_target}")
        done
        # Join the array into pip's comma-separated extras syntax.
        printf -v pytorch_extras '%s,' "${device_extras[@]}"
        pytorch_extras="${pytorch_extras%,}"
    fi
    python3 -m pip install --no-cache-dir --index-url "${PYTORCH_INDEX_URL}" "torch[${pytorch_extras}]==${PYTORCH_VERSION}"

    # TheRock places ROCm runtime libraries in a Python package. Some nightly
    # packages contain only versioned files even though Triton and Proton load
    # these libraries by their unversioned development names.
    # This should be simplified after https://github.com/triton-lang/triton/pull/11028
    rocm_sdk_lib="$(
        python3 -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib"))'
    )"
    test -d "${rocm_sdk_lib}"
    required_sdk_libraries=(libamdhip64 librocprofiler-sdk)
    optional_sdk_libraries=(librocprofiler-sdk-attach)
    for base in "${required_sdk_libraries[@]}" "${optional_sdk_libraries[@]}"; do
        if [ ! -e "${rocm_sdk_lib}/${base}.so" ]; then
            versioned="$(
                find "${rocm_sdk_lib}" -maxdepth 1 -name "${base}.so.*" -printf '%f\n' |
                    sort -V |
                    head -1
            )"
            if [ -z "${versioned}" ]; then
                if [[ " ${optional_sdk_libraries[*]} " == *" ${base} "* ]]; then
                    continue
                fi
                echo "Could not find ${base}.so or a versioned equivalent in ${rocm_sdk_lib}" >&2
                exit 1
            fi
            ln -s "${versioned}" "${rocm_sdk_lib}/${base}.so"
        fi
    done

    # Make the wheel-provided SDK libraries available to components that use
    # the system dynamic-loader search path rather than rocm_sdk preloading.
    echo "${rocm_sdk_lib}" > /etc/ld.so.conf.d/rocm-sdk.conf
    ldconfig

fi

# Uninstall any triton that is installed by the base image.
python3 -m pip uninstall -y triton pytorch-triton pytorch-triton-rocm || true

python3 -m pip install --no-cache-dir "hip-python==${HIP_PYTHON_VERSION}"

# Proton currently accepts a directory override and then loads an unversioned
# library name from it. Give every image one stable directory regardless of
# whether ROCm libraries came from the legacy APT layout or TheRock packages.
if [ "${ROCM_RELEASE_TYPE}" = "pre-therock" ]; then
    rocprofiler_source_lib="/opt/rocm/lib"
else
    rocprofiler_source_lib="${rocm_sdk_lib}"
fi
mkdir -p "${TRITON_ROCPROFILER_SDK_LIB_PATH}"
for base in librocprofiler-sdk librocprofiler-sdk-attach; do
    if [ -e "${rocprofiler_source_lib}/${base}.so" ]; then
        ln -s "${rocprofiler_source_lib}/${base}.so" \
            "${TRITON_ROCPROFILER_SDK_LIB_PATH}/${base}.so"
    fi
done
test -e "${TRITON_ROCPROFILER_SDK_LIB_PATH}/librocprofiler-sdk.so"
env -u ROCPROFILER_REGISTER_FORCE_LOAD python3 -c \
    'import ctypes, os, torch; ctypes.CDLL(os.path.join(os.environ["TRITON_ROCPROFILER_SDK_LIB_PATH"], "librocprofiler-sdk.so")); assert torch.version.hip'

# Confirm the image does not retain a packaged Triton, which would shadow
# the source checkout installed by CI.
! python3 -c 'import triton'
