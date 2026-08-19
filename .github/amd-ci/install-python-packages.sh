#!/usr/bin/env bash

set -euxo pipefail

if [ "${ROCM_RELEASE_TYPE}" = "pre-therock" ]; then
    python3 -m pip install --no-cache-dir --find-links "${PYTORCH_INDEX_URL}" "torch==${PYTORCH_VERSION}"
else
    if [ "${PYTORCH_GPU_TARGETS}" = "all" ]; then
        device_extras=(device-all)
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
    fi

    # Install one coherent TheRock distribution. PyTorch and ROCm use the same
    # device extras and exact ROCm version, while the devel extra provides the
    # headers and tools needed to build Triton.
    printf -v rocm_extras '%s,' libraries devel "${device_extras[@]}"
    rocm_extras="${rocm_extras%,}"
    printf -v pytorch_extras '%s,' "${device_extras[@]}"
    pytorch_extras="${pytorch_extras%,}"
    python3 -m pip install --no-cache-dir --index-url "${PYTORCH_INDEX_URL}" \
        "rocm[${rocm_extras}]==${ROCM_VERSION}" \
        "torch[${pytorch_extras}]==${PYTORCH_VERSION}"

    # Initialize the expanded devel tree and verify the public TheRock
    # interfaces that Triton's build and runtime discovery use.
    rocm_sdk_root="$(python3 -I "$(command -v rocm-sdk)" path --root)"
    test -d "${rocm_sdk_root}/include"
    test -f "${rocm_sdk_root}/include/hip/hip_runtime.h"
    test -f "${rocm_sdk_root}/include/rocprofiler-sdk/fwd.h"
    test -x "${rocm_sdk_root}/bin/hipcc"
    ln -s "${rocm_sdk_root}/bin/hipcc" /usr/local/bin/hipcc
    python3 - <<'PY'
import importlib.metadata
import os
from pathlib import Path

import rocm_sdk

expected = os.environ["ROCM_VERSION"]
for distribution in importlib.metadata.distributions():
    name = distribution.metadata["Name"]
    if name == "rocm" or name.startswith("rocm-sdk-"):
        if distribution.version != expected:
            raise SystemExit(
                f"mixed TheRock versions: {name}=={distribution.version}, "
                f"expected {expected}"
            )

libraries = {
    name: Path(rocm_sdk.find_libraries(name)[0]).resolve()
    for name in (
        "amdhip64",
        "rocprofiler-sdk",
        "rocprofiler-sdk-roctx",
        "roctracer64",
        "roctx64",
    )
}
for name, path in libraries.items():
    print(f"{name}: {path}")
PY

    # A TheRock image must contain no independently installed system ROCm.
    if dpkg-query -W 'amdrocm*' 'rocm-*' 2>/dev/null; then
        echo "TheRock image contains both wheel and APT ROCm packages" >&2
        exit 1
    fi
    if [ -e /opt/rocm ]; then
        echo "TheRock image contains both wheel and /opt/rocm installations" >&2
        exit 1
    fi

fi

# Uninstall any triton that is installed by the base image.
python3 -m pip uninstall -y triton pytorch-triton pytorch-triton-rocm || true

python3 -m pip install --no-cache-dir "hip-python==${HIP_PYTHON_VERSION}"

# Enforce the installation model in both directions: legacy images use only
# system ROCm, and TheRock images use only wheel ROCm.
if [ "${ROCM_RELEASE_TYPE}" = "pre-therock" ]; then
    python3 - <<'PY'
import importlib.metadata

unexpected = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata["Name"]
    if name == "rocm" or name.startswith("rocm-sdk-"):
        unexpected.append(f"{name}=={distribution.version}")
if unexpected:
    raise SystemExit(
        "pre-TheRock image contains wheel ROCm packages: "
        + ", ".join(sorted(unexpected))
    )
PY
else
    if dpkg-query -W 'amdrocm*' 'rocm-*' 2>/dev/null; then
        echo "TheRock image contains both wheel and APT ROCm packages" >&2
        exit 1
    fi
    if [ -e /opt/rocm ]; then
        echo "TheRock image contains both wheel and /opt/rocm installations" >&2
        exit 1
    fi
    python3 - <<'PY'
import importlib.metadata
import os

expected = os.environ["ROCM_VERSION"]
installed = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata["Name"]
    if name == "rocm" or name.startswith("rocm-sdk-"):
        installed[name] = distribution.version

if not installed:
    raise SystemExit("TheRock image contains no ROCm wheel distributions")
mismatched = {
    name: version
    for name, version in installed.items()
    if version != expected
}
if mismatched:
    raise SystemExit(
        f"mixed TheRock versions: {mismatched}; expected every package "
        f"to use {expected}"
    )
print(f"TheRock ROCm packages: {len(installed)} at version {expected}")
PY
fi

# Verify PyTorch agrees with the selected ROCm distribution. TheRock runtime
# selection itself is handled by rocm_sdk in Triton and Proton; pre-TheRock
# configurations use the single system installation under /opt/rocm.
python3 - <<'PY'
import os
import re
import torch

expected = os.environ["ROCM_VERSION"]
actual = torch.version.hip
expected_major_minor = re.match(r"^[0-9]+\.[0-9]+", expected)
actual_major_minor = re.match(r"^[0-9]+\.[0-9]+", actual or "")
if (
    expected_major_minor is None
    or actual_major_minor is None
    or expected_major_minor.group() != actual_major_minor.group()
):
    raise SystemExit(f"PyTorch reports ROCm {actual}, expected {expected}")
print(f"PyTorch ROCm: {actual}")
PY

# Confirm the image does not retain a packaged Triton, which would shadow
# the source checkout installed by CI.
! python3 -c 'import triton'
