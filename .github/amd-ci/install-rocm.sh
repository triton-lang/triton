#!/usr/bin/env bash

set -euxo pipefail

# TheRock apt package names and versioned /opt/rocm/core directory use only
# the ROCm major/minor version, while ROCM_VERSION retains the full release
# or nightly version. Extract the required package and directory suffix.
if [[ "${ROCM_VERSION}" =~ ^([0-9]+\.[0-9]+) ]]; then
    rocm_major_minor="${BASH_REMATCH[1]}"
else
    echo "ROCM_VERSION must begin with a major.minor version: ${ROCM_VERSION}" >&2
    exit 1
fi

case "${ROCM_RELEASE_TYPE}" in
    pre-therock)
        rocm_repo_url="https://repo.radeon.com/rocm/apt/${ROCM_REPO_DIRECTORY}"
        rocm_gpg_url="https://repo.radeon.com/rocm/rocm.gpg.key"
        rocm_distribution="noble"
        rocm_pin_origin="repo.radeon.com"
        rocm_packages=(
            rocm-dev
            rocm-libs
        )
        ;;
    nightlies)
        # Nightly repository directories are named DATE-BUILD_ID. Validate
        # that convention here so a bad argument reports clearly instead
        # of only producing a failed repository download.
        if [[ ! "${ROCM_REPO_DIRECTORY}" =~ ^[0-9]{8}-[0-9]+$ ]]; then
            echo "Invalid nightly ROCm repository directory: ${ROCM_REPO_DIRECTORY}" >&2
            exit 1
        fi
        rocm_repo_url="https://rocm.nightlies.amd.com/packages-multi-arch/deb/${ROCM_REPO_DIRECTORY}"
        rocm_source_options="arch=amd64 trusted=yes"
        rocm_distribution="stable"
        rocm_packages=(
            "amdrocm${rocm_major_minor}"
            "amdrocm-core-sdk${rocm_major_minor}"
        )
        ;;
    prereleases)
        rocm_repo_url="https://rocm.prereleases.amd.com/packages-multi-arch/ubuntu2404"
        rocm_gpg_url="https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg"
        rocm_distribution="stable"
        rocm_packages=(
            "amdrocm${rocm_major_minor}"
            "amdrocm-core-sdk${rocm_major_minor}"
        )
        ;;
    stable)
        rocm_repo_url="https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404"
        rocm_gpg_url="https://repo.amd.com/rocm/packages/gpg/rocm.gpg"
        rocm_distribution="stable"
        rocm_packages=(
            "amdrocm${rocm_major_minor}"
            "amdrocm-core-sdk${rocm_major_minor}"
        )
        ;;
    *)
        echo "Unsupported ROCM_RELEASE_TYPE: ${ROCM_RELEASE_TYPE}" >&2
        exit 1
        ;;
esac

# Signed release repositories provide a key URL. Nightly repositories use
# trusted=yes instead, so they intentionally skip key installation.
if [ -n "${rocm_gpg_url:-}" ]; then
    mkdir -p /etc/apt/keyrings
    curl -fsSL --connect-timeout 30 --retry 3 --retry-delay 5 "${rocm_gpg_url}" |
        gpg --dearmor -o /etc/apt/keyrings/amdrocm.gpg
    rocm_source_options="arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg"
fi

echo "deb [${rocm_source_options}] ${rocm_repo_url} ${rocm_distribution} main" > /etc/apt/sources.list.d/rocm.list

# Ubuntu also packages some ROCm components with versions that sort higher than
# the pre-TheRock repository versions. Prefer one coherent repository so exact
# dependencies from rocm-dev do not resolve to incompatible Ubuntu packages.
if [ -n "${rocm_pin_origin:-}" ]; then
    printf '%s\n' \
        'Package: *' \
        "Pin: release o=${rocm_pin_origin}" \
        'Pin-Priority: 600' \
        > /etc/apt/preferences.d/rocm
fi

apt-get update
apt-get install -y --no-install-recommends "${rocm_packages[@]}"
rm -rf /var/lib/apt/lists/*

# TheRock packages install into a versioned core directory. Replace the
# complete set of conventional paths together so /opt/rocm cannot combine
# directories from different ROCm installations.
if [ "${ROCM_RELEASE_TYPE}" != "pre-therock" ]; then
    core_dir="/opt/rocm/core-${rocm_major_minor}"
    core_subdirs=(bin lib include libexec share)

    for subdir in "${core_subdirs[@]}"; do
        test -d "${core_dir}/${subdir}"
    done
    for subdir in "${core_subdirs[@]}"; do
        rm -rf "/opt/rocm/${subdir}"
        ln -s "${core_dir}/${subdir}" "/opt/rocm/${subdir}"
    done
fi

# Check the SDK layout expected by later Triton builds.
test -d /opt/rocm/bin
test -d /opt/rocm/lib
test -d /opt/rocm/include
test -x /opt/rocm/bin/hipcc

# Verify the ROCm compiler driver is available through PATH
command -v hipcc
