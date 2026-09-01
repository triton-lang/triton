#!/usr/bin/env bash

set -euxo pipefail

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
    nightlies | prereleases | stable)
        # TheRock PyTorch wheels depend on a matching wheel-based ROCm
        # runtime. install-python-packages.sh installs that runtime together
        # with its devel and device packages. Installing TheRock APT packages
        # here as well would put two ROCm distributions in the image.
        if dpkg-query -W 'amdrocm*' 'rocm-*' 2>/dev/null; then
            echo "TheRock images must not contain APT ROCm packages" >&2
            exit 1
        fi
        if [ -e /opt/rocm ]; then
            echo "TheRock images must not inherit a system ROCm installation" >&2
            exit 1
        fi
        exit 0
        ;;
    *)
        echo "Unsupported ROCM_RELEASE_TYPE: ${ROCM_RELEASE_TYPE}" >&2
        exit 1
        ;;
esac

mkdir -p /etc/apt/keyrings
curl -fsSL --connect-timeout 30 --retry 3 --retry-delay 5 "${rocm_gpg_url}" |
    gpg --dearmor -o /etc/apt/keyrings/amdrocm.gpg
rocm_source_options="arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg"

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

# Make the one system ROCm installation available without setting global
# loader variables that would interfere with TheRock configurations.
echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
echo "/opt/rocm/lib64" >> /etc/ld.so.conf.d/rocm.conf
ldconfig
ln -s /opt/rocm/bin/hipcc /usr/local/bin/hipcc

# Check the SDK layout expected by later Triton builds.
test -d /opt/rocm/bin
test -d /opt/rocm/lib
test -d /opt/rocm/include
test -x /opt/rocm/bin/hipcc

# Verify the ROCm compiler driver is available through PATH
command -v hipcc
