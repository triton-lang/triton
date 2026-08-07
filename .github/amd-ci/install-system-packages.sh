#!/usr/bin/env bash

set -euxo pipefail

packages=(
    build-essential
    ca-certificates
    ccache
    clang
    curl
    file
    git
    gnupg
    lld
    pkg-config
    python3-dev
    python3-pip
    python3-venv
    unzip
    xz-utils
    zip
)

apt-get update
apt-get install -y --no-install-recommends "${packages[@]}"
rm -rf /var/lib/apt/lists/*
