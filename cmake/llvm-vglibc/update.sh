#!/bin/bash

# Script to automatically update LLVM glibc version requirements
# This script downloads LLVM pre-compiled builds and extracts their glibc dependencies

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAKE_DIR="$(dirname "$SCRIPT_DIR")"
TEMP_DIR=$(mktemp -d)

# Cleanup function
cleanup() {
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Read LLVM hash
LLVM_HASH_FILE="${CMAKE_DIR}/llvm-hash.txt"
if [[ ! -f "$LLVM_HASH_FILE" ]]; then
    echo "Error: llvm-hash.txt not found at $LLVM_HASH_FILE"
    exit 1
fi

# Read first 8 characters of the hash
LLVM_REV=$(head -c 8 "$LLVM_HASH_FILE")
echo "LLVM revision: $LLVM_REV"

# Base URL for LLVM builds
BASE_URL="https://oaitriton.blob.core.windows.net/public/llvm-builds"

# List of system suffixes to check (Linux x64 only)
SYSTEM_SUFFIXES=(
    "ubuntu-x64"
    "almalinux-x64"
)

echo "Downloading and analyzing LLVM builds..."
echo "========================================="

# Function to get glibc version from a binary
get_glibc_version() {
    local binary_path="$1"

    if [[ ! -f "$binary_path" ]]; then
        echo "Error: Binary not found at $binary_path" >&2
        return 1
    fi

    # Extract GLIBC versions using objdump
    local versions=$(objdump -T "$binary_path" 2>/dev/null | grep GLIBC_ | sed 's/.*GLIBC_\([.0-9]*\).*/\1/g' | sort -V -u)

    if [[ -z "$versions" ]]; then
        echo "Error: No GLIBC versions found in $binary_path" >&2
        return 1
    fi

    # Get the highest (last) version
    local max_version=$(echo "$versions" | tail -n 1)
    echo "$max_version"
}

# Process each system suffix
for suffix in "${SYSTEM_SUFFIXES[@]}"; do
    echo ""
    echo "Processing: $suffix"
    echo "-------------------"

    LLVM_NAME="llvm-${LLVM_REV}-${suffix}"
    LLVM_URL="${BASE_URL}/${LLVM_NAME}.tar.gz"
    DOWNLOAD_PATH="${TEMP_DIR}/${LLVM_NAME}.tar.gz"
    EXTRACT_PATH="${TEMP_DIR}/${LLVM_NAME}"

    echo "Downloading from: $LLVM_URL"

    # Download the archive
    if ! curl -f -L -o "$DOWNLOAD_PATH" "$LLVM_URL" 2>/dev/null; then
        echo "Warning: Failed to download $LLVM_URL"
        echo "Skipping $suffix"
        continue
    fi

    echo "Download complete. Extracting bin/lld..."

    # Extract only bin/lld from the archive
    mkdir -p "$EXTRACT_PATH"
    if ! tar -xzf "$DOWNLOAD_PATH" -C "$EXTRACT_PATH" --wildcards '*/bin/lld' 2>/dev/null; then
        echo "Warning: Failed to extract bin/lld from $DOWNLOAD_PATH"
        echo "Skipping $suffix"
        continue
    fi

    echo "Extraction complete. Analyzing binary..."

    # Find the extracted lld binary
    BINARY_TO_CHECK=$(find "$EXTRACT_PATH" -type f -name "lld" 2>/dev/null | head -n 1)

    if [[ -z "$BINARY_TO_CHECK" ]]; then
        echo "Warning: bin/lld not found in $LLVM_NAME"
        echo "Skipping $suffix"
        continue
    fi

    echo "Checking binary: $(basename "$BINARY_TO_CHECK")"

    # Get GLIBC version
    GLIBC_VERSION=$(get_glibc_version "$BINARY_TO_CHECK")

    if [[ $? -ne 0 ]] || [[ -z "$GLIBC_VERSION" ]]; then
        echo "Warning: Failed to determine GLIBC version for $suffix"
        echo "Skipping $suffix"
        continue
    fi

    echo "Detected minimum GLIBC version: $GLIBC_VERSION"

    # Write to file
    OUTPUT_FILE="${SCRIPT_DIR}/${suffix}.txt"
    echo "$GLIBC_VERSION" > "$OUTPUT_FILE"
    echo "Updated: $OUTPUT_FILE"
done

echo ""
echo "========================================="
echo "Update complete!"
echo ""
echo "Current LLVM glibc version requirements:"
for suffix in "${SYSTEM_SUFFIXES[@]}"; do
    VERSION_FILE="${SCRIPT_DIR}/${suffix}.txt"
    if [[ -f "$VERSION_FILE" ]]; then
        VERSION=$(cat "$VERSION_FILE")
        printf "  %-20s: %s\n" "$suffix" "$VERSION"
    else
        printf "  %-20s: (not set)\n" "$suffix"
    fi
done
