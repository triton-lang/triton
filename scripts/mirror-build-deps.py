#!/usr/bin/env python3
"""Mirror Triton's LLVM, JSON and NVIDIA build archives for Packrat.

Run on a machine with public download access, curl, and an Azure CLI login with
read/write access to oaiartifacts/wheels. No secure-cluster proxy changes are needed.
LLVM is verified against llvm-info.json. JSON and NVIDIA do not have pinned
checksums in the build metadata: their hashes are recorded after a successful
public download and used to verify caches. Existing destination archives are
assumed correct and skipped without downloading or verifying them; existing blobs
are never overwritten. Downloads persist in ~/.cache/triton/llvm-mirror (or
--download-dir) and resume on the next invocation after interruption. Verified
public archives are reused.
Cache files remain after successful uploads; remove them manually when unneeded.
Python build requirements from pip are handled by the configured package index.

Examples:
  python scripts/mirror-build-deps.py --dry-run
  python scripts/mirror-build-deps.py --dependency json
  python scripts/mirror-build-deps.py --platform ubuntu-arm64 ubuntu-x64
  python scripts/mirror-build-deps.py --dependency llvm --platform ubuntu-arm64
  python scripts/mirror-build-deps.py --download-dir /data/llvm-archives
  python scripts/mirror-build-deps.py  # All dependency groups and platforms
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path


def az_blob(operation, *args, output="none"):
    return subprocess.check_output([
        "az", "storage", "blob", operation, "--account-name", "oaiartifacts", "--auth-mode", "login",
        "--container-name", "wheels", "--only-show-errors", "--output", output, *args
    ], text=True).strip()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path, expected_sha256):
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Checksum mismatch for {path.name}: expected {expected_sha256}, got {actual_sha256}. "
                           f"The destination has not been modified. Cached bytes are preserved at {path}.")


def download_public_archive(dependency, archive):
    receipt = archive.with_name(archive.name + ".sha256")
    expected = dependency.sha256sum
    if expected is None and receipt.exists():
        expected = receipt.read_text().strip()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise RuntimeError(f"Invalid checksum receipt: {receipt}")
    if archive.exists() and expected is not None and file_sha256(archive) == expected:
        print("  Reusing verified cached archive.", flush=True)
        return expected
    if archive.exists() and receipt.exists():
        # A receipt marks a completed download; do not append to a corrupt cache.
        verify_sha256(archive, expected)

    archive.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl", "--fail", "--location", "--show-error", "--retry", "3", "--proto", "=https", "--proto-redir", "=https",
        "--continue-at", "-", "--output",
        str(archive), "--url", dependency.source_url
    ]
    hostname = urllib.parse.urlsplit(dependency.source_url).hostname
    if hostname is not None and hostname.endswith(".blob.core.windows.net"):
        # Anonymous Azure Blob requests need this API version to honor Range.
        command.extend(["--header", "x-ms-version: 2011-08-18"])
    subprocess.run(command, check=True)
    actual = file_sha256(archive)
    if expected is not None and actual != expected:
        verify_sha256(archive, expected)
    # Publish completion only after curl succeeded and any pinned checksum passed.
    pending_receipt = receipt.with_name(receipt.name + ".tmp")
    pending_receipt.write_text(actual + "\n")
    pending_receipt.replace(receipt)
    return actual


def mirror_archive(dependency, download_dir, dry_run=False):
    blob_name = f"triton_wheel/{dependency.mirror_path}"
    destination = f"az://oaiartifacts/wheels/{blob_name}"
    expected = dependency.sha256sum
    print(f"{dependency.source_url}\n  -> {destination}\n  SHA256: {expected or 'recorded after public download'}",
          flush=True)
    # Preserve the previous LLVM cache layout so existing downloads can resume.
    relative_path = dependency.filename if dependency.mirror_path.startswith("llvm/") else dependency.mirror_path
    archive = download_dir / relative_path
    print(f"  Cache: {archive}", flush=True)
    if dry_run:
        return

    exists = az_blob("exists", "--name", blob_name, "--query", "exists", output="tsv")
    if exists not in {"true", "false"}:
        raise RuntimeError(f"Unexpected Azure blob existence response: {exists!r}")
    if exists == "true":
        print("  Already mirrored; assuming correct and skipping.", flush=True)
        return

    download_public_archive(dependency, archive)
    # Refuse to overwrite even if another uploader creates the blob meanwhile.
    az_blob("upload", "--name", blob_name, "--file", str(archive), "--overwrite", "false", "--validate-content")
    print("  Uploaded verified archive.", flush=True)


def get_dependencies(llvm_info, platforms, groups):
    # Use the same definitions as the build without importing native Triton.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from build_helpers import get_json_archive, get_llvm_archive, get_nvidia_toolchain_packages

    archives = []
    if "json" in groups:
        archives.append(get_json_archive())
    if "llvm" in groups:
        archives.extend(get_llvm_archive(platform, llvm_info) for platform in platforms)
    if "nvidia" in groups:
        arches = {
            "sbsa" if platform.endswith("-arm64") else "x86_64"
            for platform in platforms
            if platform.startswith(("ubuntu-", "almalinux-", "macos-"))
        }
        for arch in sorted(arches):
            archives.extend(package.archive("linux", arch) for package in get_nvidia_toolchain_packages())
    # CUPTI headers/libraries and multiple LLVM distributions can share an archive.
    return list({archive.mirror_path: archive for archive in archives}.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llvm-info", type=Path, default=Path(__file__).resolve().parents[1] / "cmake/llvm-info.json",
                        help="Pinned LLVM metadata (default: this checkout's cmake/llvm-info.json)")
    parser.add_argument("--platform", nargs="+",
                        help="Platform suffixes to copy (default: all platforms in the metadata)")
    parser.add_argument("--dependency", nargs="+", choices=["llvm", "json", "nvidia"],
                        default=["llvm", "json", "nvidia"], help="Dependency groups to mirror (default: all)")
    parser.add_argument("--download-dir", type=Path, default=Path.home() / ".cache/triton/llvm-mirror",
                        help="Persistent download cache (default: ~/.cache/triton/llvm-mirror)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the copy plan without network requests or credentials")
    args = parser.parse_args(argv)
    download_dir = args.download_dir.expanduser().resolve()

    with args.llvm_info.open() as file:
        llvm_info = json.load(file)
    platforms = list(dict.fromkeys(args.platform or llvm_info["sha256sum"]))
    unknown = set(platforms) - llvm_info["sha256sum"].keys()
    if unknown:
        parser.error(f"Unknown platforms: {', '.join(sorted(unknown))}. "
                     f"Available: {', '.join(llvm_info['sha256sum'])}")
    if not args.dry_run:
        for command in ("az", "curl"):
            if shutil.which(command) is None:
                parser.error(f"Required command not found: {command}")

    for dependency in get_dependencies(llvm_info, platforms, args.dependency):
        mirror_archive(dependency, download_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
