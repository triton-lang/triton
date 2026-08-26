#!/usr/bin/env python3
"""Build reproducible Docker images used by Triton AMD CI."""

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DOCKERFILE = SCRIPT_DIR / "Dockerfile"

DEFAULT_IMAGE_REPOSITORY = "triton-amd-ci"
IMAGE_REPOSITORY_ENV = "TRITON_AMD_CI_IMAGE_REPOSITORY"

REQUIRED_BUILD_ARGUMENTS = (
    "BASE_IMAGE",
    "ROCM_VERSION",
    "ROCM_RELEASE_TYPE",
    "ROCM_REPO_DIRECTORY",
    "PYTORCH_VERSION",
    "PYTORCH_INDEX_URL",
    "HIP_PYTHON_VERSION",
    "PYTORCH_GPU_TARGETS",
)

OPTIONAL_BUILD_ARGUMENTS = (
    "PYTORCH_DEVICE_WHEEL_URL",
    "PYTORCH_EXTRA_INDEX_URL",
)

BUILD_ARGUMENTS = REQUIRED_BUILD_ARGUMENTS + OPTIONAL_BUILD_ARGUMENTS

CONFIGURATIONS = {
    # Mirrors the ROCm and PyTorch versions used by the current gfx90a CI
    # runner while building the stack independently from an Ubuntu base.
    "rocm-7.0-pytorch-2.8-gfx90a": {
        "tag": "rocm7.0.0-pytorch2.8.0-gfx90a-r1",
        "build_args": {
            "BASE_IMAGE": ("ubuntu@sha256:"
                           "4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"),
            "ROCM_VERSION": "7.0.0",
            "ROCM_RELEASE_TYPE": "pre-therock",
            "ROCM_REPO_DIRECTORY": "7.0",
            "PYTORCH_VERSION": "2.8.0+rocm7.0.0.git64359f59",
            "PYTORCH_INDEX_URL": ("https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/"),
            "HIP_PYTHON_VERSION": "7.0.2.555.40",
            "PYTORCH_GPU_TARGETS": "gfx90a",
        },
    },
    # Mirrors the ROCm and PyTorch versions shared by the current gfx942 and
    # gfx950 CI runners while building independently from an Ubuntu base.
    "rocm-7.2.4-pytorch-2.10-gfx942-gfx950": {
        "tag": "rocm7.2.4-pytorch2.10.0-gfx942-gfx950-r1",
        "build_args": {
            "BASE_IMAGE": ("ubuntu@sha256:"
                           "4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"),
            "ROCM_VERSION": "7.2.4",
            "ROCM_RELEASE_TYPE": "pre-therock",
            "ROCM_REPO_DIRECTORY": "7.2.4",
            "PYTORCH_VERSION": ("2.10.0+rocm7.2.4.lw.git3d3aa833"),
            "PYTORCH_INDEX_URL": ("https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/"),
            "HIP_PYTHON_VERSION": "7.2.2.562.43",
            "PYTORCH_GPU_TARGETS": "gfx942,gfx950",
        },
    },
    "rocm-10.1-pytorch-2.15-nightly-20260820": {
        "tag": ("rocm10.1.0a20260820-pytorch2.15.0a0-"
                "gfx90a-gfx942-gfx950-gfx1250-r1"),
        "build_args": {
            "BASE_IMAGE": ("ubuntu@sha256:"
                           "4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"),
            "ROCM_VERSION": "10.1.0a20260820",
            "ROCM_RELEASE_TYPE": "nightlies",
            "ROCM_REPO_DIRECTORY": "20260820-32315755045",
            "PYTORCH_VERSION": "2.15.0a0+rocm10.1.0a20260820",
            "PYTORCH_INDEX_URL": ("https://rocm.nightlies.amd.com/whl-multi-arch/"),
            "HIP_PYTHON_VERSION": "7.2.2.562.43",
            "PYTORCH_GPU_TARGETS": "gfx90a,gfx942,gfx950,gfx1250",
        },
    },
    "rocm-7.15-pytorch-2.11-nightly-gfx1250-73a658d5": {
        "tag": ("rocm7.15.0.dev0-pytorch2.11.0.dev-gfx1250-"
                "73a658d5-r1"),
        "build_args": {
            "BASE_IMAGE": ("ubuntu@sha256:"
                           "4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"),
            "ROCM_VERSION": ("7.15.0.dev0+"
                             "73a658d545d8b8aaf0aa3d08c0c80bb37667878a"),
            "ROCM_RELEASE_TYPE": "prereleases",
            "ROCM_REPO_DIRECTORY": "73a658d545d8b8aaf0aa3d08c0c80bb37667878a",
            "PYTORCH_VERSION": ("2.11.0+devrocm7.15.0.dev0."
                                "73a658d545d8b8aaf0aa3d08c0c80bb37667878a"),
            "PYTORCH_INDEX_URL": ("https://rocm.nightlies.amd.com/"
                                  "whl-multi-arch/"),
            "PYTORCH_EXTRA_INDEX_URL": ("https://rocm.devreleases.amd.com/"
                                        "whl-multi-arch/"),
            "PYTORCH_DEVICE_WHEEL_URL": (
                "https://rocm.devreleases.amd.com/whl-multi-arch/"
                "amd_torch_device_gfx1250-2.11.0%2Bdevrocm7.15.0.dev0."
                "73a658d545d8b8aaf0aa3d08c0c80bb37667878a-"
                "cp312-cp312-linux_x86_64.whl"
            ),
            "HIP_PYTHON_VERSION": "7.2.2.562.43",
            "PYTORCH_GPU_TARGETS": "gfx1250",
        },
    },
}


def validate_configurations() -> None:
    required_arguments = set(REQUIRED_BUILD_ARGUMENTS)
    allowed_arguments = set(BUILD_ARGUMENTS)
    tags = set()

    for name, configuration in CONFIGURATIONS.items():
        if not isinstance(name, str) or not name:
            raise ValueError("configuration names must be nonempty strings")
        if not isinstance(configuration, dict):
            raise ValueError(f"{name}: configuration must be a dictionary")

        tag = configuration.get("tag")
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"{name}: tag must be a nonempty string")
        if tag in tags:
            raise ValueError(f"{name}: duplicate tag: {tag}")
        tags.add(tag)

        build_args = configuration.get("build_args")
        if not isinstance(build_args, dict):
            raise ValueError(f"{name}: build_args must be a dictionary")

        actual_arguments = set(build_args)
        missing = required_arguments - actual_arguments
        unexpected = actual_arguments - allowed_arguments
        if missing:
            raise ValueError(f"{name}: missing build arguments: {', '.join(sorted(missing))}")
        if unexpected:
            raise ValueError(f"{name}: unexpected build arguments: "
                             f"{', '.join(sorted(unexpected))}")
        for argument, value in build_args.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name}: {argument} must be a nonempty string")

        device_wheel_url = build_args.get("PYTORCH_DEVICE_WHEEL_URL")
        extra_index_url = build_args.get("PYTORCH_EXTRA_INDEX_URL")
        if bool(device_wheel_url) != bool(extra_index_url):
            raise ValueError(
                f"{name}: PYTORCH_DEVICE_WHEEL_URL and "
                "PYTORCH_EXTRA_INDEX_URL must be specified together"
            )

        if build_args["ROCM_RELEASE_TYPE"] == "nightlies" and not re.fullmatch(r"[0-9]{8}-[0-9]+",
                                                                               build_args["ROCM_REPO_DIRECTORY"]):
            raise ValueError(f"{name}: ROCM_REPO_DIRECTORY must have the form "
                             "YYYYMMDD-NUMERIC_ID for nightlies")

        gpu_targets = build_args["PYTORCH_GPU_TARGETS"]
        if gpu_targets != "all" and not re.fullmatch(r"gfx[0-9][0-9a-z-]*(?:,gfx[0-9][0-9a-z-]*)*", gpu_targets):
            raise ValueError(f"{name}: PYTORCH_GPU_TARGETS must be 'all' or "
                             "a comma-separated list of gfx targets")


def image_reference(configuration_name: str, image_repository: str) -> str:
    return f"{image_repository}:{CONFIGURATIONS[configuration_name]['tag']}"


def docker_command(configuration_name: str, image_repository: str) -> list[str]:
    configuration = CONFIGURATIONS[configuration_name]
    command = [
        "docker",
        "build",
        "--file",
        str(DOCKERFILE),
    ]
    for argument in BUILD_ARGUMENTS:
        if argument not in configuration["build_args"]:
            continue
        command.extend([
            "--build-arg",
            f"{argument}={configuration['build_args'][argument]}",
        ])
    command.extend([
        "--tag",
        image_reference(configuration_name, image_repository),
        str(REPOSITORY_ROOT),
    ])
    return command


def print_command(command: list[str]) -> None:
    print(shlex.join(command))


def list_configurations() -> None:
    for name, configuration in CONFIGURATIONS.items():
        build_args = configuration["build_args"]
        print(name)
        print(f"  tag: {configuration['tag']}")
        print(f"  ROCm: {build_args['ROCM_VERSION']} "
              f"({build_args['ROCM_REPO_DIRECTORY']})")
        print(f"  PyTorch: {build_args['PYTORCH_VERSION']}")
        print(f"  GPU targets: {build_args['PYTORCH_GPU_TARGETS']}")


def show_configuration(name: str, image_repository: str) -> None:
    configuration = CONFIGURATIONS[name]
    print(f"name: {name}")
    print(f"image: {image_reference(name, image_repository)}")
    for argument in BUILD_ARGUMENTS:
        if argument in configuration["build_args"]:
            print(f"{argument}={configuration['build_args'][argument]}")


def build_configuration(name: str, image_repository: str, dry_run: bool) -> None:
    command = docker_command(name, image_repository)
    print_command(command)
    if not dry_run:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        image = image_reference(name, image_repository)
        print(f"Built {image}")
        print(f"Publish manually with: docker push {image}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Docker images used by Triton AMD CI.")
    parser.add_argument(
        "--image-repository",
        default=os.environ.get(IMAGE_REPOSITORY_ENV, DEFAULT_IMAGE_REPOSITORY),
        help=("Docker image repository "
              f"(default: ${IMAGE_REPOSITORY_ENV} or "
              f"{DEFAULT_IMAGE_REPOSITORY})"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print Docker commands without running them",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list available configurations")

    show_parser = subparsers.add_parser("show", help="show one configuration")
    show_parser.add_argument("configuration", choices=CONFIGURATIONS)

    build_parser = subparsers.add_parser("build", help="build one configuration")
    build_parser.add_argument("configuration", choices=CONFIGURATIONS)

    subparsers.add_parser("build-all", help="build every configuration")
    return parser.parse_args()


def main() -> int:
    try:
        validate_configurations()
        args = parse_arguments()

        if args.command == "list":
            list_configurations()
        elif args.command == "show":
            show_configuration(args.configuration, args.image_repository)
        elif args.command == "build":
            build_configuration(args.configuration, args.image_repository, args.dry_run)
        elif args.command == "build-all":
            for name in CONFIGURATIONS:
                build_configuration(name, args.image_repository, args.dry_run)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
