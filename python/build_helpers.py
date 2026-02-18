import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional


def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    return Path(get_base_dir()) / "build" / dir_name


def get_cmake_dir():
    cmake_dir = os.getenv("TRITON_BUILD_DIR", default=_get_cmake_dir())
    cmake_dir = Path(cmake_dir)
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


@dataclass
class BuildHelperArgs:
    cache_path: str
    offline_build: bool
    llvm_system_suffix: Optional[str]
    llvm_syspath: Optional[str]
    json_syspath: Optional[str]
    ptxas_path: Optional[str]
    ptxas_blackwell_path: Optional[str]
    cuobjdump_path: Optional[str]
    nvdisasm_path: Optional[str]
    cudacrt_path: Optional[str]
    cudart_path: Optional[str]
    cupti_include_path: Optional[str]
    cupti_lib_path: Optional[str]
    cupti_lib_blackwell_path: Optional[str]


def _normalize_bool(value: str, default: str = "") -> bool:
    effective_value = value if value is not None else default
    return effective_value.upper() in ["ON", "1", "YES", "TRUE", "Y"]


def _normalize_optional(value: str) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _normalize_optional_path(value: str) -> Optional[str]:
    normalized = _normalize_optional(value)
    if normalized is None:
        return None
    return os.path.abspath(os.path.expanduser(normalized))


def _normalize_required_path(value: str, name: str) -> str:
    normalized = _normalize_optional_path(value)
    if normalized is None:
        raise RuntimeError(f"{name} must be provided to build_helpers.py")
    return normalized


def open_url(url):
    user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
    headers = {
        "User-Agent": user_agent,
    }
    request = urllib.request.Request(url, None, headers)
    # Set timeout to 300 seconds to prevent the request from hanging forever.
    return urllib.request.urlopen(request, timeout=300)


def update_symlink(link_path, source_path):
    source_path = Path(source_path)
    link_path = Path(link_path)

    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        shutil.rmtree(link_path)

    print(f"creating symlink: {link_path} -> {source_path}", file=sys.stderr)
    link_path.absolute().parent.mkdir(parents=True, exist_ok=True)  # Ensure link's parent directory exists
    link_path.symlink_to(source_path.absolute(), target_is_directory=True)


# --- third party packages -----


@dataclass
class Package:
    package: str
    name: str
    url: str
    include_flag: str
    lib_flag: str
    syspath_var_name: str
    sym_name: Optional[str] = None


def get_json_package_info():
    json_version_path = os.path.join(get_base_dir(), "cmake", "json-version.txt")
    with open(json_version_path, "r") as json_version_file:
        version = json_version_file.read().strip()
    url = f"https://github.com/nlohmann/json/releases/download/{version}/include.zip"
    return Package("json", "", url, "JSON_INCLUDE_DIR", "", "JSON_SYSPATH")


def is_linux_os(os_id):
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release", "r") as os_release_file:
            os_release_content = os_release_file.read()
            return f'ID="{os_id}"' in os_release_content
    return False


def get_llvm_package_info(helper_args: BuildHelperArgs):
    system = platform.system()
    try:
        arch = {"x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[platform.machine()]
    except KeyError:
        arch = platform.machine()
    if helper_args.llvm_system_suffix:
        system_suffix = helper_args.llvm_system_suffix
    elif system == "Darwin":
        system_suffix = f"macos-{arch}"
    elif system == "Linux":
        if arch == "arm64" and is_linux_os("almalinux"):
            system_suffix = "almalinux-arm64"
        elif arch == "arm64":
            system_suffix = "ubuntu-arm64"
        elif arch == "x64":
            vglibc = tuple(map(int, platform.libc_ver()[1].split(".")))
            vglibc = vglibc[0] * 100 + vglibc[1]
            if vglibc > 228:
                # Ubuntu 24 LTS (v2.39)
                # Ubuntu 22 LTS (v2.35)
                # Ubuntu 20 LTS (v2.31)
                system_suffix = "ubuntu-x64"
            else:
                # Manylinux_2.28 (v2.28)
                # AlmaLinux 8 (v2.28)
                system_suffix = "almalinux-x64"
        else:
            print(
                f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
            )
            return Package("llvm", "LLVM-C.lib", "", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")
    else:
        print(
            f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
        )
        return Package("llvm", "LLVM-C.lib", "", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")
    llvm_hash_path = os.path.join(get_base_dir(), "cmake", "llvm-hash.txt")
    with open(llvm_hash_path, "r") as llvm_hash_file:
        rev = llvm_hash_file.read(8)
    name = f"llvm-{rev}-{system_suffix}"
    # Create a stable symlink that doesn't include revision
    sym_name = f"llvm-{system_suffix}"
    url = f"https://oaitriton.blob.core.windows.net/public/llvm-builds/{name}.tar.gz"
    return Package("llvm", name, url, "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH", sym_name=sym_name)


def _get_syspath_override(package_syspath_var_name: str, helper_args: BuildHelperArgs) -> Optional[str]:
    syspath_overrides = {
        "LLVM_SYSPATH": helper_args.llvm_syspath,
        "JSON_SYSPATH": helper_args.json_syspath,
    }
    return syspath_overrides.get(package_syspath_var_name)


def _get_thirdparty_package_cmake_vars(package: Package, helper_args: BuildHelperArgs):
    cache_path = helper_args.cache_path
    package_root_dir = os.path.join(cache_path, package.package)
    package_dir = os.path.join(package_root_dir, package.name)
    syspath_override = _get_syspath_override(package.syspath_var_name, helper_args)
    if syspath_override is not None:
        package_dir = syspath_override
    version_file_path = os.path.join(package_dir, "version.txt")

    input_defined = syspath_override is not None
    input_exists = os.path.exists(version_file_path)
    input_compatible = input_exists and Path(version_file_path).read_text() == package.url

    if helper_args.offline_build and not input_defined:
        raise RuntimeError(f"Requested an offline build but {package.syspath_var_name} is not set")
    if not helper_args.offline_build and not input_defined and not input_compatible:
        with contextlib.suppress(Exception):
            shutil.rmtree(package_root_dir)
        os.makedirs(package_root_dir, exist_ok=True)
        print(f"downloading and extracting {package.url} ...")
        with open_url(package.url) as response:
            if package.url.endswith(".zip"):
                file_bytes = BytesIO(response.read())
                with zipfile.ZipFile(file_bytes, "r") as file:
                    file.extractall(path=package_root_dir)
            else:
                with tarfile.open(fileobj=response, mode="r|*") as file:
                    # Use extractall without filter for Python version < 3.12 compatibility
                    if hasattr(tarfile, "data_filter"):
                        file.extractall(path=package_root_dir, filter="data")
                    else:
                        file.extractall(path=package_root_dir)
        # write version url to package_dir
        with open(os.path.join(package_dir, "version.txt"), "w") as file:
            file.write(package.url)
    if package.sym_name is not None:
        sym_link_path = os.path.join(package_root_dir, package.sym_name)
        update_symlink(sym_link_path, package_dir)

    cmake_vars = {}
    if package.include_flag:
        cmake_vars[package.include_flag] = f"{package_dir}/include"
    if package.lib_flag:
        cmake_vars[package.lib_flag] = f"{package_dir}/lib"
    if package.syspath_var_name:
        cmake_vars[package.syspath_var_name] = package_dir
    return cmake_vars


def get_thirdparty_cmake_vars(packages: list[str], helper_args: BuildHelperArgs):
    package_infos = []
    for package in packages:
        if package == "llvm":
            package_infos.append(get_llvm_package_info(helper_args))
        elif package == "json":
            package_infos.append(get_json_package_info())
        else:
            raise ValueError(f"Unsupported package '{package}'")

    cmake_vars = {}
    for package_info in package_infos:
        cmake_vars.update(_get_thirdparty_package_cmake_vars(package_info, helper_args))
    return cmake_vars


def _cmake_escape(value: str) -> str:
    return value.replace("\\", "/").replace('"', '\\"')


def write_thirdparty_cmake_vars(output: str, packages: list[str], helper_args: BuildHelperArgs):
    cmake_vars = get_thirdparty_cmake_vars(packages, helper_args)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as output_file:
        for key, value in sorted(cmake_vars.items()):
            output_file.write(f'if(NOT DEFINED {key} OR "${{{key}}}" STREQUAL "")\n')
            output_file.write(f'  set({key} "{_cmake_escape(value)}")\n')
            output_file.write('endif()\n')


# --- nvidia toolchain helpers -----


def download_and_copy(name, src_func, dst_path, override_path, version, url_func, helper_args: BuildHelperArgs):
    if helper_args.offline_build:
        return
    cache_path = helper_args.cache_path
    if override_path is not None:
        return
    base_dir = get_base_dir()
    system = platform.system()
    arch = platform.machine()
    # NOTE: This might be wrong for jetson if both grace chips and jetson chips return aarch64
    arch = {"arm64": "sbsa", "aarch64": "sbsa"}.get(arch, arch)
    supported = {"Linux": "linux", "Darwin": "linux"}
    url = url_func(supported[system], arch, version)
    src_path = src_func(supported[system], arch, version)
    tmp_path = os.path.join(cache_path, "nvidia", name)  # path to cache the download
    dst_path = os.path.join(base_dir, "third_party", "nvidia", "backend", dst_path)  # final binary path
    src_path = os.path.join(tmp_path, src_path)
    download = not os.path.exists(src_path)
    if os.path.exists(dst_path) and system == "Linux" and shutil.which(dst_path) is not None:
        curr_version = subprocess.check_output([dst_path, "--version"]).decode("utf-8").strip()
        curr_version = re.search(r"V([.|\d]+)", curr_version)
        assert curr_version is not None, f"No version information for {dst_path}"
        download = download or curr_version.group(1) != version
    if download:
        print(f"downloading and extracting {url} ...")
        with open_url(url) as url_file, tarfile.open(fileobj=url_file, mode="r|*") as tar_file:
            # Use extractall without filter for Python version < 3.12 compatibility
            if hasattr(tarfile, "data_filter"):
                tar_file.extractall(path=tmp_path, filter="data")
            else:
                tar_file.extractall(path=tmp_path)
    os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
    print(f"copy {src_path} to {dst_path} ...")
    if os.path.isdir(src_path):
        # Use copy (not copy2) so destination mtimes are refreshed and Ninja sees dependent headers as updated.
        shutil.copytree(src_path, dst_path, copy_function=shutil.copy, dirs_exist_ok=True)
    else:
        shutil.copy(src_path, dst_path)


def download_and_copy_dependencies(helper_args: BuildHelperArgs):
    nvidia_version_path = os.path.join(get_base_dir(), "cmake", "nvidia-toolchain-version.json")
    with open(nvidia_version_path, "r") as nvidia_version_file:
        # parse this json file to get the version of the nvidia toolchain
        nvidia_toolchain_version = json.load(nvidia_version_file)

    exe_extension = sysconfig.get_config_var("EXE")
    download_and_copy(
        name="nvcc",
        src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin/ptxas{exe_extension}",
        dst_path="bin/ptxas",
        override_path=helper_args.ptxas_path,
        version=nvidia_toolchain_version["ptxas"],
        url_func=lambda system, arch, version:
        f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
        helper_args=helper_args,
    )

    # We download a separate ptxas for blackwell, since there are some bugs when using it for hopper
    download_and_copy(
        name="nvcc",
        src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin/ptxas{exe_extension}",
        dst_path="bin/ptxas-blackwell",
        override_path=helper_args.ptxas_blackwell_path,
        version=nvidia_toolchain_version["ptxas-blackwell"],
        url_func=lambda system, arch, version:
        f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
        helper_args=helper_args,
    )
    download_and_copy(
        name="cuobjdump",
        src_func=lambda system, arch, version:
        f"cuda_cuobjdump-{system}-{arch}-{version}-archive/bin/cuobjdump{exe_extension}",
        dst_path="bin/cuobjdump",
        override_path=helper_args.cuobjdump_path,
        version=nvidia_toolchain_version["cuobjdump"],
        url_func=lambda system, arch, version:
        f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cuobjdump/{system}-{arch}/cuda_cuobjdump-{system}-{arch}-{version}-archive.tar.xz",
        helper_args=helper_args,
    )
    download_and_copy(
        name="nvdisasm",
        src_func=lambda system, arch, version:
        f"cuda_nvdisasm-{system}-{arch}-{version}-archive/bin/nvdisasm{exe_extension}",
        dst_path="bin/nvdisasm",
        override_path=helper_args.nvdisasm_path,
        version=nvidia_toolchain_version["nvdisasm"],
        url_func=lambda system, arch, version:
        f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvdisasm/{system}-{arch}/cuda_nvdisasm-{system}-{arch}-{version}-archive.tar.xz",
        helper_args=helper_args,
    )
    crt = "crt" if int(nvidia_toolchain_version["cudacrt"].split(".")[0]) >= 13 else "nvcc"
    download_and_copy(
        name="nvcc",
        src_func=lambda system, arch, version: f"cuda_{crt}-{system}-{arch}-{version}-archive/include",
        dst_path="include",
        override_path=helper_args.cudacrt_path,
        version=nvidia_toolchain_version["cudacrt"],
        url_func=lambda system, arch, version:
        f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_{crt}/{system}-{arch}/cuda_{crt}-{system}-{arch}-{version}-archive.tar.xz",
        helper_args=helper_args,
    )
    download_and_copy(
        name="cudart",
        src_func=lambda system, arch, version: f"cuda_cudart-{system}-{arch}-{version}-archive/include",
        dst_path="include",
        override_path=helper_args.cudart_path,
        version=nvidia_toolchain_version["cudart"],
        url_func=lambda system, arch, version:
        f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/{system}-{arch}/cuda_cudart-{system}-{arch}-{version}-archive.tar.xz",
        helper_args=helper_args,
    )
    download_and_copy(
        name="cupti",
        src_func=lambda system, arch, version: f"cuda_cupti-{system}-{arch}-{version}-archive/include",
        dst_path="include",
        override_path=helper_args.cupti_include_path,
        version=nvidia_toolchain_version["cupti"],
        url_func=lambda system, arch, version:
        f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/{system}-{arch}/cuda_cupti-{system}-{arch}-{version}-archive.tar.xz",
        helper_args=helper_args,
    )
    download_and_copy(
        name="cupti",
        src_func=lambda system, arch, version: f"cuda_cupti-{system}-{arch}-{version}-archive/lib",
        dst_path="lib/cupti",
        override_path=helper_args.cupti_lib_path,
        version=nvidia_toolchain_version["cupti"],
        url_func=lambda system, arch, version:
        f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/{system}-{arch}/cuda_cupti-{system}-{arch}-{version}-archive.tar.xz",
        helper_args=helper_args,
    )
    download_and_copy(
        name="cupti",
        src_func=lambda system, arch, version: f"cuda_cupti-{system}-{arch}-{version}-archive/lib",
        dst_path="lib/cupti-blackwell",
        override_path=helper_args.cupti_lib_blackwell_path,
        version=nvidia_toolchain_version["cupti-blackwell"],
        url_func=lambda system, arch, version:
        f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/{system}-{arch}/cuda_cupti-{system}-{arch}-{version}-archive.tar.xz",
        helper_args=helper_args,
    )


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--triton-cache-path", required=True, help="Cache directory path")
    parser.add_argument("--triton-offline-build", action="store_true", help="Build without downloading dependencies")
    parser.add_argument("--triton-llvm-system-suffix", default="", help="Override LLVM system suffix")
    parser.add_argument("--llvm-syspath", default="", help="Path override for LLVM_SYSPATH")
    parser.add_argument("--json-syspath", default="", help="Path override for JSON_SYSPATH")
    parser.add_argument("--triton-ptxas-path", default="", help="Path override for TRITON_PTXAS_PATH")
    parser.add_argument(
        "--triton-ptxas-blackwell-path",
        default="",
        help="Path override for TRITON_PTXAS_BLACKWELL_PATH",
    )
    parser.add_argument("--triton-cuobjdump-path", default="", help="Path override for TRITON_CUOBJDUMP_PATH")
    parser.add_argument("--triton-nvdisasm-path", default="", help="Path override for TRITON_NVDISASM_PATH")
    parser.add_argument("--triton-cudacrt-path", default="", help="Path override for TRITON_CUDACRT_PATH")
    parser.add_argument("--triton-cudart-path", default="", help="Path override for TRITON_CUDART_PATH")
    parser.add_argument(
        "--triton-cupti-include-path",
        default="",
        help="Path override for TRITON_CUPTI_INCLUDE_PATH",
    )
    parser.add_argument("--triton-cupti-lib-path", default="", help="Path override for TRITON_CUPTI_LIB_PATH")
    parser.add_argument("--triton-cupti-lib-blackwell-path", default="",
                        help="Path override for TRITON_CUPTI_LIB_BLACKWELL_PATH")


def normalize_parsed_args(parsed_args) -> BuildHelperArgs:
    return BuildHelperArgs(
        cache_path=_normalize_required_path(parsed_args.triton_cache_path, "TRITON_CACHE_PATH"),
        offline_build=parsed_args.triton_offline_build,
        llvm_system_suffix=_normalize_optional(parsed_args.triton_llvm_system_suffix),
        llvm_syspath=_normalize_optional_path(parsed_args.llvm_syspath),
        json_syspath=_normalize_optional_path(parsed_args.json_syspath),
        ptxas_path=_normalize_optional_path(parsed_args.triton_ptxas_path),
        ptxas_blackwell_path=_normalize_optional_path(parsed_args.triton_ptxas_blackwell_path),
        cuobjdump_path=_normalize_optional_path(parsed_args.triton_cuobjdump_path),
        nvdisasm_path=_normalize_optional_path(parsed_args.triton_nvdisasm_path),
        cudacrt_path=_normalize_optional_path(parsed_args.triton_cudacrt_path),
        cudart_path=_normalize_optional_path(parsed_args.triton_cudart_path),
        cupti_include_path=_normalize_optional_path(parsed_args.triton_cupti_include_path),
        cupti_lib_path=_normalize_optional_path(parsed_args.triton_cupti_lib_path),
        cupti_lib_blackwell_path=_normalize_optional_path(parsed_args.triton_cupti_lib_blackwell_path),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Triton build helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download_and_copy_dependencies",
        help="Download and copy NVIDIA toolchain dependencies",
    )
    add_common_args(download_parser)

    write_vars_parser = subparsers.add_parser(
        "write_thirdparty_cmake_vars",
        help="Resolve third-party packages and write CMake variable assignments",
    )
    add_common_args(write_vars_parser)
    write_vars_parser.add_argument("--output", required=True, help="Path to the output CMake file")
    write_vars_parser.add_argument(
        "--packages",
        nargs="+",
        required=True,
        choices=["llvm", "json"],
        help="Third-party packages to resolve",
    )

    parsed_args = parser.parse_args(argv)
    helper_args = normalize_parsed_args(parsed_args)
    if parsed_args.command == "download_and_copy_dependencies":
        download_and_copy_dependencies(helper_args)
    elif parsed_args.command == "write_thirdparty_cmake_vars":
        write_thirdparty_cmake_vars(output=parsed_args.output, packages=parsed_args.packages, helper_args=helper_args)


if __name__ == "__main__":
    main()
