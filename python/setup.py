import os
import platform
import re
import contextlib
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
import urllib.request
import json
from io import BytesIO
from distutils.command.clean import clean
from pathlib import Path
from typing import List, Optional

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from dataclasses import dataclass

from distutils.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from wheel.bdist_wheel import bdist_wheel

import pybind11

from build_helpers import get_base_dir, get_cmake_dir


@dataclass
class Backend:
    name: str
    package_data: List[str]
    language_package_data: List[str]
    tools_package_data: List[str]
    src_dir: str
    backend_dir: str
    language_dir: Optional[str]
    tools_dir: Optional[str]
    install_dir: str
    is_external: bool


class BackendInstaller:

    @staticmethod
    def prepare(backend_name: str, backend_src_dir: str = None, is_external: bool = False):
        # Initialize submodule if there is one for in-tree backends.
        if not is_external:
            root_dir = os.path.join(os.pardir, "third_party")
            assert backend_name in os.listdir(
                root_dir), f"{backend_name} is requested for install but not present in {root_dir}"

            try:
                subprocess.run(["git", "submodule", "update", "--init", f"{backend_name}"], check=True,
                               stdout=subprocess.DEVNULL, cwd=root_dir)
            except subprocess.CalledProcessError:
                pass
            except FileNotFoundError:
                pass

            backend_src_dir = os.path.join(root_dir, backend_name)

        backend_path = os.path.abspath(os.path.join(backend_src_dir, "backend"))
        assert os.path.exists(backend_path), f"{backend_path} does not exist!"

        language_dir = os.path.abspath(os.path.join(backend_src_dir, "language"))
        if not os.path.exists(language_dir):
            language_dir = None

        tools_dir = os.path.abspath(os.path.join(backend_src_dir, "tools"))
        if not os.path.exists(tools_dir):
            tools_dir = None

        for file in ["compiler.py", "driver.py"]:
            assert os.path.exists(os.path.join(backend_path, file)), f"${file} does not exist in ${backend_path}"

        install_dir = os.path.join(os.path.dirname(__file__), "triton", "backends", backend_name)
        package_data = [f"{os.path.relpath(p, backend_path)}/*" for p, _, _, in os.walk(backend_path)]

        language_package_data = []
        if language_dir is not None:
            language_package_data = [f"{os.path.relpath(p, language_dir)}/*" for p, _, _, in os.walk(language_dir)]

        tools_package_data = []
        if tools_dir is not None:
            tools_package_data = [f"{os.path.relpath(p, tools_dir)}/*" for p, _, _, in os.walk(tools_dir)]

        return Backend(name=backend_name, package_data=package_data, language_package_data=language_package_data,
                       tools_package_data=tools_package_data, src_dir=backend_src_dir, backend_dir=backend_path,
                       language_dir=language_dir, tools_dir=tools_dir, install_dir=install_dir, is_external=is_external)

    # Copy all in-tree backends under triton/third_party.
    @staticmethod
    def copy(active):
        return [BackendInstaller.prepare(backend) for backend in active]

    # Copy all external plugins provided by the `TRITON_PLUGIN_DIRS` env var.
    # TRITON_PLUGIN_DIRS is a semicolon-separated list of paths to the plugins.
    # Expect to find the name of the backend under dir/backend/name.conf
    @staticmethod
    def copy_externals():
        backend_dirs = os.getenv("TRITON_PLUGIN_DIRS")
        if backend_dirs is None:
            return []
        backend_dirs = backend_dirs.strip().split(";")
        backend_names = [Path(os.path.join(dir, "backend", "name.conf")).read_text().strip() for dir in backend_dirs]
        return [
            BackendInstaller.prepare(backend_name, backend_src_dir=backend_src_dir, is_external=True)
            for backend_name, backend_src_dir in zip(backend_names, backend_dirs)
        ]


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_build_type():
    if check_env_flag("DEBUG"):
        return "Debug"
    elif check_env_flag("REL_WITH_DEB_INFO"):
        return "RelWithDebInfo"
    elif check_env_flag("TRITON_REL_BUILD_WITH_ASSERTS"):
        return "TritonRelBuildWithAsserts"
    elif check_env_flag("TRITON_BUILD_WITH_O1"):
        return "TritonBuildWithO1"
    else:
        # TODO: change to release when stable enough
        return "TritonRelBuildWithAsserts"


def get_env_with_keys(key: list):
    for k in key:
        if k in os.environ:
            return os.environ[k]
    return ""


def is_offline_build() -> bool:
    """
    Downstream projects and distributions which bootstrap their own dependencies from scratch
    and run builds in offline sandboxes
    may set `TRITON_OFFLINE_BUILD` in the build environment to prevent any attempts at downloading
    pinned dependencies from the internet or at using dependencies vendored in-tree.

    Dependencies must be defined using respective search paths (cf. `syspath_var_name` in `Package`).
    Missing dependencies lead to an early abortion.
    Dependencies' compatibility is not verified.

    Note that this flag isn't tested by the CI and does not provide any guarantees.
    """
    return check_env_flag("TRITON_OFFLINE_BUILD", "")


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


# json
def get_json_package_info():
    url = "https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip"
    return Package("json", "", url, "JSON_INCLUDE_DIR", "", "JSON_SYSPATH")


# llvm
def get_llvm_package_info():
    system = platform.system()
    try:
        arch = {"x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[platform.machine()]
    except KeyError:
        arch = platform.machine()
    if system == "Darwin":
        system_suffix = f"macos-{arch}"
    elif system == "Linux":
        if arch == 'arm64':
            system_suffix = 'ubuntu-arm64'
        elif arch == 'x64':
            vglibc = tuple(map(int, platform.libc_ver()[1].split('.')))
            vglibc = vglibc[0] * 100 + vglibc[1]
            if vglibc > 228:
                # Ubuntu 24 LTS (v2.39)
                # Ubuntu 22 LTS (v2.35)
                # Ubuntu 20 LTS (v2.31)
                system_suffix = "ubuntu-x64"
            elif vglibc > 217:
                # Manylinux_2.28 (v2.28)
                # AlmaLinux 8 (v2.28)
                system_suffix = "almalinux-x64"
            else:
                # Manylinux_2014 (v2.17)
                # CentOS 7 (v2.17)
                system_suffix = "centos-x64"
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
    # use_assert_enabled_llvm = check_env_flag("TRITON_USE_ASSERT_ENABLED_LLVM", "False")
    # release_suffix = "assert" if use_assert_enabled_llvm else "release"
    llvm_hash_path = os.path.join(get_base_dir(), "cmake", "llvm-hash.txt")
    with open(llvm_hash_path, "r") as llvm_hash_file:
        rev = llvm_hash_file.read(8)
    name = f"llvm-{rev}-{system_suffix}"
    # Create a stable symlink that doesn't include revision
    sym_name = f"llvm-{system_suffix}"
    url = f"https://oaitriton.blob.core.windows.net/public/llvm-builds/{name}.tar.gz"
    return Package("llvm", name, url, "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH", sym_name=sym_name)


def open_url(url):
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0'
    headers = {
        'User-Agent': user_agent,
    }
    request = urllib.request.Request(url, None, headers)
    # Set timeout to 300 seconds to prevent the request from hanging forever.
    return urllib.request.urlopen(request, timeout=300)


# ---- package data ---


def get_triton_cache_path():
    user_home = os.getenv("TRITON_HOME")
    if not user_home:
        user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".triton")


def update_symlink(link_path, source_path):
    source_path = Path(source_path)
    link_path = Path(link_path)

    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        shutil.rmtree(link_path)

    print(f"creating symlink: {link_path} -> {source_path}", file=sys.stderr)
    link_path.absolute().parent.mkdir(parents=True, exist_ok=True)  # Ensure link's parent directory exists
    link_path.symlink_to(source_path, target_is_directory=True)


def get_thirdparty_packages(packages: list):
    triton_cache_path = get_triton_cache_path()
    thirdparty_cmake_args = []
    for p in packages:
        package_root_dir = os.path.join(triton_cache_path, p.package)
        package_dir = os.path.join(package_root_dir, p.name)
        if os.environ.get(p.syspath_var_name):
            package_dir = os.environ[p.syspath_var_name]
        version_file_path = os.path.join(package_dir, "version.txt")

        input_defined = p.syspath_var_name in os.environ
        input_exists = os.path.exists(version_file_path)
        input_compatible = input_exists and Path(version_file_path).read_text() == p.url

        if is_offline_build() and not input_defined:
            raise RuntimeError(f"Requested an offline build but {p.syspath_var_name} is not set")
        if not is_offline_build() and not input_defined and not input_compatible:
            with contextlib.suppress(Exception):
                shutil.rmtree(package_root_dir)
            os.makedirs(package_root_dir, exist_ok=True)
            print(f'downloading and extracting {p.url} ...')
            with open_url(p.url) as response:
                if p.url.endswith(".zip"):
                    file_bytes = BytesIO(response.read())
                    with zipfile.ZipFile(file_bytes, "r") as file:
                        file.extractall(path=package_root_dir)
                else:
                    with tarfile.open(fileobj=response, mode="r|*") as file:
                        file.extractall(path=package_root_dir)
            # write version url to package_dir
            with open(os.path.join(package_dir, "version.txt"), "w") as f:
                f.write(p.url)
        if p.include_flag:
            thirdparty_cmake_args.append(f"-D{p.include_flag}={package_dir}/include")
        if p.lib_flag:
            thirdparty_cmake_args.append(f"-D{p.lib_flag}={package_dir}/lib")
        if p.sym_name is not None:
            sym_link_path = os.path.join(package_root_dir, p.sym_name)
            update_symlink(sym_link_path, package_dir)

    return thirdparty_cmake_args


def download_and_copy(name, src_func, dst_path, variable, version, url_func):
    if is_offline_build():
        return
    triton_cache_path = get_triton_cache_path()
    if variable in os.environ:
        return
    base_dir = os.path.dirname(__file__)
    system = platform.system()
    arch = platform.machine()
    # NOTE: This might be wrong for jetson if both grace chips and jetson chips return aarch64
    arch = {"arm64": "sbsa", "aarch64": "sbsa"}.get(arch, arch)
    supported = {"Linux": "linux", "Darwin": "linux"}
    url = url_func(supported[system], arch, version)
    src_path = src_func(supported[system], arch, version)
    tmp_path = os.path.join(triton_cache_path, "nvidia", name)  # path to cache the download
    dst_path = os.path.join(base_dir, os.pardir, "third_party", "nvidia", "backend", dst_path)  # final binary path
    src_path = os.path.join(tmp_path, src_path)
    download = not os.path.exists(src_path)
    if os.path.exists(dst_path) and system == "Linux" and shutil.which(dst_path) is not None:
        curr_version = subprocess.check_output([dst_path, "--version"]).decode("utf-8").strip()
        curr_version = re.search(r"V([.|\d]+)", curr_version)
        assert curr_version is not None, f"No version information for {dst_path}"
        download = download or curr_version.group(1) != version
    if download:
        print(f'downloading and extracting {url} ...')
        file = tarfile.open(fileobj=open_url(url), mode="r|*")
        file.extractall(path=tmp_path)
    os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
    print(f'copy {src_path} to {dst_path} ...')
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        shutil.copy(src_path, dst_path)


# ---- cmake extension ----


class CMakeClean(clean):

    def initialize_options(self):
        clean.initialize_options(self)
        self.build_temp = get_cmake_dir()


class CMakeBuildPy(build_py):

    def run(self) -> None:
        self.run_command('build_ext')
        return super().run()


class CMakeExtension(Extension):

    def __init__(self, name, path, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path


class CMakeBuild(build_ext):

    user_options = build_ext.user_options + \
        [('base-dir=', None, 'base directory of Triton')]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.base_dir = get_base_dir()

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        match = re.search(r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", out.decode())
        cmake_major, cmake_minor = int(match.group("major")), int(match.group("minor"))
        if (cmake_major, cmake_minor) < (3, 18):
            raise RuntimeError("CMake >= 3.18.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def get_pybind11_cmake_args(self):
        pybind11_sys_path = get_env_with_keys(["PYBIND11_SYSPATH"])
        if pybind11_sys_path:
            pybind11_include_dir = os.path.join(pybind11_sys_path, "include")
        else:
            pybind11_include_dir = pybind11.get_include()
        return [f"-Dpybind11_INCLUDE_DIR='{pybind11_include_dir}'", f"-Dpybind11_DIR='{pybind11.get_cmake_dir()}'"]

    def get_proton_cmake_args(self):
        cmake_args = get_thirdparty_packages([get_json_package_info()])
        cmake_args += self.get_pybind11_cmake_args()
        cupti_include_dir = get_env_with_keys(["TRITON_CUPTI_INCLUDE_PATH"])
        if cupti_include_dir == "":
            cupti_include_dir = os.path.join(get_base_dir(), "third_party", "nvidia", "backend", "include")
        cmake_args += ["-DCUPTI_INCLUDE_DIR=" + cupti_include_dir]
        roctracer_include_dir = get_env_with_keys(["TRITON_ROCTRACER_INCLUDE_PATH"])
        if roctracer_include_dir == "":
            roctracer_include_dir = os.path.join(get_base_dir(), "third_party", "amd", "backend", "include")
        cmake_args += ["-DROCTRACER_INCLUDE_DIR=" + roctracer_include_dir]
        return cmake_args

    def build_extension(self, ext):
        lit_dir = shutil.which('lit')
        ninja_dir = shutil.which('ninja')
        # lit is used by the test suite
        thirdparty_cmake_args = get_thirdparty_packages([get_llvm_package_info()])
        thirdparty_cmake_args += self.get_pybind11_cmake_args()
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # python directories
        python_include_dir = sysconfig.get_path("platinclude")
        cmake_args = [
            "-G", "Ninja",  # Ninja is much faster than make
            "-DCMAKE_MAKE_PROGRAM=" +
            ninja_dir,  # Pass explicit path to ninja otherwise cmake may cache a temporary path
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DLLVM_ENABLE_WERROR=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir, "-DTRITON_BUILD_TUTORIALS=OFF",
            "-DTRITON_BUILD_PYTHON_MODULE=ON", "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DPython3_INCLUDE_DIR=" + python_include_dir,
            "-DTRITON_CODEGEN_BACKENDS=" + ';'.join([b.name for b in backends if not b.is_external]),
            "-DTRITON_PLUGIN_DIRS=" + ';'.join([b.src_dir for b in backends if b.is_external])
        ]
        if lit_dir is not None:
            cmake_args.append("-DLLVM_EXTERNAL_LIT=" + lit_dir)
        cmake_args.extend(thirdparty_cmake_args)

        # configuration
        cfg = get_build_type()
        build_args = ["--config", cfg]

        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
        else:
            max_jobs = os.getenv("MAX_JOBS", str(2 * os.cpu_count()))
            build_args += ['-j' + max_jobs]

        if check_env_flag("TRITON_BUILD_WITH_CLANG_LLD"):
            cmake_args += [
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_LINKER=lld",
                "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            ]

        # Note that asan doesn't work with binaries that use the GPU, so this is
        # only useful for tools like triton-opt that don't run code on the GPU.
        #
        # I tried and gave up getting msan to work.  It seems that libstdc++'s
        # std::string does not play nicely with clang's msan (I didn't try
        # gcc's).  I was unable to configure clang to ignore the error, and I
        # also wasn't able to get libc++ to work, but that doesn't mean it's
        # impossible. :)
        if check_env_flag("TRITON_BUILD_WITH_ASAN"):
            cmake_args += [
                "-DCMAKE_C_FLAGS=-fsanitize=address",
                "-DCMAKE_CXX_FLAGS=-fsanitize=address",
            ]

        # environment variables we will pass through to cmake
        passthrough_args = [
            "TRITON_BUILD_PROTON",
            "TRITON_BUILD_TUTORIALS",
            "TRITON_BUILD_WITH_CCACHE",
            "TRITON_PARALLEL_LINK_JOBS",
        ]
        cmake_args += [f"-D{option}={os.getenv(option)}" for option in passthrough_args if option in os.environ]

        if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
            cmake_args += self.get_proton_cmake_args()

        if is_offline_build():
            # unit test builds fetch googletests from GitHub
            cmake_args += ["-DTRITON_BUILD_UT=OFF"]

        cmake_args_append = os.getenv("TRITON_APPEND_CMAKE_ARGS")
        if cmake_args_append is not None:
            cmake_args += shlex.split(cmake_args_append)

        env = os.environ.copy()
        cmake_dir = get_cmake_dir()
        subprocess.check_call(["cmake", self.base_dir] + cmake_args, cwd=cmake_dir, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=cmake_dir)
        subprocess.check_call(["cmake", "--build", ".", "--target", "mlir-doc"], cwd=cmake_dir)


nvidia_version_path = os.path.join(get_base_dir(), "cmake", "nvidia-toolchain-version.json")
with open(nvidia_version_path, "r") as nvidia_version_file:
    # parse this json file to get the version of the nvidia toolchain
    NVIDIA_TOOLCHAIN_VERSION = json.load(nvidia_version_file)

exe_extension = sysconfig.get_config_var("EXE")
download_and_copy(
    name="nvcc",
    src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin/ptxas{exe_extension}",
    dst_path="bin/ptxas",
    variable="TRITON_PTXAS_PATH",
    version=NVIDIA_TOOLCHAIN_VERSION["ptxas"],
    url_func=lambda system, arch, version:
    f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
)
# We download a separate ptxas for blackwell, since there are some bugs when using it for hopper
download_and_copy(
    name="nvcc",
    src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin/ptxas{exe_extension}",
    dst_path="bin/ptxas-blackwell",
    variable="TRITON_PTXAS_PATH",
    version=NVIDIA_TOOLCHAIN_VERSION["ptxas-blackwell"],
    url_func=lambda system, arch, version:
    f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
)
download_and_copy(
    name="cuobjdump",
    src_func=lambda system, arch, version:
    f"cuda_cuobjdump-{system}-{arch}-{version}-archive/bin/cuobjdump{exe_extension}",
    dst_path="bin/cuobjdump",
    variable="TRITON_CUOBJDUMP_PATH",
    version=NVIDIA_TOOLCHAIN_VERSION["cuobjdump"],
    url_func=lambda system, arch, version:
    f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cuobjdump/{system}-{arch}/cuda_cuobjdump-{system}-{arch}-{version}-archive.tar.xz",
)
download_and_copy(
    name="nvdisasm",
    src_func=lambda system, arch, version:
    f"cuda_nvdisasm-{system}-{arch}-{version}-archive/bin/nvdisasm{exe_extension}",
    dst_path="bin/nvdisasm",
    variable="TRITON_NVDISASM_PATH",
    version=NVIDIA_TOOLCHAIN_VERSION["nvdisasm"],
    url_func=lambda system, arch, version:
    f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvdisasm/{system}-{arch}/cuda_nvdisasm-{system}-{arch}-{version}-archive.tar.xz",
)
download_and_copy(
    name="nvcc",
    src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/include",
    dst_path="include",
    variable="TRITON_CUDACRT_PATH",
    version=NVIDIA_TOOLCHAIN_VERSION["cudacrt"],
    url_func=lambda system, arch, version:
    f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
)
download_and_copy(
    name="cudart",
    src_func=lambda system, arch, version: f"cuda_cudart-{system}-{arch}-{version}-archive/include",
    dst_path="include",
    variable="TRITON_CUDART_PATH",
    version=NVIDIA_TOOLCHAIN_VERSION["cudart"],
    url_func=lambda system, arch, version:
    f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/{system}-{arch}/cuda_cudart-{system}-{arch}-{version}-archive.tar.xz",
)
download_and_copy(
    name="cupti",
    src_func=lambda system, arch, version: f"cuda_cupti-{system}-{arch}-{version}-archive/include",
    dst_path="include",
    variable="TRITON_CUPTI_INCLUDE_PATH",
    version=NVIDIA_TOOLCHAIN_VERSION["cupti"],
    url_func=lambda system, arch, version:
    f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/{system}-{arch}/cuda_cupti-{system}-{arch}-{version}-archive.tar.xz",
)
download_and_copy(
    name="cupti",
    src_func=lambda system, arch, version: f"cuda_cupti-{system}-{arch}-{version}-archive/lib",
    dst_path="lib/cupti",
    variable="TRITON_CUPTI_LIB_PATH",
    version=NVIDIA_TOOLCHAIN_VERSION["cupti"],
    url_func=lambda system, arch, version:
    f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/{system}-{arch}/cuda_cupti-{system}-{arch}-{version}-archive.tar.xz",
)
backends = [*BackendInstaller.copy(["nvidia", "amd"]), *BackendInstaller.copy_externals()]


def add_link_to_backends():
    for backend in backends:
        update_symlink(backend.install_dir, backend.backend_dir)

        if backend.language_dir:
            # Link the contents of each backend's `language` directory into
            # `triton.language.extra`.
            extra_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "triton", "language", "extra"))
            for x in os.listdir(backend.language_dir):
                src_dir = os.path.join(backend.language_dir, x)
                install_dir = os.path.join(extra_dir, x)
                update_symlink(install_dir, src_dir)

        if backend.tools_dir:
            # Link the contents of each backend's `tools` directory into
            # `triton.tools.extra`.
            extra_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "triton", "tools", "extra"))
            for x in os.listdir(backend.tools_dir):
                src_dir = os.path.join(backend.tools_dir, x)
                install_dir = os.path.join(extra_dir, x)
                update_symlink(install_dir, src_dir)


def add_link_to_proton():
    proton_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "third_party", "proton", "proton"))
    proton_install_dir = os.path.join(os.path.dirname(__file__), "triton", "profiler")
    update_symlink(proton_install_dir, proton_dir)


def add_links():
    add_link_to_backends()
    if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
        add_link_to_proton()


class plugin_install(install):

    def run(self):
        add_links()
        install.run(self)


class plugin_develop(develop):

    def run(self):
        add_links()
        develop.run(self)


class plugin_bdist_wheel(bdist_wheel):

    def run(self):
        add_links()
        bdist_wheel.run(self)


class plugin_egginfo(egg_info):

    def run(self):
        add_links()
        egg_info.run(self)


package_data = {
    "triton/tools/extra": sum((b.tools_package_data for b in backends), []),
    **{f"triton/backends/{b.name}": b.package_data
       for b in backends}, "triton/language/extra": sum((b.language_package_data for b in backends), [])
}


def get_extra_packages(extra_name):
    packages = []
    extra_file_extensions = {"language": (".py"), "tools": (".c", ".h", ".cpp")}
    assert extra_name in extra_file_extensions, f"{extra_name} extra is not valid"

    for backend in backends:
        backend_extra_dir = getattr(backend, f"{extra_name}_dir", None)
        if backend_extra_dir is None:
            continue

        # Walk the specified directory of each backend to enumerate
        # any subpackages, which will be added to extra_package.
        for dir, dirs, files in os.walk(backend_extra_dir, followlinks=True):
            if not any(f for f in files if f.endswith(extra_file_extensions[extra_name])) or dir == backend_extra_dir:
                # Ignore directories with no relevant files
                # or the root directory
                continue
            subpackage = os.path.relpath(dir, backend_extra_dir)
            package = os.path.join(f"triton/{extra_name}/extra", subpackage)
            packages.append(package)

    return list(packages)


def get_packages():
    packages = [
        "triton",
        "triton/_C",
        "triton/compiler",
        "triton/language",
        "triton/language/extra",
        "triton/runtime",
        "triton/backends",
        "triton/tools",
        "triton/tools/extra",
    ]
    packages += [f'triton/backends/{backend.name}' for backend in backends]
    packages += get_extra_packages("language")
    packages += get_extra_packages("tools")
    if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
        packages += ["triton/profiler"]

    return packages


def get_entry_points():
    entry_points = {}
    if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
        entry_points["console_scripts"] = [
            "proton-viewer = triton.profiler.viewer:main",
            "proton = triton.profiler.proton:main",
        ]
    return entry_points


def get_git_commit_hash(length=8):
    try:
        cmd = ['git', 'rev-parse', f'--short={length}', 'HEAD']
        return "+git{}".format(subprocess.check_output(cmd).strip().decode('utf-8'))
    except Exception:
        return ""


setup(
    name=os.environ.get("TRITON_WHEEL_NAME", "triton"),
    version="3.2.0" + get_git_commit_hash() + os.environ.get("TRITON_WHEEL_VERSION_SUFFIX", ""),
    author="Philippe Tillet",
    author_email="phil@openai.com",
    description="A language and compiler for custom Deep Learning operations",
    long_description="",
    install_requires=["setuptools>=40.8.0"],
    packages=get_packages(),
    entry_points=get_entry_points(),
    package_data=package_data,
    include_package_data=True,
    ext_modules=[CMakeExtension("triton", "triton/_C/")],
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": CMakeBuildPy,
        "clean": CMakeClean,
        "install": plugin_install,
        "develop": plugin_develop,
        "bdist_wheel": plugin_bdist_wheel,
        "egg_info": plugin_egginfo,
    },
    zip_safe=False,
    # for PyPI
    keywords=["Compiler", "Deep Learning"],
    url="https://github.com/triton-lang/triton/",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    test_suite="tests",
    extras_require={
        "build": [
            "cmake>=3.20",
            "lit",
        ],
        "tests": [
            "autopep8",
            "isort",
            "numpy",
            "pytest",
            "pytest-forked",
            "pytest-xdist",
            "scipy>=1.7.1",
            "llnl-hatchet",
        ],
        "tutorials": [
            "matplotlib",
            "pandas",
            "tabulate",
        ],
    },
)
