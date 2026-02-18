import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.clean import clean
from pathlib import Path
from typing import Optional

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install
from setuptools.command.sdist import sdist

from dataclasses import dataclass

import pybind11

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel

try:
    from setuptools.command.editable_wheel import editable_wheel
except ImportError:
    # create a dummy class, since there is no command to override
    class editable_wheel:
        pass


sys.path.insert(0, os.path.dirname(__file__))

from python.build_helpers import get_base_dir, get_cmake_dir


def is_git_repo():
    """Return True if this file resides in a git repository"""
    return (Path(__file__).parent / ".git").is_dir()


@dataclass
class Backend:
    name: str
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
            root_dir = "third_party"
            assert backend_name in os.listdir(
                root_dir), f"{backend_name} is requested for install but not present in {root_dir}"

            if is_git_repo():
                try:
                    subprocess.run(["git", "submodule", "update", "--init", f"{backend_name}"], check=True,
                                   stdout=subprocess.DEVNULL, cwd=root_dir)
                except subprocess.CalledProcessError:
                    pass
                except FileNotFoundError:
                    pass

            backend_src_dir = os.path.join(root_dir, backend_name)

        backend_path = os.path.join(backend_src_dir, "backend")
        assert os.path.exists(backend_path), f"{backend_path} does not exist!"

        language_dir = os.path.join(backend_src_dir, "language")
        if not os.path.exists(language_dir):
            language_dir = None

        tools_dir = os.path.join(backend_src_dir, "tools")
        if not os.path.exists(tools_dir):
            tools_dir = None

        for file in ["compiler.py", "driver.py"]:
            assert os.path.exists(os.path.join(backend_path, file)), f"${file} does not exist in ${backend_path}"

        install_dir = os.path.join(os.path.dirname(__file__), "python", "triton", "backends", backend_name)

        return Backend(name=backend_name, src_dir=backend_src_dir, backend_dir=backend_path, language_dir=language_dir,
                       tools_dir=tools_dir, install_dir=install_dir, is_external=is_external)

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
    link_path.symlink_to(source_path.absolute(), target_is_directory=True)


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
        if (cmake_major, cmake_minor) < (3, 20):
            raise RuntimeError("CMake >= 3.20 is required")

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
        cmake_args = self.get_pybind11_cmake_args()
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
        assert ninja_dir is not None, "ninja not found!"
        thirdparty_cmake_args = self.get_pybind11_cmake_args()
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        wheeldir = os.path.dirname(extdir)

        # create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # python directories
        python_include_dir = sysconfig.get_path("platinclude")
        cmake_args = [
            "-G",
            "Ninja",  # Ninja is much faster than make
            "-DCMAKE_MAKE_PROGRAM=" +
            ninja_dir,  # Pass explicit path to ninja otherwise cmake may cache a temporary path
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DLLVM_ENABLE_WERROR=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DTRITON_BUILD_PYTHON_MODULE=ON",
            "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DPython3_INCLUDE_DIR=" + python_include_dir,
            "-DTRITON_CODEGEN_BACKENDS=" + ';'.join([b.name for b in backends if not b.is_external]),
            "-DTRITON_PLUGIN_DIRS=" + ';'.join([b.src_dir for b in backends if b.is_external]),
            "-DTRITON_WHEEL_DIR=" + wheeldir,
            f"-DTRITON_CACHE_PATH={get_triton_cache_path()}",
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

        if check_env_flag("LLVM_BUILD_SHARED_LIBS"):
            cmake_args += ["-DLLVM_BUILD_SHARED_LIBS=1"]
        else:
            cmake_args += ["-DLLVM_BUILD_SHARED_LIBS=0"]

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
            "TRITON_BUILD_WITH_CCACHE",
            "TRITON_PARALLEL_LINK_JOBS",
            "TRITON_OFFLINE_BUILD",
            "TRITON_LLVM_SYSTEM_SUFFIX",
            "LLVM_SYSPATH",
            "JSON_SYSPATH",
            "TRITON_CUDACRT_PATH",
            "TRITON_CUDART_PATH",
            "TRITON_CUOBJDUMP_PATH",
            "TRITON_CUPTI_INCLUDE_PATH",
            "TRITON_CUPTI_LIB_PATH",
            "TRITON_CUPTI_LIB_BLACKWELL_PATH",
            "TRITON_NVDISASM_PATH",
            "TRITON_PTXAS_PATH",
            "TRITON_PTXAS_BLACKWELL_PATH",
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
        update_symlink(Path(self.base_dir) / "compile_commands.json", cmake_dir / "compile_commands.json")
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=cmake_dir)
        subprocess.check_call(["cmake", "--build", ".", "--target", "mlir-doc"], cwd=cmake_dir)


backends = [*BackendInstaller.copy(["nvidia", "amd"]), *BackendInstaller.copy_externals()]


def get_package_dirs():
    yield ("", "python")

    for backend in backends:
        # we use symlinks for external plugins
        if backend.is_external:
            continue

        yield (f"triton.backends.{backend.name}", backend.backend_dir)

        if backend.language_dir:
            # Install the contents of each backend's `language` directory into
            # `triton.language.extra`.
            for x in os.listdir(backend.language_dir):
                yield (f"triton.language.extra.{x}", os.path.join(backend.language_dir, x))

        if backend.tools_dir:
            # Install the contents of each backend's `tools` directory into
            # `triton.tools.extra`.
            for x in os.listdir(backend.tools_dir):
                yield (f"triton.tools.extra.{x}", os.path.join(backend.tools_dir, x))

    if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
        yield ("triton.profiler", "third_party/proton/proton")
        yield ("triton.profiler.hooks", "third_party/proton/proton/hooks")


def get_packages():
    yield from find_packages(where="python")

    for backend in backends:
        yield f"triton.backends.{backend.name}"

        if backend.language_dir:
            # Install the contents of each backend's `language` directory into
            # `triton.language.extra`.
            for x in os.listdir(backend.language_dir):
                yield f"triton.language.extra.{x}"

        if backend.tools_dir:
            # Install the contents of each backend's `tools` directory into
            # `triton.tools.extra`.
            for x in os.listdir(backend.tools_dir):
                yield f"triton.tools.extra.{x}"

    if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
        yield "triton.profiler"


def add_link_to_backends(external_only):
    for backend in backends:
        if external_only and not backend.is_external:
            continue

        update_symlink(backend.install_dir, backend.backend_dir)

        if backend.language_dir:
            # Link the contents of each backend's `language` directory into
            # `triton.language.extra`.
            extra_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "python", "triton", "language",
                                                     "extra"))
            for x in os.listdir(backend.language_dir):
                src_dir = os.path.join(backend.language_dir, x)
                install_dir = os.path.join(extra_dir, x)
                update_symlink(install_dir, src_dir)

        if backend.tools_dir:
            # Link the contents of each backend's `tools` directory into
            # `triton.tools.extra`.
            extra_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "python", "triton", "tools", "extra"))
            for x in os.listdir(backend.tools_dir):
                src_dir = os.path.join(backend.tools_dir, x)
                install_dir = os.path.join(extra_dir, x)
                update_symlink(install_dir, src_dir)


def add_link_to_proton():
    proton_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "third_party", "proton", "proton"))
    proton_install_dir = os.path.join(os.path.dirname(__file__), "python", "triton", "profiler")
    update_symlink(proton_install_dir, proton_dir)


def add_links(external_only):
    add_link_to_backends(external_only=external_only)
    if not external_only and check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
        add_link_to_proton()


class plugin_bdist_wheel(bdist_wheel):

    def run(self):
        add_links(external_only=True)
        super().run()


class plugin_develop(develop):

    def run(self):
        add_links(external_only=False)
        super().run()


class plugin_editable_wheel(editable_wheel):

    def run(self):
        add_links(external_only=False)
        super().run()


class plugin_egg_info(egg_info):

    def run(self):
        add_links(external_only=True)
        super().run()


class plugin_install(install):

    def run(self):
        add_links(external_only=True)
        super().run()


class plugin_sdist(sdist):

    def run(self):
        for backend in backends:
            if backend.is_external:
                raise RuntimeError("sdist cannot be used with TRITON_PLUGIN_DIRS")
        super().run()


def get_entry_points():
    entry_points = {}
    if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
        entry_points["console_scripts"] = [
            "proton-viewer = triton.profiler.viewer:main",
            "proton = triton.profiler.proton:main",
        ]
    entry_points["triton.backends"] = [f"{b.name} = triton.backends.{b.name}" for b in backends]
    return entry_points


def get_git_commit_hash(length=8):
    try:
        cmd = ['git', 'rev-parse', f'--short={length}', 'HEAD']
        return "+git{}".format(subprocess.check_output(cmd).strip().decode('utf-8'))
    except Exception:
        return ""


def get_git_branch():
    try:
        cmd = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        return subprocess.check_output(cmd).strip().decode('utf-8')
    except Exception:
        return ""


def get_git_version_suffix():
    if not is_git_repo():
        return ""  # Not a git checkout
    branch = get_git_branch()
    if branch.startswith("release"):
        return ""
    else:
        return get_git_commit_hash()


def get_triton_version_suffix():
    # Either "" or "+<githash>", "<githash>" itself does not contain any plus-characters.
    git_sfx = get_git_version_suffix()
    # Should start with "+" that will replaced with "-" if needed
    env_sfx = os.environ.get("TRITON_WHEEL_VERSION_SUFFIX", "")
    # version suffix can only contain one plus-character
    if "+" in git_sfx and "+" in env_sfx:
        env_sfx = env_sfx.replace("+", "-")
    return git_sfx + env_sfx


# keep it separate for easy substitution
TRITON_VERSION = "3.6.0" + get_triton_version_suffix()

# Dynamically define supported Python versions and classifiers
MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 14)

PYTHON_REQUIRES = f">={MIN_PYTHON[0]}.{MIN_PYTHON[1]},<{MAX_PYTHON[0]}.{MAX_PYTHON[1] + 1}"
BASE_CLASSIFIERS = [
    "Development Status :: 4 - Beta", "Intended Audience :: Developers", "Topic :: Software Development :: Build Tools"
]
PYTHON_CLASSIFIERS = [
    f"Programming Language :: Python :: {MIN_PYTHON[0]}.{m}" for m in range(MIN_PYTHON[1], MAX_PYTHON[1] + 1)
]
CLASSIFIERS = BASE_CLASSIFIERS + PYTHON_CLASSIFIERS

setup(
    name=os.environ.get("TRITON_WHEEL_NAME", "triton"),
    version=TRITON_VERSION,
    author="Philippe Tillet",
    author_email="phil@openai.com",
    description="A language and compiler for custom Deep Learning operations",
    long_description="",
    license="MIT",
    install_requires=[
        "importlib-metadata; python_version < '3.10'",
    ],
    packages=list(get_packages()),
    package_dir=dict(get_package_dirs()),
    entry_points=get_entry_points(),
    include_package_data=True,
    exclude_package_data={
        "": [
            "__pycache__",
            "__pycache__/*",
            "*.py[cod]",
        ]
    },
    ext_modules=[CMakeExtension("triton", "triton/_C/")],
    cmdclass={
        "bdist_wheel": plugin_bdist_wheel,
        "build_ext": CMakeBuild,
        "build_py": CMakeBuildPy,
        "clean": CMakeClean,
        "develop": plugin_develop,
        "editable_wheel": plugin_editable_wheel,
        "egg_info": plugin_egg_info,
        "install": plugin_install,
        "sdist": plugin_sdist,
    },
    zip_safe=False,
    # for PyPI
    keywords=["Compiler", "Deep Learning"],
    url="https://github.com/triton-lang/triton/",
    python_requires=PYTHON_REQUIRES,
    classifiers=CLASSIFIERS,
    extras_require={
        "build": [
            "cmake>=3.20,<4.0",
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
