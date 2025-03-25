import os
import shutil
import sys
import functools
import tarfile
import zipfile
from io import BytesIO
import urllib.request
from pathlib import Path
import hashlib
from dataclasses import dataclass

use_triton_shared = True
necessary_third_party = ["triton_shared"]
default_backends = ["nvidia", "amd"]
extend_backends = []
ext_sourcedir = "triton/_C/"
flagtree_backend = os.getenv("FLAGTREE_BACKEND", "").lower()
flagtree_plugin = os.getenv("FLAGTREE_PLUGIN", "").lower()


@dataclass
class FlagTreeBackend:
    name: str
    url: str
    tag: str


flagtree_backend_info = {
    "triton_shared":
    FlagTreeBackend(name="triton_shared", url="https://github.com/microsoft/triton-shared.git",
                    tag="380b87122c88af131530903a702d5318ec59bb33"),
    "cambricon":
    FlagTreeBackend(name="cambricon", url="https://github.com/Cambricon/triton-linalg.git",
                    tag="00f51c2e48a943922f86f03d58e29f514def646d"),
}

set_llvm_env = lambda path: set_env({
    'LLVM_BUILD_DIR': path,
    'LLVM_INCLUDE_DIRS': Path(path) / "include",
    'LLVM_LIBRARY_DIR': Path(path) / "lib",
    'LLVM_SYSPATH': path,
})


class FlagTreeCache:

    def __init__(self):
        self.flagtree_dir = os.path.dirname(os.getcwd())
        self.dir_name = ".flagtree"
        self.sub_dirs = {}
        self.cache_files = {}
        self.dir_path = self._get_cache_dir_path()
        self._create_cache_dir()
        if flagtree_backend:
            self._create_subdir(subdir_name=flagtree_backend)

    @functools.lru_cache(maxsize=None)
    def _get_cache_dir_path(self) -> Path:
        _cache_dir = os.environ.get("FLAGTREE_CACHE_DIR")
        if _cache_dir is None:
            _cache_dir = Path.home() / self.dir_name
        else:
            _cache_dir = Path(_cache_dir)
        return _cache_dir

    def _create_cache_dir(self) -> Path:
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path, exist_ok=True)

    def _create_subdir(self, subdir_name, path=None):
        if path is None:
            subdir_path = Path(self.dir_path) / subdir_name
        else:
            subdir_path = Path(path) / subdir_name

        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
        self.sub_dirs[subdir_name] = subdir_path

    def _md5(self, file_path):
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as file:
            while chunk := file.read(4096):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _download(self, url, path, file_name):
        MAX_RETRY_COUNT = 4
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0'
        headers = {
            'User-Agent': user_agent,
        }
        request = urllib.request.Request(url, None, headers)
        retry_count = MAX_RETRY_COUNT
        content = None
        print(f'downloading {url} ...')
        while (retry_count):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    content = response.read()
                    break
            except Exception:
                retry_count -= 1
                print(f"\n[{MAX_RETRY_COUNT - retry_count}] retry to downloading and extracting {url}")

        if retry_count == 0:
            raise RuntimeError("The download failed, probably due to network problems")

        print(f'extracting {url} ...')
        file_bytes = BytesIO(content)
        file_names = []
        if url.endswith(".zip"):
            with zipfile.ZipFile(file_bytes, "r") as file:
                file.extractall(path=path)
                file_names = file.namelist()
        else:
            with tarfile.open(fileobj=file_bytes, mode="r|*") as file:
                file.extractall(path=path)
                file_names = file.getnames()
        os.rename(Path(path) / file_names[0], Path(path) / file_name)

    def check_file(self, file_name=None, url=None, path=None, md5_digest=None):
        origin_file_path = None
        if url is not None:
            origin_file_name = url.split("/")[-1].split('.')[0]
            origin_file_path = self.cache_files.get(origin_file_name, "")
        if path is not None:
            _path = path
        else:
            _path = self.cache_files.get(file_name, "")
        empty = (not os.path.exists(_path)) or (origin_file_path and not os.path.exists(origin_file_path))
        if empty:
            return False
        if md5_digest is None:
            return True
        else:
            cur_md5 = self._md5(_path)
            return cur_md5[:8] == md5_digest

    def clear(self):
        shutil.rmtree(self.dir_path)

    def reverse_copy(self, src_path, cache_file_path, md5_digest):
        if src_path is None or not os.path.exists(src_path):
            return False
        if os.path.exists(cache_file_path):
            return False
        copy_needed = True
        if md5_digest is None or self._md5(src_path) == md5_digest:
            copy_needed = False
        if copy_needed:
            print(f"copying {src_path} to {cache_file_path}")
            if os.path.isdir(src_path):
                shutil.copytree(src_path, cache_file_path, dirs_exist_ok=True)
            else:
                shutil.copy(src_path, cache_file_path)
            return True
        return False

    def store(self, file=None, condition=None, url=None, copy_src_path=None, copy_dst_path=None, files=None,
              md5_digest=None, pre_hock=None, post_hock=None):

        if not condition or (pre_hock and pre_hock()):
            return
        is_url = False if url is None else True
        path = self.sub_dirs[flagtree_backend] if flagtree_backend else self.dir_path

        if files is not None:
            for single_files in files:
                self.cache_files[single_files] = Path(path) / single_files
        else:
            self.cache_files[file] = Path(path) / file
            if url is not None:
                origin_file_name = url.split("/")[-1].split('.')[0]
                self.cache_files[origin_file_name] = Path(path) / file
            if copy_dst_path is not None:
                dst_path_root = Path(self.flagtree_dir) / copy_dst_path
                dst_path = Path(dst_path_root) / file
                if self.reverse_copy(dst_path, self.cache_files[file], md5_digest):
                    return

        if is_url and not self.check_file(file_name=file, url=url, md5_digest=md5_digest):
            self._download(url, path, file_name=file)

        if copy_dst_path is not None:
            file_lists = [file] if files is None else list(files)
            for single_file in file_lists:
                dst_path_root = Path(self.flagtree_dir) / copy_dst_path
                os.makedirs(dst_path_root, exist_ok=True)
                dst_path = Path(dst_path_root) / single_file
                if not self.check_file(path=dst_path, md5_digest=md5_digest):
                    if copy_src_path:
                        src_path = Path(copy_src_path) / single_file
                    else:
                        src_path = self.cache_files[single_file]
                    print(f"copying {src_path} to {dst_path}")
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy(src_path, dst_path)
        post_hock(self.cache_files[file]) if post_hock else False

    def get(self, file_name) -> Path:
        return self.cache_files[file_name]


class CommonUtils:

    @staticmethod
    def unlink():
        cur_path = os.path.dirname(__file__)
        backends_dir_path = Path(cur_path) / "triton" / "backends"
        if not os.path.exists(backends_dir_path):
            return
        for name in os.listdir(backends_dir_path):
            exist_backend_path = os.path.join(backends_dir_path, name)
            if not os.path.isdir(exist_backend_path):
                continue
            if name.startswith('__'):
                continue
            if os.path.islink(exist_backend_path):
                os.unlink(exist_backend_path)
            if os.path.exists(exist_backend_path):
                shutil.rmtree(exist_backend_path)

    @staticmethod
    def skip_package_dir(package):
        if 'backends' in package or 'profiler' in package:
            return True
        if flagtree_backend in ['cambricon']:
            if package not in ['triton', 'triton/_C']:
                return True
        return False

    @staticmethod
    def get_package_dir(packages):
        package_dict = {}
        if flagtree_backend and flagtree_backend != 'cambricon':
            connection = []
            backend_triton_path = f"../third_party/{flagtree_backend}/python/"
            for package in packages:
                if CommonUtils.skip_package_dir(package):
                    continue
                pair = (package, f"{backend_triton_path}{package}")
                connection.append(pair)
            package_dict.update(connection)
        return package_dict

    @staticmethod
    def download_third_party():
        import git
        MAX_RETRY = 4
        global use_triton_shared, flagtree_backend
        third_party_base_dir = Path(os.path.dirname(os.path.dirname(__file__))) / "third_party"

        def git_clone(lib, lib_path):
            global use_triton_shared
            print(f"Clone {lib.name} into {lib_path} ...")
            retry_count = MAX_RETRY
            while (retry_count):
                try:
                    repo = git.Repo.clone_from(lib.url, lib_path)
                    repo.git.checkout(lib.tag)
                    if lib.name in flagtree_backend_info:
                        sub_triton_path = Path(lib_path) / "triton"
                        if os.path.exists(sub_triton_path):
                            shutil.rmtree(sub_triton_path)
                    print(f"successfully clone {lib.name} into {lib_path} ...")
                    return
                except Exception:
                    retry_count -= 1
                    print(f"\n[{MAX_RETRY - retry_count}] retry to clone {lib.name} to  {lib_path}")

            print(f"Unable to clone third_party {lib.name}")
            if lib.name in necessary_third_party:
                use_triton_shared = False
                print("\n\ttriton_shared is compiled by default, but for "
                      "some reason we couldn't download triton_shared\n"
                      "as third_party (most likely for network reasons), "
                      "so we couldn't compile triton_shared\n")

        third_partys = []
        if os.environ.get("USE_TRITON_SHARED", "ON") == "ON" and not flagtree_backend:
            third_partys.append(flagtree_backend_info["triton_shared"])
        else:
            use_triton_shared = False
        if flagtree_backend in flagtree_backend_info:
            third_partys.append(flagtree_backend_info[flagtree_backend])

        for lib in third_partys:
            lib_path = Path(third_party_base_dir) / lib.name
            if not os.path.exists(lib_path):
                git_clone(lib=lib, lib_path=lib_path)
            else:
                print(f'Found third_party {lib.name} at {lib_path}\n')


def handle_flagtree_backend():
    global ext_sourcedir
    if flagtree_backend:
        print(f"flagtree_backend is {flagtree_backend}")
        extend_backends.append(flagtree_backend)
    if "editable_wheel" in sys.argv:
        ext_sourcedir = os.path.abspath(f"../third_party/{flagtree_backend}/python/{ext_sourcedir}") + "/"
    if use_triton_shared and not flagtree_backend:
        default_backends.append("triton_shared")


def set_env(env_dict: dict):
    for env_k, env_v in env_dict.items():
        os.environ[env_k] = str(env_v)


def check_env(env_val):
    return os.environ.get(env_val, '') != ''


CommonUtils.download_third_party()
handle_flagtree_backend()
cache = FlagTreeCache()

# iluvatar
cache.store(
    file="iluvatarTritonPlugin.so", condition=("iluvatar" == flagtree_backend) and (flagtree_plugin == ''), url=
    "https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/iluvatarTritonPlugin-cpython3.10-glibc2.30-glibcxx3.4.28-cxxabi1.3.12-ubuntu-x86_64.tar.gz",
    copy_dst_path="third_party/iluvatar", md5_digest="7d4e136c")

cache.store(
    file="iluvatar-llvm18-x86_64",
    condition=("iluvatar" == flagtree_backend),
    url="https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/iluvatar-llvm18-x86_64.tar.gz",
    pre_hock=lambda: check_env('LLVM_BUILD_DIR'),
    post_hock=set_llvm_env,
)

# xpu(kunlunxin)
cache.store(
    file="XTDK-llvm18-ubuntu2004_x86_64",
    condition=("xpu" == flagtree_backend),
    url="https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/XTDK-llvm18-ubuntu2004_x86_64.tar",
    pre_hock=lambda: check_env('LLVM_BUILD_DIR'),
    post_hock=set_llvm_env,
)

cache.store(file="xre-Linux-x86_64", condition=("xpu" == flagtree_backend),
            url="https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/xre-Linux-x86_64.tar.gz",
            copy_dst_path='python/_deps/xre3')

cache.store(
    files=("clang", "xpu-xxd", "xpu3-crt.xpu", "xpu-kernel.t", "ld.lld", "llvm-readelf", "llvm-objdump",
           "llvm-objcopy"), condition=("xpu" == flagtree_backend),
    copy_src_path=f"{os.environ.get('LLVM_BUILD_DIR','')}/bin", copy_dst_path="third_party/xpu/backend/xpu3/bin")

cache.store(files=("libclang_rt.builtins-xpu3.a", "libclang_rt.builtins-xpu3s.a"),
            condition=("xpu" == flagtree_backend), copy_src_path=f"{os.environ.get('LLVM_BUILD_DIR','')}/lib/linux",
            copy_dst_path="third_party/xpu/backend/xpu3/lib/linux")

cache.store(files=("include", "so"), condition=("xpu" == flagtree_backend),
            copy_src_path=f"{cache.dir_path}/xpu/xre-Linux-x86_64", copy_dst_path="third_party/xpu/backend/xpu3")

# mthreads
cache.store(
    file="mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64",
    condition=("mthreads" == flagtree_backend),
    url=
    "https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64.tar.gz",
    pre_hock=lambda: check_env('LLVM_BUILD_DIR'),
    post_hock=set_llvm_env,
)
