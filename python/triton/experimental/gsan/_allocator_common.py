"""Build and handle types shared by the independent CUDA allocators."""
from enum import IntEnum
import hashlib
from pathlib import Path

from triton.runtime import driver as runtime_driver
from triton.runtime.build import compile_module_from_file


class ShareableHandleType(IntEnum):
    POSIX_FILE_DESCRIPTOR = 0x1
    FABRIC = 0x8


def load_allocator(source: str, name: str, *, torch_integration: bool = False, extra_sources=()):
    # Loading the direct allocator must not initialize PyTorch CUDA.
    if not torch_integration and runtime_driver.active.get_current_target().backend != "cuda":
        raise RuntimeError("GSan allocators require the CUDA backend.")
    from triton.backends.nvidia.driver import library_dirs, include_dirs
    directory = Path(__file__).resolve().parent / "src"
    # compile_module_from_file hashes the source but not included headers.
    # Invalidate both allocators when the shared helpers or layouts change.
    digest = hashlib.sha256()
    for header in sorted(p for p in directory.iterdir() if p.suffix in (".h", ".cuh")):
        digest.update(header.name.encode())
        digest.update(header.read_bytes())
    libraries = ["libcuda.so.1"]
    library_paths = library_dirs()
    includes = list(include_dirs)
    flags = [str(directory / extra_source) for extra_source in extra_sources]
    for extra_source in extra_sources:
        digest.update((directory / extra_source).read_bytes())
    if torch_integration:
        import torch
        if torch.version.cuda is None:
            raise RuntimeError("GSan allocators require a CUDA build of PyTorch")
        torch_path = Path(torch.__file__).parent
        torch_lib = str(torch_path / "lib")
        includes.append(str(torch_path / "include"))
        library_paths = [*library_paths, torch_lib]
        libraries.extend(["c10_cuda", "c10"])
        flags.extend([f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", f"-Wl,-rpath,{torch_lib}"])
        # Some PyTorch distributions extend the virtual allocator interface.
        # Detect those declarations when compiling against their own headers.
        interface = (torch_path / "include/c10/cuda/CUDACachingAllocator.h").read_text()
        if "attachCudaAllocObserver" in interface:
            flags.append("-DTRITON_TORCH_EXTENDED_ALLOCATOR=1")
        from torch.utils.cpp_extension import include_paths, library_paths as torch_library_paths
        includes.extend(include_paths(device_type="cuda"))
        library_paths.extend(torch_library_paths(device_type="cuda"))
        libraries.append("cudart")
        for key in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI"):
            value = getattr(torch._C, f"_PYBIND11_{key}", None)
            if value is not None:
                flags.append(f'-DPYBIND11_{key}="{value}"')
        digest.update(interface.encode())
        digest.update(str(torch.__version__).encode())
        digest.update(str(torch.version.git_version).encode())
    return compile_module_from_file(
        src_path=str(directory / source),
        name=name,
        library_dirs=library_paths,
        include_dirs=includes,
        libraries=libraries,
        ccflags=[*flags, f"-DTRITON_ALLOCATOR_HEADERS_HASH={digest.hexdigest()}"],
    )
