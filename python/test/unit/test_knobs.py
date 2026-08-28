import os
import pytest
import shutil
import triton
from triton._internal_testing import is_hip

from pathlib import Path


def test_knobs_utils(fresh_knobs) -> None:
    triton.knobs.propagate_env = False

    class test_knobs(triton.knobs.base_knobs):
        foo: triton.knobs.env_str = triton.knobs.env_str("FOO", "triton")
        bar: triton.knobs.env_bool = triton.knobs.env_bool("BAR", True)
        baz: triton.knobs.env_opt_str = triton.knobs.env_opt_str("BAZ")
        quux: triton.knobs.env_opt_bool = triton.knobs.env_opt_bool("QUUX")

    instance = test_knobs()

    # Make sure knobs works
    assert instance.knobs == {
        "foo": "triton",
        "bar": True,
        "baz": None,
        "quux": None,
    }

    # Now make sure copying works properly, otherwise all other tests in this
    # file aren't trustworthy.
    instance.bar = False
    instance.quux = True
    assert instance.foo == "triton"
    assert not instance.bar
    assert instance.baz is None
    assert instance.quux
    assert instance.knobs == {
        "foo": "triton",
        "bar": False,
        "baz": None,
        "quux": True,
    }

    second = instance.copy()
    assert second.foo == "triton"
    assert not second.bar
    assert second.baz is None
    assert second.quux

    second.foo = "tritium"
    assert instance.foo != "tritium"
    assert second.foo == "tritium"

    # Ditto on trustworthiness if reset() doesn't work.
    second.reset()
    assert second.knobs == {
        "foo": "triton",
        "bar": True,
        "baz": None,
        "quux": None,
    }
    # Triple check original instance didn't change.
    assert instance.knobs == {
        "foo": "triton",
        "bar": False,
        "baz": None,
        "quux": True,
    }


@pytest.mark.parametrize(("capability", "enable_fp_fusion", "disable_opt", "disable_optimization"), [
    (80, True, None, False),
    (90, False, "1", True),
    (107, True, "disable-lsr", False),
    (90, True, "0", False),
])
def test_nvidia_codegen_options(capability, enable_fp_fusion, disable_opt, disable_optimization, fresh_knobs,
                                monkeypatch):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import compiler

    calls = []

    def run_codegen(source, triple, processor, features, **kwargs):
        calls.append((source, triple, processor, features, kwargs))
        return ".version 9.0\n.target sm_90\n.visible .entry test_kernel() {}\n"

    monkeypatch.setattr(compiler, "compile_nvptx", run_codegen)
    if disable_opt is not None:
        monkeypatch.setenv("DISABLE_LLVM_OPT", disable_opt)
    else:
        monkeypatch.delenv("DISABLE_LLVM_OPT", raising=False)

    backend = compiler.CUDABackend(GPUTarget("cuda", capability, 32))
    options = compiler.CUDAOptions(ptx_version=90, enable_fp_fusion=enable_fp_fusion)
    metadata = {}
    ptx = backend.make_ptx("test llvm ir", metadata, options, capability)

    assert len(calls) == 1
    source, triple, processor, features, arguments = calls[0]
    llvm_capability = 100 if capability == 107 else capability
    assert source == "test llvm ir"
    assert triple == "nvptx64-nvidia-cuda"
    assert processor == compiler.sm_arch_from_capability(llvm_capability)
    assert features == "+ptx90"
    assert arguments["enable_fp_fusion"] is enable_fp_fusion
    assert arguments["disable_optimization"] is disable_optimization
    assert arguments["disabled_passes"] == ("disable-lsr" if disable_opt == "disable-lsr" else "")
    assert metadata["name"] == "test_kernel"
    assert f".target sm_{capability}" in ptx


def test_nvidia_codegen_path_override(fresh_knobs, monkeypatch):
    from triton.backends.nvidia import compiler

    monkeypatch.delenv("TRITON_NVIDIA_CODEGEN_PATH", raising=False)
    library = Path(compiler.get_nvidia_codegen_path())
    assert library.parent == Path(compiler.__file__).parent / "lib"
    assert "triton_nvidia_codegen" in library.name

    monkeypatch.setenv("TRITON_NVIDIA_CODEGEN_PATH", "/custom/nvidia/codegen.so")
    assert compiler.get_nvidia_codegen_path() == "/custom/nvidia/codegen.so"


def test_nvidia_codegen_revision_invalidates_backend_hash(fresh_knobs, monkeypatch):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import compiler

    monkeypatch.setattr(compiler, "get_ptxas_version", lambda arch: "ptxas version")
    monkeypatch.setattr(compiler, "get_nvidia_codegen_revision", lambda: "first-revision-1")
    backend = compiler.CUDABackend(GPUTarget("cuda", 90, 32))
    first_hash = backend.hash()

    monkeypatch.setattr(compiler, "get_nvidia_codegen_revision", lambda: "second-revision-1")
    backend.hash.cache_clear()
    second_hash = backend.hash()

    assert first_hash != second_hash


@pytest.mark.skipif(is_hip(), reason="NVPTX code generation is unavailable on AMD")
def test_nvidia_codegen_inlines_functions_from_bitcode(fresh_knobs, monkeypatch):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import compiler

    source = '''
target triple = "nvptx64-nvidia-cuda"

define internal i32 @helper(i32 %value) {
  %result = add i32 %value, 1
  ret i32 %result
}

define ptx_kernel void @inline_kernel(ptr addrspace(1) %out, i32 %value) {
  %result = call i32 @helper(i32 %value)
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}
'''
    library = compiler._load_nvidia_codegen(compiler.get_nvidia_codegen_path())
    compile_bitcode = library.triton_nvptx_compile
    inputs = []

    def check_bitcode(bitcode, size, *args):
        inputs.append(bitcode)
        assert bitcode.startswith(b"BC\xc0\xde")
        assert b"\x00" in bitcode
        assert size == len(bitcode)
        return compile_bitcode(bitcode, size, *args)

    monkeypatch.setattr(library, "triton_nvptx_compile", check_bitcode)
    backend = compiler.CUDABackend(GPUTarget("cuda", 90, 32))
    ptx = backend.make_ptx(source, {}, compiler.CUDAOptions(ptx_version=80), 90)

    assert len(inputs) == 1
    assert ".visible .entry inline_kernel" in ptx
    assert "call.uni" not in ptx
    assert "helper(" not in ptx


@pytest.mark.skipif(is_hip(), reason="NVPTX code generation is unavailable on AMD")
def test_nvidia_codegen_reports_invalid_llvm_ir(fresh_knobs):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import compiler

    backend = compiler.CUDABackend(GPUTarget("cuda", 90, 32))
    with pytest.raises(RuntimeError, match="failed to parse LLVM IR.*expected top-level entity"):
        backend.make_ptx("invalid LLVM IR", {}, compiler.CUDAOptions(ptx_version=80), 90)


@pytest.mark.skipif(is_hip(), reason="NVPTX code generation is unavailable on AMD")
def test_nvidia_codegen_uses_short_shared_memory_pointers(fresh_knobs):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import compiler

    source = '''
target triple = "nvptx64-nvidia-cuda"

@global_smem = external addrspace(3) global [0 x i8], align 1

define ptx_kernel void @short_pointer_kernel(ptr addrspace(1) %out, i32 %offset) {
  %shared = getelementptr [0 x i8], ptr addrspace(3) @global_smem, i32 0, i32 %offset
  %value = load i8, ptr addrspace(3) %shared, align 1
  store i8 %value, ptr addrspace(1) %out, align 1
  ret void
}
'''
    backend = compiler.CUDABackend(GPUTarget("cuda", 90, 32))
    ptx = backend.make_ptx(source, {}, compiler.CUDAOptions(ptx_version=80), 90)

    assert any(
        line.startswith("mov.b32") and "global_smem" in line for line in (line.strip() for line in ptx.splitlines()))


def test_knobs_scope(fresh_knobs, monkeypatch):
    fresh_knobs.amd.use_buffer_atomics = True

    # Update env *after* the __set__() does
    monkeypatch.setenv("AMDGCN_USE_BUFFER_ATOMICS", "0")

    assert fresh_knobs.amd.use_buffer_atomics

    # Just to prove that use_buffer_ops is coming from env
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
    assert not fresh_knobs.amd.use_buffer_ops
    monkeypatch.delenv("AMDGCN_USE_BUFFER_OPS")
    assert fresh_knobs.amd.use_buffer_ops

    with fresh_knobs.amd.scope():
        # Use the environment
        del fresh_knobs.amd.use_buffer_atomics
        fresh_knobs.amd.use_buffer_ops = False

        assert not fresh_knobs.amd.use_buffer_atomics
        assert not fresh_knobs.amd.use_buffer_ops

    assert fresh_knobs.amd.use_buffer_atomics
    assert fresh_knobs.amd.use_buffer_ops

    # Just to prove that use_buffer_ops is coming from env
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
    assert not fresh_knobs.amd.use_buffer_ops
    monkeypatch.delenv("AMDGCN_USE_BUFFER_OPS")
    assert fresh_knobs.amd.use_buffer_ops


def test_env_updated(fresh_knobs, monkeypatch):
    fresh_knobs.amd.use_buffer_ops = False
    assert os.getenv("AMDGCN_USE_BUFFER_OPS") == "0"
    # Just triple checking both APIs give us what we expect
    assert os.environ["AMDGCN_USE_BUFFER_OPS"] == "0"

    fresh_knobs.cache.home_dir = "/foo/bar"
    assert os.getenv("TRITON_HOME") == "/foo/bar"
    assert os.environ["TRITON_HOME"] == "/foo/bar"


@pytest.mark.parametrize("truthy, falsey", [("1", "0"), ("true", "false"), ("True", "False"), ("TRUE", "FALSE"),
                                            ("y", "n"), ("YES", "NO"), ("ON", "OFF")])
def test_read_env(truthy, falsey, fresh_knobs_including_libraries, monkeypatch):
    fresh_knobs = fresh_knobs_including_libraries
    # bool defaulting to False
    assert not fresh_knobs.runtime.debug
    # bool defaulting to True
    assert fresh_knobs.language.default_fp_fusion
    # str defaulting to None
    assert fresh_knobs.compilation.use_ir_loc is None
    # str defaulting to not None
    assert fresh_knobs.cache.dir.endswith(".triton/cache")
    # class defaulting to None
    assert fresh_knobs.cache.manager_class is None
    # set[str] defaulting to empty
    assert len(fresh_knobs.build.backend_dirs) == 0

    monkeypatch.setenv("TRITON_DEFAULT_FP_FUSION", falsey)
    monkeypatch.setenv("TRITON_DEBUG", truthy)
    monkeypatch.setenv("USE_IR_LOC", "ttir")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/triton_cache")
    monkeypatch.setenv("TRITON_HOME", "/tmp/triton_home")
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "triton.runtime.cache:FileCacheManager")
    monkeypatch.setenv("TRITON_CUDACRT_PATH", "/tmp/cuda/crt")
    monkeypatch.setenv("TRITON_CUDART_PATH", "/tmp/cuda/rt")

    triton.knobs.refresh_knobs()
    assert fresh_knobs.runtime.debug
    assert not fresh_knobs.language.default_fp_fusion
    assert fresh_knobs.compilation.use_ir_loc == "ttir"
    assert fresh_knobs.cache.home_dir == "/tmp/triton_home"
    assert fresh_knobs.cache.dir == "/tmp/triton_cache"
    assert fresh_knobs.cache.dump_dir == "/tmp/triton_home/.triton/dump"
    assert fresh_knobs.cache.override_dir == "/tmp/triton_home/.triton/override"

    from triton.runtime.cache import FileCacheManager

    assert fresh_knobs.cache.manager_class == FileCacheManager

    assert fresh_knobs.build.backend_dirs == {"/tmp/cuda/crt", "/tmp/cuda/rt"}


def test_triton_home(fresh_knobs, monkeypatch):
    initial_home = fresh_knobs.cache.home_dir
    assert initial_home == os.path.expanduser("~/")
    assert fresh_knobs.cache.dir == os.path.join(initial_home, ".triton/cache")
    assert fresh_knobs.cache.dump_dir == os.path.join(initial_home, ".triton/dump")
    assert fresh_knobs.cache.override_dir == os.path.join(initial_home, ".triton/override")

    monkeypatch.setenv("TRITON_HOME", "/tmp/triton_home")
    assert fresh_knobs.cache.dir == "/tmp/triton_home/.triton/cache"
    assert fresh_knobs.cache.dump_dir == "/tmp/triton_home/.triton/dump"
    assert fresh_knobs.cache.override_dir == "/tmp/triton_home/.triton/override"

    fresh_knobs.cache.home_dir = "/tmp/user/triton_home"
    assert fresh_knobs.cache.dir == "/tmp/user/triton_home/.triton/cache"
    assert fresh_knobs.cache.dump_dir == "/tmp/user/triton_home/.triton/dump"
    assert fresh_knobs.cache.override_dir == "/tmp/user/triton_home/.triton/override"


def test_set_knob_directly(fresh_knobs_including_libraries, monkeypatch):
    fresh_knobs = fresh_knobs_including_libraries
    assert fresh_knobs.cache.dir.endswith(".triton/cache")

    fresh_knobs.cache.dir = "/tmp/triton_cache"
    assert fresh_knobs.cache.dir == "/tmp/triton_cache"

    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/other_triton_cache")
    assert fresh_knobs.cache.dir == "/tmp/triton_cache"

    # Disable propagation to verify resetting/del behavior
    triton.knobs.propagate_env = False

    fresh_knobs.cache.dir = fresh_knobs.env
    assert fresh_knobs.cache.dir == "/tmp/other_triton_cache"

    fresh_knobs.cache.dir = "/tmp/triton_cache"
    fresh_knobs.cache.reset()
    assert fresh_knobs.cache.dir == "/tmp/other_triton_cache"

    triton.knobs.propagate_env = True

    # Just in case, lets check all the other datatypes too
    fresh_knobs.language.default_fp_fusion = False
    fresh_knobs.amd.use_block_pingpong = True
    fresh_knobs.redis.port = 6380
    fresh_knobs.nvidia.mock_ptx_version = "42.0.1"

    from triton.runtime.cache import FileCacheManager

    class TestManagerClass(FileCacheManager):
        pass

    fresh_knobs.cache.manager_class = TestManagerClass

    monkeypatch.setenv("TRITON_CUDART_PATH", "/tmp/the/real/cudart")
    monkeypatch.setenv("TRITON_DEFAULT_FP_FUSION", "1")
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "0")
    monkeypatch.setenv("TRITON_REDIS_PORT", "6381")
    monkeypatch.setenv("TRITON_MOCK_PTX_VERSION", "1.0.0")
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "triton.runtime.cache:FileCacheManager")

    assert not fresh_knobs.language.default_fp_fusion
    assert fresh_knobs.amd.use_block_pingpong
    assert fresh_knobs.redis.port == 6380
    assert fresh_knobs.nvidia.mock_ptx_version == "42.0.1"
    assert fresh_knobs.cache.manager_class == TestManagerClass

    # Make sure both setting `.env` or deleting resets to env vars.
    fresh_knobs.language.default_fp_fusion = fresh_knobs.env
    fresh_knobs.amd.use_block_pingpong = fresh_knobs.env
    fresh_knobs.redis.port = fresh_knobs.env
    del fresh_knobs.nvidia.mock_ptx_version
    del fresh_knobs.cache.manager_class

    assert fresh_knobs.build.backend_dirs == {"/tmp/the/real/cudart"}
    assert fresh_knobs.language.default_fp_fusion
    assert not fresh_knobs.amd.use_block_pingpong
    assert fresh_knobs.redis.port == 6381
    assert fresh_knobs.nvidia.mock_ptx_version == "1.0.0"
    assert fresh_knobs.cache.manager_class == FileCacheManager


@pytest.mark.skipif(
    is_hip(),
    reason="PTXAS is not installed on AMD",
)
def test_nvidia_tool(fresh_knobs, tmp_path, monkeypatch):
    triton_root = Path(fresh_knobs.__file__).parent
    default_ptxas = triton_root / "backends/nvidia/bin/ptxas"

    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options is None

    tmp_ptxas = tmp_path / "ptxas-special"
    shutil.copy(default_ptxas, tmp_ptxas)
    monkeypatch.setenv("TRITON_PTXAS_PATH", str(tmp_ptxas))
    monkeypatch.setenv("PTXAS_OPTIONS", "--verbose")
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options == "--verbose"

    # Don't prop so that the `del` is correctly tested
    fresh_knobs.propagate_env = False
    fresh_knobs.nvidia.ptxas = str(default_ptxas)
    fresh_knobs.nvidia.ptxas_options = "--device-debug"
    fresh_knobs.propagate_env = True
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options == "--device-debug"

    del fresh_knobs.nvidia.ptxas
    del fresh_knobs.nvidia.ptxas_options
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options == "--verbose"

    # Triple check scope works
    with fresh_knobs.nvidia.scope():
        fresh_knobs.nvidia.ptxas = str(default_ptxas)
        fresh_knobs.nvidia.ptxas_options = "--device-debug"
        assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()
        assert fresh_knobs.nvidia.ptxas_options == "--device-debug"

    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options == "--verbose"

    monkeypatch.delenv("TRITON_PTXAS_PATH")
    monkeypatch.delenv("PTXAS_OPTIONS")
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()
    assert fresh_knobs.nvidia.ptxas_options is None


def test_opt_bool(fresh_knobs_including_libraries, monkeypatch):
    fresh_knobs = fresh_knobs_including_libraries
    assert fresh_knobs.amd.use_block_pingpong is None
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "0")
    assert not fresh_knobs.amd.use_block_pingpong
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "1")
    assert fresh_knobs.amd.use_block_pingpong
    monkeypatch.delenv("TRITON_HIP_USE_BLOCK_PINGPONG")
    assert fresh_knobs.amd.use_block_pingpong is None
