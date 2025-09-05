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


def test_knobs_scope(fresh_knobs, monkeypatch):
    fresh_knobs.amd.global_prefetch = 4
    fresh_knobs.amd.local_prefetch = 3

    # Update env *after* the __set__() does
    monkeypatch.setenv("TRITON_HIP_LOCAL_PREFETCH", "17")

    assert fresh_knobs.amd.global_prefetch == 4
    assert fresh_knobs.amd.local_prefetch == 3

    # Just to prove that use_buffer_ops is coming from env
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
    assert not fresh_knobs.amd.use_buffer_ops

    with fresh_knobs.amd.scope():
        fresh_knobs.amd.global_prefetch = 5
        # Use the environment
        del fresh_knobs.amd.local_prefetch
        fresh_knobs.amd.use_buffer_ops = False

        assert fresh_knobs.amd.global_prefetch == 5
        assert fresh_knobs.amd.local_prefetch == 17
        assert not fresh_knobs.amd.use_buffer_ops

    assert fresh_knobs.amd.global_prefetch == 4
    assert fresh_knobs.amd.local_prefetch == 3

    # Just to prove that use_buffer_ops is coming from env
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
    assert not fresh_knobs.amd.use_buffer_ops


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
def test_read_env(truthy, falsey, fresh_knobs, monkeypatch):
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


def test_set_knob_directly(fresh_knobs, monkeypatch):
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
    fresh_knobs.amd.global_prefetch = 5
    fresh_knobs.nvidia.mock_ptx_version = "42.0.1"

    from triton.runtime.cache import FileCacheManager

    class TestManagerClass(FileCacheManager):
        pass

    fresh_knobs.cache.manager_class = TestManagerClass

    monkeypatch.setenv("TRITON_CUDART_PATH", "/tmp/the/real/cudart")
    monkeypatch.setenv("TRITON_DEFAULT_FP_FUSION", "1")
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "0")
    monkeypatch.setenv("TRITON_HIP_GLOBAL_PREFETCH", "2")
    monkeypatch.setenv("TRITON_MOCK_PTX_VERSION", "1.0.0")
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "triton.runtime.cache:FileCacheManager")

    assert not fresh_knobs.language.default_fp_fusion
    assert fresh_knobs.amd.use_block_pingpong
    assert fresh_knobs.amd.global_prefetch == 5
    assert fresh_knobs.nvidia.mock_ptx_version == "42.0.1"
    assert fresh_knobs.cache.manager_class == TestManagerClass

    # Make sure both setting `.env` or deleting resets to env vars.
    fresh_knobs.language.default_fp_fusion = fresh_knobs.env
    fresh_knobs.amd.use_block_pingpong = fresh_knobs.env
    fresh_knobs.amd.global_prefetch = fresh_knobs.env
    del fresh_knobs.nvidia.mock_ptx_version
    del fresh_knobs.cache.manager_class

    assert fresh_knobs.build.backend_dirs == {"/tmp/the/real/cudart"}
    assert fresh_knobs.language.default_fp_fusion
    assert not fresh_knobs.amd.use_block_pingpong
    assert fresh_knobs.amd.global_prefetch == 2
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

    tmp_ptxas = tmp_path / "ptxas-special"
    shutil.copy(default_ptxas, tmp_ptxas)
    monkeypatch.setenv("TRITON_PTXAS_PATH", str(tmp_ptxas))
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()

    # Don't prop so that the `del` is correctly tested
    fresh_knobs.propagate_env = False
    fresh_knobs.nvidia.ptxas = str(default_ptxas)
    fresh_knobs.propagate_env = True
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()

    del fresh_knobs.nvidia.ptxas
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()

    # Triple check scope works
    with fresh_knobs.nvidia.scope():
        fresh_knobs.nvidia.ptxas = str(default_ptxas)
        assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()

    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()

    monkeypatch.delenv("TRITON_PTXAS_PATH")
    assert Path(fresh_knobs.nvidia.ptxas.path).resolve() == default_ptxas.resolve()


def test_opt_bool(fresh_knobs, monkeypatch):
    assert fresh_knobs.amd.use_block_pingpong is None
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "0")
    assert not fresh_knobs.amd.use_block_pingpong
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "1")
    assert fresh_knobs.amd.use_block_pingpong
    monkeypatch.delenv("TRITON_HIP_USE_BLOCK_PINGPONG")
    assert fresh_knobs.amd.use_block_pingpong is None
