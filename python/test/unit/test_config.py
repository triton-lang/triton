import pytest
import shutil
import triton

from pathlib import Path


def test_config_utils() -> None:

    class test_config(triton.config.base_config):
        foo: triton.config.env_str = triton.config.env_str("FOO", "triton")
        bar: triton.config.env_bool = triton.config.env_bool("BAR", True)
        baz: triton.config.env_opt_str = triton.config.env_opt_str("BAZ")
        quux: triton.config.env_opt_bool = triton.config.env_opt_bool("QUUX")

    instance = test_config()

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


def test_config_scope(fresh_config, monkeypatch):
    monkeypatch.setenv("TRITON_HIP_LOCAL_PREFETCH", "17")
    fresh_config.amd.global_prefetch = 4
    fresh_config.amd.local_prefetch = 3

    assert fresh_config.amd.global_prefetch == 4
    assert fresh_config.amd.local_prefetch == 3
    assert fresh_config.amd.use_buffer_ops

    # Just to prove that use_buffer_ops is coming from env
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
    assert not fresh_config.amd.use_buffer_ops
    monkeypatch.delenv("AMDGCN_USE_BUFFER_OPS")
    assert fresh_config.amd.use_buffer_ops

    with fresh_config.amd.scope():
        fresh_config.amd.global_prefetch = 5
        # Use the environment
        del fresh_config.amd.local_prefetch
        fresh_config.amd.use_buffer_ops = False

        assert fresh_config.amd.global_prefetch == 5
        assert fresh_config.amd.local_prefetch == 17
        assert not fresh_config.amd.use_buffer_ops

    assert fresh_config.amd.global_prefetch == 4
    assert fresh_config.amd.local_prefetch == 3
    assert fresh_config.amd.use_buffer_ops

    # Just to prove that use_buffer_ops is coming from env
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
    assert not fresh_config.amd.use_buffer_ops
    monkeypatch.delenv("AMDGCN_USE_BUFFER_OPS")
    assert fresh_config.amd.use_buffer_ops


@pytest.mark.parametrize("truthy_falsey", [("1", "0"), ("true", "false"), ("True", "False"), ("TRUE", "FALSE"),
                                           ("y", "n"), ("YES", "NO"), ("ON", "OFF")])
def test_read_env(truthy_falsey, fresh_config, monkeypatch):
    truthy, falsey = truthy_falsey

    # bool defaulting to False
    assert not fresh_config.runtime.debug
    # bool defaulting to True
    assert fresh_config.language.default_fp_fusion
    # str defaulting to None
    assert fresh_config.compilation.use_ir_loc is None
    # str defaulting to not None
    assert fresh_config.cache.dir.endswith(".triton/cache")
    # class defaulting to None
    assert fresh_config.cache.manager_class is None
    # set[str] defaulting to empty
    assert len(fresh_config.build.backend_dirs) == 0

    monkeypatch.setenv("TRITON_DEFAULT_FP_FUSION", falsey)
    monkeypatch.setenv("TRITON_DEBUG", truthy)
    monkeypatch.setenv("USE_IR_LOC", "ttir")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/triton_cache")
    monkeypatch.setenv("TRITON_HOME", "/tmp/triton_home")
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "triton.runtime.cache:FileCacheManager")
    monkeypatch.setenv("TRITON_CUDACRT_PATH", "/tmp/cuda/crt")
    monkeypatch.setenv("TRITON_CUDART_PATH", "/tmp/cuda/rt")

    assert fresh_config.runtime.debug
    assert not fresh_config.language.default_fp_fusion
    assert fresh_config.compilation.use_ir_loc == "ttir"
    assert fresh_config.cache.dir == "/tmp/triton_cache"
    assert fresh_config.cache.dump_dir == "/tmp/triton_home/.triton/dump"
    assert fresh_config.cache.override_dir == "/tmp/triton_home/.triton/override"

    from triton.runtime.cache import FileCacheManager
    assert fresh_config.cache.manager_class == FileCacheManager

    assert fresh_config.build.backend_dirs == {"/tmp/cuda/crt", "/tmp/cuda/rt"}


def test_set_config_directly(fresh_config, monkeypatch):
    assert fresh_config.cache.dir.endswith(".triton/cache")

    fresh_config.cache.dir = "/tmp/triton_cache"
    assert fresh_config.cache.dir == "/tmp/triton_cache"

    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/other_triton_cache")
    assert fresh_config.cache.dir == "/tmp/triton_cache"

    fresh_config.cache.dir = fresh_config.env
    assert fresh_config.cache.dir == "/tmp/other_triton_cache"

    fresh_config.cache.dir = "/tmp/triton_cache"
    fresh_config.cache.reset()
    assert fresh_config.cache.dir == "/tmp/other_triton_cache"

    # Just in case, lets check all the other datatypes too
    fresh_config.language.default_fp_fusion = False
    fresh_config.amd.use_block_pingpong = True
    fresh_config.amd.global_prefetch = 5
    fresh_config.nvidia.mock_ptx_version = "42.0.1"

    from triton.runtime.cache import FileCacheManager

    class TestManagerClass(FileCacheManager):
        pass

    fresh_config.cache.manager_class = TestManagerClass

    monkeypatch.setenv("TRITON_CUDART_PATH", "/tmp/the/real/cudart")
    monkeypatch.setenv("TRITON_DEFAULT_FP_FUSION", "1")
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "0")
    monkeypatch.setenv("TRITON_HIP_GLOBAL_PREFETCH", "2")
    monkeypatch.setenv("TRITON_MOCK_PTX_VERSION", "1.0.0")
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "triton.runtime.cache:FileCacheManager")

    assert not fresh_config.language.default_fp_fusion
    assert fresh_config.amd.use_block_pingpong
    assert fresh_config.amd.global_prefetch == 5
    assert fresh_config.nvidia.mock_ptx_version == "42.0.1"
    assert fresh_config.cache.manager_class == TestManagerClass

    # Make sure both setting `.env` or deleting resets to env vars.
    fresh_config.language.default_fp_fusion = fresh_config.env
    fresh_config.amd.use_block_pingpong = fresh_config.env
    fresh_config.amd.global_prefetch = fresh_config.env
    del fresh_config.nvidia.mock_ptx_version
    del fresh_config.cache.manager_class

    assert fresh_config.build.backend_dirs == {"/tmp/the/real/cudart"}
    assert fresh_config.language.default_fp_fusion
    assert not fresh_config.amd.use_block_pingpong
    assert fresh_config.amd.global_prefetch == 2
    assert fresh_config.nvidia.mock_ptx_version == "1.0.0"
    assert fresh_config.cache.manager_class == FileCacheManager


def test_nvidia_tool(fresh_config, tmp_path, monkeypatch):
    triton_root = Path(__file__).parent.parent.parent / "triton"
    default_ptxas = triton_root / "backends/nvidia/bin/ptxas"

    assert default_ptxas.exists()
    assert Path(fresh_config.nvidia.ptxas.path).resolve() == default_ptxas.resolve()

    tmp_ptxas = tmp_path / "ptxas-special"
    shutil.copy(default_ptxas, tmp_ptxas)
    monkeypatch.setenv("TRITON_PTXAS_PATH", str(tmp_ptxas))
    assert Path(fresh_config.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()

    fresh_config.nvidia.ptxas = str(default_ptxas)
    assert Path(fresh_config.nvidia.ptxas.path).resolve() == default_ptxas.resolve()

    del fresh_config.nvidia.ptxas
    assert Path(fresh_config.nvidia.ptxas.path).resolve() == tmp_ptxas.resolve()

    monkeypatch.delenv("TRITON_PTXAS_PATH")
    assert Path(fresh_config.nvidia.ptxas.path).resolve() == default_ptxas.resolve()


def test_opt_bool(fresh_config, monkeypatch):
    assert fresh_config.amd.use_block_pingpong is None
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "0")
    assert not fresh_config.amd.use_block_pingpong
    monkeypatch.setenv("TRITON_HIP_USE_BLOCK_PINGPONG", "1")
    assert fresh_config.amd.use_block_pingpong
    monkeypatch.delenv("TRITON_HIP_USE_BLOCK_PINGPONG")
    assert fresh_config.amd.use_block_pingpong is None
