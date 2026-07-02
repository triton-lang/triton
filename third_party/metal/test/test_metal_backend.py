"""Tests for the Metal backend registration and basic functionality."""

import sys
import pytest

# Skip entire module on non-macOS
pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason="Metal requires macOS")


class TestMetalBackendRegistration:
    """Test that the Metal backend is properly registered and discoverable."""

    def test_backend_in_registry(self):
        """Metal backend should be discoverable in the backends registry."""
        from triton.backends import backends
        # On macOS, metal should be registered
        if sys.platform == 'darwin':
            assert 'metal' in backends

    def test_backend_has_compiler(self):
        """Metal backend should have a compiler class."""
        from triton.backends import backends
        if 'metal' not in backends:
            pytest.skip("Metal backend not installed")
        backend = backends['metal']
        assert backend.compiler is not None

    def test_backend_has_driver(self):
        """Metal backend should have a driver class."""
        from triton.backends import backends
        if 'metal' not in backends:
            pytest.skip("Metal backend not installed")
        backend = backends['metal']
        assert backend.driver is not None


class TestMetalDriver:
    """Test Metal driver functionality."""

    def test_driver_import(self):
        """MetalDriver should be importable."""
        from triton.backends.metal.driver import MetalDriver
        assert MetalDriver is not None

    def test_driver_is_active_on_macos(self):
        """MetalDriver.is_active() should return True on macOS with Apple Silicon."""
        from triton.backends.metal.driver import MetalDriver
        # On macOS, Metal should be active
        result = MetalDriver.is_active()
        assert isinstance(result, bool)
        if sys.platform == 'darwin':
            assert result is True

    def test_get_current_target(self):
        """Should return a valid GPUTarget for Metal."""
        from triton.backends.metal.driver import MetalDriver
        if not MetalDriver.is_active():
            pytest.skip("Metal not active")
        driver = MetalDriver()
        target = driver.get_current_target()
        assert target.backend == 'metal'
        assert target.warp_size == 32
        assert target.arch >= 7  # Apple7 (M1) minimum

    def test_get_active_torch_device(self):
        """Should return MPS torch device when available."""
        from triton.backends.metal.driver import MetalDriver
        if not MetalDriver.is_active():
            pytest.skip("Metal not active")
        driver = MetalDriver()
        try:
            import torch
            device = driver.get_active_torch_device()
            assert device is not None
        except ImportError:
            pytest.skip("torch not installed")

    def test_map_python_to_cpp_type(self):
        """Type mapping should work for all supported types."""
        from triton.backends.metal.driver import MetalDriver
        driver = MetalDriver()
        assert driver.map_python_to_cpp_type("i32") == "int32_t"
        assert driver.map_python_to_cpp_type("fp32") == "float"
        assert driver.map_python_to_cpp_type("i64") == "int64_t"
        assert driver.map_python_to_cpp_type("*fp32") == "uint64_t"

    def test_gpu_family_detection(self):
        """GPU family detection should return a valid family number."""
        from triton.backends.metal.driver import _get_metal_gpu_family
        family = _get_metal_gpu_family()
        assert family >= 7  # Minimum Apple7 (M1)
        assert family <= 15  # Reasonable upper bound for future chips


class TestMetalCompiler:
    """Test Metal compiler functionality."""

    def test_compiler_import(self):
        """MetalBackend should be importable."""
        from triton.backends.metal.compiler import MetalBackend
        assert MetalBackend is not None

    def test_supports_target(self):
        """MetalBackend should support metal targets."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        metal_target = GPUTarget("metal", 9, 32)
        cuda_target = GPUTarget("cuda", 90, 32)

        assert MetalBackend.supports_target(metal_target) is True
        assert MetalBackend.supports_target(cuda_target) is False

    def test_parse_options(self):
        """Options parsing should return valid MetalOptions."""
        from triton.backends.metal.compiler import MetalBackend, MetalOptions
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        options = backend.parse_options({})

        assert isinstance(options, MetalOptions)
        assert options.num_warps == 4
        assert options.warp_size == 32
        assert options.backend_name == 'metal'
        assert options.arch == "apple9"

    def test_parse_options_custom(self):
        """Custom options should be respected."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        options = backend.parse_options({'num_warps': 8, 'num_stages': 3})

        assert options.num_warps == 8
        assert options.num_stages == 3

    def test_add_stages(self):
        """add_stages should populate the stages dict with expected keys."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        options = backend.parse_options({})

        stages = {}
        backend.add_stages(stages, options)

        assert "ttir" in stages
        assert "ttgir" in stages
        assert "llir" in stages
        assert "msl" in stages
        assert "metallib" in stages

    def test_hash(self):
        """Backend hash should be consistent."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)

        h1 = backend.hash()
        h2 = backend.hash()
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) > 0

    def test_binary_ext(self):
        """Binary extension should be 'metallib'."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        assert backend.binary_ext == "metallib"

    def test_options_hash(self):
        """MetalOptions hash should be deterministic."""
        from triton.backends.metal.compiler import MetalOptions

        opt1 = MetalOptions(num_warps=4)
        opt2 = MetalOptions(num_warps=4)
        opt3 = MetalOptions(num_warps=8)

        assert opt1.hash() == opt2.hash()
        assert opt1.hash() != opt3.hash()


class TestMetalLanguageExtensions:
    """Test Metal-specific language extensions."""

    def test_import_language(self):
        """Metal language extensions should be importable."""
        from triton.language.extra.metal import get_simd_width, get_max_threadgroup_size

        assert get_simd_width() == 32
        assert get_max_threadgroup_size() == 1024

    def test_max_threadgroup_memory(self):
        """Should report correct max threadgroup memory."""
        from triton.language.extra.metal import get_max_threadgroup_memory

        assert get_max_threadgroup_memory() == 32768


class TestMetalOptions:
    """Test MetalOptions dataclass validation."""

    def test_default_options(self):
        """Default options should be valid."""
        from triton.backends.metal.compiler import MetalOptions

        opts = MetalOptions()
        assert opts.num_warps == 4
        assert opts.num_stages == 2
        assert opts.warp_size == 32
        assert opts.max_threadgroup_memory == 32768
        assert opts.max_threads_per_threadgroup == 1024
        assert opts.enable_fp_fusion is True
        assert opts.backend_name == 'metal'

    def test_num_warps_must_be_power_of_two(self):
        """num_warps must be a power of 2."""
        from triton.backends.metal.compiler import MetalOptions

        # Valid
        MetalOptions(num_warps=1)
        MetalOptions(num_warps=2)
        MetalOptions(num_warps=4)
        MetalOptions(num_warps=8)

        # Invalid
        with pytest.raises(AssertionError):
            MetalOptions(num_warps=3)
        with pytest.raises(AssertionError):
            MetalOptions(num_warps=5)

    def test_options_hash_stability(self):
        """Hash should be stable across calls."""
        from triton.backends.metal.compiler import MetalOptions

        opts = MetalOptions(num_warps=4, num_stages=2)
        h1 = opts.hash()
        h2 = opts.hash()
        assert h1 == h2

    def test_extern_libs_normalization(self):
        """extern_libs dict should be normalized to tuple."""
        from triton.backends.metal.compiler import MetalOptions

        opts = MetalOptions(extern_libs={'test': '/path/to/lib'})
        assert isinstance(opts.extern_libs, tuple)
        assert opts.extern_libs == (('test', '/path/to/lib'),)
