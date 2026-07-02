"""Unit tests for Metal compiler internals."""

import sys
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason="Metal requires macOS")


class TestMetalVersion:
    """Test Metal version detection."""

    def test_get_metal_version(self):
        """get_metal_version should return a string."""
        from triton.backends.metal.compiler import get_metal_version
        version = get_metal_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_version_cached(self):
        """Version detection should be cached."""
        from triton.backends.metal.compiler import get_metal_version
        v1 = get_metal_version()
        v2 = get_metal_version()
        assert v1 == v2


class TestMetalArch:
    """Test Metal architecture string generation."""

    def test_known_families(self):
        """Known GPU families should map correctly."""
        from triton.backends.metal.compiler import get_metal_arch
        assert get_metal_arch(7) == "apple7"
        assert get_metal_arch(8) == "apple8"
        assert get_metal_arch(9) == "apple9"
        assert get_metal_arch(10) == "apple10"

    def test_unknown_family_fallback(self):
        """Unknown families should generate a reasonable default."""
        from triton.backends.metal.compiler import get_metal_arch
        result = get_metal_arch(11)
        assert result == "apple11"


class TestMinDotSize:
    """Test dot product minimum size constraints."""

    def test_min_dot_size(self):
        """min_dot_size should return a function that gives (8,8,8) tiles."""
        from triton.backends.metal.compiler import min_dot_size
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        check_fn = min_dot_size(target)

        class MockType:

            class scalar:
                primitive_bitwidth = 16

        result = check_fn(MockType(), MockType())
        assert result == (8, 8, 8)

    def test_min_dot_size_fp32(self):
        """FP32 dots should also use 8x8x8 tiles."""
        from triton.backends.metal.compiler import min_dot_size
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        check_fn = min_dot_size(target)

        class MockType:

            class scalar:
                primitive_bitwidth = 32

        result = check_fn(MockType(), MockType())
        assert result == (8, 8, 8)


class TestMetalBackendInit:
    """Test MetalBackend initialization."""

    def test_create_backend(self):
        """Should create a MetalBackend instance."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        assert backend.target == target
        assert backend.binary_ext == "metallib"

    def test_supports_target_positive(self):
        """Should support 'metal' backend targets."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        assert MetalBackend.supports_target(GPUTarget("metal", 7, 32))
        assert MetalBackend.supports_target(GPUTarget("metal", 8, 32))
        assert MetalBackend.supports_target(GPUTarget("metal", 9, 32))
        assert MetalBackend.supports_target(GPUTarget("metal", 10, 32))

    def test_supports_target_negative(self):
        """Should not support non-metal targets."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        assert not MetalBackend.supports_target(GPUTarget("cuda", 90, 32))
        assert not MetalBackend.supports_target(GPUTarget("hip", "gfx940", 64))

    def test_get_module_map_empty(self):
        """Module map should be empty (no external libs like libdevice)."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        assert backend.get_module_map() == {}

    def test_load_dialects_no_error(self):
        """load_dialects should not raise."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        # Should not raise
        backend.load_dialects(None)


class TestCompilationStages:
    """Test compilation stage registration."""

    def test_triton_language_stages(self):
        """Triton language should have all stages."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        options = backend.parse_options({})

        stages = {}
        backend.add_stages(stages, options)

        expected_stages = ["ttir", "ttgir", "llir", "msl", "metallib"]
        for stage in expected_stages:
            assert stage in stages, f"Missing stage: {stage}"
            assert callable(stages[stage])

    def test_stage_order(self):
        """Stages should be in correct compilation order."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        options = backend.parse_options({})

        stages = {}
        backend.add_stages(stages, options)

        stage_keys = list(stages.keys())
        assert stage_keys.index("ttir") < stage_keys.index("ttgir")
        assert stage_keys.index("ttgir") < stage_keys.index("llir")
        assert stage_keys.index("llir") < stage_keys.index("msl")
        assert stage_keys.index("msl") < stage_keys.index("metallib")


class TestOptionsValidation:
    """Test MetalOptions validation."""

    def test_default_options_valid(self):
        """Default options should pass validation."""
        from triton.backends.metal.compiler import MetalOptions
        opts = MetalOptions()
        assert opts.num_warps == 4

    def test_zero_warps_invalid(self):
        """num_warps=0 should raise."""
        from triton.backends.metal.compiler import MetalOptions
        with pytest.raises(AssertionError):
            MetalOptions(num_warps=0)

    def test_negative_warps_invalid(self):
        """Negative num_warps should raise."""
        from triton.backends.metal.compiler import MetalOptions
        with pytest.raises(AssertionError):
            MetalOptions(num_warps=-1)
