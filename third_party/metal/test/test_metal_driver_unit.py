"""Unit tests for Metal driver internals - can run without GPU."""

import sys
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason="Metal requires macOS")


class TestMetalAvailability:
    """Test Metal availability detection."""

    def test_metal_is_available_returns_bool(self):
        """_metal_is_available should always return a boolean."""
        from triton.backends.metal.driver import _metal_is_available
        result = _metal_is_available()
        assert isinstance(result, bool)

    def test_metal_available_on_macos(self):
        """On macOS, Metal should generally be available."""
        from triton.backends.metal.driver import _metal_is_available
        assert _metal_is_available() is True


class TestGPUFamilyDetection:
    """Test GPU family detection logic."""

    def test_returns_integer(self):
        """GPU family should be an integer."""
        from triton.backends.metal.driver import _get_metal_gpu_family
        family = _get_metal_gpu_family()
        assert isinstance(family, int)

    def test_valid_range(self):
        """GPU family should be in a valid range for Apple Silicon."""
        from triton.backends.metal.driver import _get_metal_gpu_family
        family = _get_metal_gpu_family()
        assert 7 <= family <= 15

    def test_cached(self):
        """GPU family detection should be cached."""
        from triton.backends.metal.driver import _get_metal_gpu_family
        f1 = _get_metal_gpu_family()
        f2 = _get_metal_gpu_family()
        assert f1 == f2


class TestDeviceName:
    """Test device name detection."""

    def test_returns_string(self):
        """Device name should be a string."""
        from triton.backends.metal.driver import _get_metal_device_name
        name = _get_metal_device_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_contains_apple(self):
        """On Apple Silicon, device name should reference Apple."""
        from triton.backends.metal.driver import _get_metal_device_name
        name = _get_metal_device_name()
        # Should contain Apple or a chip model name
        assert "Apple" in name or "M" in name or "GPU" in name


class TestTypeMapping:
    """Test Triton type to C++ type mapping."""

    def test_integer_types(self):
        from triton.backends.metal.driver import ty_to_cpp
        assert ty_to_cpp("i8") == "int8_t"
        assert ty_to_cpp("i16") == "int16_t"
        assert ty_to_cpp("i32") == "int32_t"
        assert ty_to_cpp("i64") == "int64_t"

    def test_unsigned_types(self):
        from triton.backends.metal.driver import ty_to_cpp
        assert ty_to_cpp("u8") == "uint8_t"
        assert ty_to_cpp("u16") == "uint16_t"
        assert ty_to_cpp("u32") == "uint32_t"
        assert ty_to_cpp("u64") == "uint64_t"

    def test_float_types(self):
        from triton.backends.metal.driver import ty_to_cpp
        assert ty_to_cpp("fp16") == "float"
        assert ty_to_cpp("bf16") == "float"
        assert ty_to_cpp("fp32") == "float"
        assert ty_to_cpp("fp64") == "double"

    def test_pointer_types(self):
        from triton.backends.metal.driver import ty_to_cpp
        assert ty_to_cpp("*fp32") == "uint64_t"
        assert ty_to_cpp("*i32") == "uint64_t"
        assert ty_to_cpp("*fp16") == "uint64_t"

    def test_invalid_type_raises(self):
        from triton.backends.metal.driver import ty_to_cpp
        with pytest.raises(KeyError):
            ty_to_cpp("invalid_type")


class TestMetalLauncher:
    """Test MetalLauncher initialization."""

    def test_launcher_creation(self):
        """MetalLauncher should initialize with metadata."""
        from triton.backends.metal.driver import MetalLauncher

        class MockMetadata:
            global_scratch_size = 0
            global_scratch_align = 1
            profile_scratch_size = 0
            profile_scratch_align = 1

        src = type('MockSrc', (), {'constants': {}, 'signature': {}})()
        metadata = MockMetadata()
        metadata_dict = {
            "name": "test_kernel",
            "compile_mode": "jit_msl",
            "shared": 1024,
            "num_warps": 4,
        }

        # MetalLauncher uses dict-style metadata
        launcher = MetalLauncher(src, metadata_dict)
        assert launcher.kernel_name == "test_kernel"
        assert launcher.shared_memory == 1024
        assert launcher.num_warps == 4


class TestMetalUtils:
    """Test MetalUtils singleton."""

    def test_singleton(self):
        """MetalUtils should be a singleton."""
        from triton.backends.metal.driver import MetalUtils
        u1 = MetalUtils()
        u2 = MetalUtils()
        assert u1 is u2

    def test_get_device_name(self):
        """Should return a device name string."""
        from triton.backends.metal.driver import MetalUtils
        utils = MetalUtils()
        name = utils.get_device_name()
        assert isinstance(name, str)

    def test_get_gpu_family(self):
        """Should return an integer GPU family."""
        from triton.backends.metal.driver import MetalUtils
        utils = MetalUtils()
        family = utils.get_gpu_family()
        assert isinstance(family, int)
        assert family >= 7
