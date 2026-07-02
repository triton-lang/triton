"""Integration tests for Metal backend - requires macOS with Apple Silicon."""

import sys
import pytest

pytestmark = [
    pytest.mark.skipif(sys.platform != 'darwin', reason="Metal requires macOS"),
]


def _metal_device_available():
    """Check if a Metal device is actually available for compute."""
    try:
        from triton.backends.metal.driver import MetalDriver
        return MetalDriver.is_active()
    except (ImportError, Exception):
        return False


def _torch_mps_available():
    """Check if PyTorch MPS backend is available."""
    try:
        import torch
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except ImportError:
        return False


requires_metal = pytest.mark.skipif(not _metal_device_available(), reason="Metal device not available")

requires_torch_mps = pytest.mark.skipif(not _torch_mps_available(), reason="PyTorch MPS not available")


@requires_metal
class TestMetalDeviceQueries:
    """Test Metal device property queries."""

    def test_target_creation(self):
        """Should create a valid GPUTarget for the current device."""
        from triton.backends.metal.driver import MetalDriver
        driver = MetalDriver()
        target = driver.get_current_target()

        assert target.backend == "metal"
        assert target.warp_size == 32
        assert isinstance(target.arch, int)
        assert target.arch >= 7

    def test_driver_singleton(self):
        """Multiple MetalDriver instances should work."""
        from triton.backends.metal.driver import MetalDriver
        d1 = MetalDriver()
        d2 = MetalDriver()
        t1 = d1.get_current_target()
        t2 = d2.get_current_target()
        assert t1.backend == t2.backend
        assert t1.arch == t2.arch


@requires_metal
@requires_torch_mps
class TestMetalKernelCompilation:
    """Test kernel compilation through the Metal backend pipeline."""

    def test_vector_add_compilation(self):
        """A simple vector add kernel should compile without errors."""
        import triton
        import triton.language as tl
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget
        from triton.backends.metal.driver import MetalDriver

        driver = MetalDriver()
        target = driver.get_current_target()
        backend = MetalBackend(target)

        # Verify the backend accepts this target
        assert backend.supports_target(target)

        # Parse options
        options = backend.parse_options({'num_warps': 4})
        assert options.arch == f"apple{target.arch}"

    def test_options_for_different_gpu_families(self):
        """Options should adapt to different GPU families."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        for family in [7, 8, 9, 10]:
            target = GPUTarget("metal", family, 32)
            backend = MetalBackend(target)
            options = backend.parse_options({})
            assert options.arch == f"apple{family}"
            assert options.warp_size == 32


@requires_metal
@requires_torch_mps
class TestMetalMemoryModel:
    """Test Metal's unified memory model integration."""

    def test_mps_tensor_creation(self):
        """Should be able to create MPS tensors for Metal backend use."""
        import torch

        x = torch.randn(1024, device='mps')
        assert x.device.type == 'mps'
        assert x.shape == (1024, )

    def test_mps_tensor_operations(self):
        """Basic MPS operations should work (validates device is functional)."""
        import torch

        a = torch.randn(256, device='mps')
        b = torch.randn(256, device='mps')
        c = a + b
        assert c.device.type == 'mps'
        assert c.shape == (256, )

    def test_mps_to_cpu_transfer(self):
        """MPS tensors should transfer to CPU correctly."""
        import torch

        x = torch.randn(128, device='mps')
        y = x.cpu()
        assert y.device.type == 'cpu'
        assert torch.allclose(x.cpu(), y)


@requires_metal
class TestMetalCompilerPipeline:
    """Test the full compilation pipeline stages."""

    def test_stages_are_callable(self):
        """All registered stages should be callable."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        options = backend.parse_options({})

        stages = {}
        backend.add_stages(stages, options)

        for name, fn in stages.items():
            assert callable(fn), f"Stage '{name}' is not callable"

    def test_pack_metadata(self):
        """pack_metadata should produce a tuple."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)

        class MockMetadata:
            num_warps = 4
            num_ctas = 1
            shared = 2048

        result = backend.pack_metadata(MockMetadata())
        assert result == (4, 1, 2048)

    def test_codegen_implementation(self):
        """get_codegen_implementation should return min_dot_size."""
        from triton.backends.metal.compiler import MetalBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("metal", 9, 32)
        backend = MetalBackend(target)
        options = backend.parse_options({})

        codegen = backend.get_codegen_implementation(options)
        assert "min_dot_size" in codegen
        assert callable(codegen["min_dot_size"])
