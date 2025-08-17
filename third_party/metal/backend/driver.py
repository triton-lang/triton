from loguru import logger
from typing import Optional

# Delegate heavy runtime work to the dedicated runtime module. This keeps the
# driver thin and makes runtime.bind_library easy to patch in tests.
try:
    from third_party.metal.backend import runtime as runtime_mod
except Exception:
    runtime_mod = None  # Binding errors will surface when load_binary is called.

class MetalDriver:
    """
    MetalDriver provides runtime interaction for Metal backend.
    """
    def load_binary(self, binary: bytes, metadata: Optional[dict] = None):
        """
        Load compiled Metal binary for execution and return a handle.

        This implementation is intentionally thin: it delegates to
        third_party.metal.backend.runtime.bind_library so tests can patch the
        single entrypoint `bind_library`.
        """
        metadata = metadata or {}
        if runtime_mod is None:
            logger.warning("Runtime module unavailable; returning a lazy stub handle.")
            # Create a minimal stub that mimics the previous shape so tests remain deterministic.
            class _Stub:
                def __init__(self, binary_bytes, metadata):
                    self.binary_bytes = binary_bytes
                    self.metadata = metadata
                    self.is_stub = True
                def launch_kernel(self, *args, **kwargs):
                    raise RuntimeError("Metal runtime unavailable: cannot launch kernel.")
            return _Stub(binary, {"platform": "unknown", "library_size": len(binary), **metadata})

        # Delegate to runtime.bind_library; let it raise ImportError or MetalRuntimeError
        return runtime_mod.bind_library(binary, metadata=metadata)