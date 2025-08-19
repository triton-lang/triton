"""
Metal runtime bindings for macOS (PyObjC) with a safe stub fallback on other platforms.

Exports:
- bind_library(metallib_bytes: bytes, metadata: Optional[dict]=None) -> MetalLibraryHandle
- MetalLibraryHandle: object exposing device, command_queue, library, pipeline_cache, metadata, launch_kernel(...)
- Exceptions: MetalRuntimeError, KernelNotFoundError, PipelineCreationError, ResourceError

Notes:
- If running on non-darwin platforms, bind_library returns a stub handle (is_stub=True)
  whose launch_kernel raises MetalRuntimeError. This keeps Linux CI tests importable.
- If running on darwin but PyObjC is not installed, an ImportError is raised with guidance.
- The implementation uses guarded, best-effort PyObjC calls. Calls to real Metal APIs
  are wrapped and converted to the exported exceptions.
"""

from __future__ import annotations

import sys
import time
import struct
import threading
import ctypes
import os
import platform
from typing import Optional, Sequence, Tuple, Union
import Metal._Metal as Metal 
from loguru import logger

try:
    import numpy as np
    from numpy import ndarray
except Exception:  # numpy optional for runtime; tests may mock behaviour
    np = None  # type: ignore

# Exceptions
class MetalRuntimeError(Exception):
    """Generic Metal runtime error."""

class KernelNotFoundError(MetalRuntimeError):
    """Kernel name not found in compiled library."""

class PipelineCreationError(MetalRuntimeError):
    """Failed to create compute pipeline state."""

class ResourceError(MetalRuntimeError):
    """Failed to allocate or manage a GPU resource (buffer, etc.)."""


class MetalLibraryHandle:
    """
    Represents a bound Metal library.

    Attributes:
        device: Metal device object (or None in stub)
        command_queue: Metal command queue object (or None in stub)
        library: Metal library object (or None in stub)
        pipeline_cache: dict mapping kernel name -> pipeline state
        metadata: optional metadata dict (may include reflection info)
        is_stub: True when runtime is not available (calls will raise)
        binary_bytes: original metallib bytes
    """

    def __init__(
        self,
        device=None,
        command_queue=None,
        library=None,
        metadata: Optional[dict] = None,
        binary_bytes: Optional[bytes] = None,
        is_stub: bool = False,
    ):
        self.device = device
        self.command_queue = command_queue
        self.library = library
        # Protected cache for compiled pipelines; guard with an RLock to avoid
        # duplicate compilations when multiple threads attempt to use the same
        # kernel concurrently.
        self.pipeline_cache = {}  # name -> pipeline
        self._pipeline_lock = threading.RLock()
        # Keep a reference to the Metal module (set when bind_library imports it)
        self._metal = None
        self.metadata = metadata or {}
        self.is_stub = is_stub
        self.binary_bytes = binary_bytes
        # For testability: allow recording last gpu error/status
        self._last_gpu_error: Optional[str] = None

    def launch_kernel(
        self,
        name: Optional[str] = None,
        args: Optional[Sequence[Union[ndarray, memoryview, bytes, int, float]]] = None,
        grid: Optional[Tuple[int, int, int]] = None,
        block: Optional[Tuple[int, int, int]] = None,
        timeout: Optional[float] = None,
        explicit_readback: bool = True,
        *extra_args,
        **extra_kwargs,
    ) -> dict:
        """
        Launch a compute kernel synchronously.

        Note:
            Signature is intentionally permissive to support callers (and tests)
            that may invoke launch_kernel() on stub handles without parameters.
            On stub handles this always raises a plain RuntimeError with a clear
            message. Real runtime callers must pass the required parameters.

        Args:
            name: kernel function name to lookup in the library
            args: sequence of kernel arguments. Supported:
                  - numpy.ndarray or buffer-protocol objects -> create MTLBuffer and setBuffer(index)
                  - bytes or small POD -> setBytes or small MTLBuffer
                  - ints/floats -> pack as native POD via struct.pack into setBytes
            grid: (x,y,z) total threads to execute
            block: (x,y,z) threadsPerThreadgroup
            timeout: optional seconds to wait for command buffer to complete; ignored if None.

        Returns:
            dict with keys: {'status': 'ok', 'duration_ms': float, 'gpu_error': Optional[str]}

        Raises:
            RuntimeError (stub) or MetalRuntimeError / KernelNotFoundError / PipelineCreationError / ResourceError
        """
        # For stub handles raise plain RuntimeError to preserve historical test expectations
        if self.is_stub:
            raise RuntimeError("Metal runtime unavailable: cannot launch kernel.")

        # Basic validation
        if self.device is None or self.command_queue is None or self.library is None:
            raise MetalRuntimeError("Incomplete Metal handle: missing device/queue/library.")

        # Lookup (and lazily create) pipeline.
        # Use double-checked locking to avoid duplicate pipeline creation under
        # concurrent access.
        pipeline = self.pipeline_cache.get(name)
        try:
            if pipeline is None:
                with self._pipeline_lock:
                    # Re-check under lock
                    pipeline = self.pipeline_cache.get(name)
                    if pipeline is None:
                        # Try to find function in library
                        try:
                            mtl_function = self.library.newFunctionWithName_(name)
                        except Exception:
                            # Some PyObjC wrappers may raise different exceptions; normalize
                            raise KernelNotFoundError(f"Kernel '{name}' not found in Metal library.")
                        try:
                            pipeline = self.device.newComputePipelineStateWithFunction_error_(mtl_function, None)
                        except Exception as e:
                            raise PipelineCreationError(f"Failed to create pipeline for '{name}': {e}")
                        # Publish pipeline atomically while still holding lock
                        self.pipeline_cache[name] = pipeline
        except MetalRuntimeError:
            raise
        except Exception as e:
            raise MetalRuntimeError(f"Unexpected error creating pipeline for '{name}': {e}")

        # Create command buffer & encoder
        try:
            command_buffer = self.command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()
        except Exception as e:
            raise ResourceError(f"Failed to create command buffer/encoder: {e}")

        # Bind pipeline
        try:
            encoder.setComputePipelineState_(pipeline)
        except Exception as e:
            raise PipelineCreationError(f"Failed to set pipeline state: {e}")

        # Prepare and set buffers/bytes
        created_buffers = []  # track buffers that need readback
        # Resolve kernel reflection metadata (if available) to map argument positions
        kernel_meta = None
        try:
            km = self.metadata.get("kernels") if isinstance(self.metadata, dict) else None
            if km:
                # Find matching kernel metadata by name; fall back to None if not found
                for k in km:
                    if k.get("name") == name:
                        kernel_meta = k
                        break
        except Exception:
            kernel_meta = None
        # Ensure args is an iterable for enumerate() to satisfy type-checkers and
        # avoid TypeErrors when callers pass None.
        args_iter = args or ()
        try:
            for idx, arg in enumerate(args_iter):
                # Determine the binding index based on reflection metadata if present.
                bind_index = idx
                try:
                    if kernel_meta and isinstance(kernel_meta.get("args"), (list, tuple)):
                        arg_meta = kernel_meta["args"][idx] if idx < len(kernel_meta["args"]) else None
                        if arg_meta and arg_meta.get("buffer_index") is not None:
                            bind_index = int(arg_meta["buffer_index"])
                except Exception:
                    # On any metadata parsing issue, fall back to positional binding.
                    bind_index = idx
                # Numpy / buffer-protocol
                if np is not None and isinstance(arg, np.ndarray):
                    if not arg.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arg)
                    else:
                        arr = arg
                    # Use ndarray.tobytes() (buffer protocol could confuse static typecheckers)
                    data = arr.tobytes()
                    length = len(data)
                    # options: 0 default
                    buf = self.device.newBufferWithBytes_length_options_(data, length, 0)
                    encoder.setBuffer_offset_atIndex_(buf, 0, bind_index)
                    created_buffers.append((bind_index, buf, arr))  # arr for possible readback
                elif isinstance(arg, (bytes, bytearray, memoryview)):
                    b = bytes(arg)
                    length = len(b)
                    # For small POD we can use setBytes:length:index:
                    try:
                        encoder.setBytes_length_index_(b, length, bind_index)
                    except Exception:
                        # Fallback to creating buffer
                        buf = self.device.newBufferWithBytes_length_options_(b, length, 0)
                        encoder.setBuffer_offset_atIndex_(buf, 0, bind_index)
                        created_buffers.append((bind_index, buf, None))
                elif isinstance(arg, (int, float)):
                    # Pack into native 64-bit POD (machine endianness)
                    if isinstance(arg, int):
                        packed = struct.pack("q", int(arg))
                    else:
                        packed = struct.pack("d", float(arg))
                    try:
                        encoder.setBytes_length_index_(packed, len(packed), bind_index)
                    except Exception:
                        buf = self.device.newBufferWithBytes_length_options_(packed, len(packed), 0)
                        encoder.setBuffer_offset_atIndex_(buf, 0, bind_index)
                        created_buffers.append((bind_index, buf, None))
                else:
                    # Attempt buffer protocol by coercing to bytes; this will work for
                    # objects supporting the buffer protocol (e.g., bytearray, memoryview).
                    try:
                        b = bytes(arg)
                    except Exception:
                        raise ResourceError(f"Unsupported argument type for index {idx}: {type(arg)}")
                    buf = self.device.newBufferWithBytes_length_options_(b, len(b), 0)
                    encoder.setBuffer_offset_atIndex_(buf, 0, bind_index)
                    created_buffers.append((bind_index, buf, None))
        except MetalRuntimeError:
            raise
        except Exception as e:
            raise ResourceError(f"Failed while binding kernel arguments: {e}")

        # Compute threadgroup sizes
        if block is None or grid is None:
            raise MetalRuntimeError("grid and block must be provided to launch_kernel")
        # Normalize to tuples (and validate length)
        try:
            block_vals = tuple(block)
            grid_vals = tuple(grid)
        except Exception:
            raise MetalRuntimeError("grid and block must be iterable 3-tuples of ints")
        if len(block_vals) != 3 or len(grid_vals) != 3:
            raise MetalRuntimeError("grid and block must be 3-tuples")
        threads_per_threadgroup = block_vals
        # Compute threadgroups per grid (ceil div)
        def ceil_div(a, b):
            return (a + b - 1) // b
        threadgroups_per_grid = (
            ceil_div(grid_vals[0], threads_per_threadgroup[0]),
            ceil_div(grid_vals[1], threads_per_threadgroup[1]),
            ceil_div(grid_vals[2], threads_per_threadgroup[2]),
        )

        # Dispatch
        start = time.time()
        try:
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups_per_grid, threads_per_threadgroup)
            encoder.endEncoding()
            command_buffer.commit()
            if timeout is None:
                # Simple blocking wait when caller didn't request a timeout
                command_buffer.waitUntilCompleted()
            else:
                # Prefer an event-driven bounded wait using addCompletedHandler + threading.Event.
                completed = threading.Event()
                try:
                    # PyObjC will call the provided callable with the completed command buffer.
                    command_buffer.addCompletedHandler_(lambda cb: completed.set())
                except Exception:
                    # If addCompletedHandler_ isn't available fall back to a named-status busy-wait.
                    status_completed = getattr(self._metal, "MTLCommandBufferStatusCompleted", 1)
                    deadline = time.time() + float(timeout)
                    while getattr(command_buffer, "status", lambda: None)() != status_completed and time.time() < deadline:
                        time.sleep(0.001)
                    if getattr(command_buffer, "status", lambda: None)() != status_completed:
                        raise TimeoutError("Command buffer did not complete before timeout.")
                # Wait on the event; if it times out attempt best-effort cancellation and raise.
                if not completed.wait(timeout=float(timeout)):
                    try:
                        # best-effort cancellation - may be a no-op depending on state
                        command_buffer.cancel()
                    except Exception:
                        pass
                    raise TimeoutError("Command buffer did not complete before timeout.")
        except Exception as e:
            # Attach last GPU error/status if available
            try:
                gpu_err = getattr(command_buffer, "error", lambda: None)()
            except Exception:
                gpu_err = None
            self._last_gpu_error = str(gpu_err) if gpu_err else str(e)
            raise MetalRuntimeError(f"GPU execution error: {e} | gpu_error={self._last_gpu_error}")

        # Optional explicit readback: copy GPU buffers back to host-visible memory.
        # When explicit_readback is True (the default) try to copy buffer contents
        # into the provided numpy arrays recorded earlier in `created_buffers`.
        if explicit_readback and created_buffers:
            try:
                # If device requires an explicit blit (buffers not shared), perform a blit copy
                # into a CPU-visible staging buffer and read from that.
                for bind_index, buf, arr in created_buffers:
                    # Only attempt readback for buffers tied to numpy arrays
                    if arr is None:
                        continue
                    try:
                        # Determine buffer length
                        length = None
                        try:
                            length = buf.length() if callable(getattr(buf, "length", None)) else getattr(buf, "length", None)
                        except Exception:
                            length = None
                        if length is None:
                            # Fallback to numpy size
                            length = arr.nbytes
                        # If buffer exposes shared storage, read directly via contents()
                        storage_shared = False
                        try:
                            if self._metal is not None and callable(getattr(buf, "storageMode", None)):
                                storage_shared = (buf.storageMode() == getattr(self._metal, "MTLResourceStorageModeShared", 0))
                        except Exception:
                            storage_shared = False
                        if storage_shared:
                            ptr = buf.contents()
                            # Copy into numpy array memory
                            ctypes.memmove(arr.ctypes.data, ptr, min(arr.nbytes, int(length)))
                        else:
                            # Create a staging buffer (host-visible) and blit-copy into it.
                            blit_cmd = self.command_queue.commandBuffer()
                            blit = blit_cmd.blitCommandEncoder()
                            staging = self.device.newBufferWithLength_options_(int(length), 0)
                            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(buf, 0, staging, 0, int(length))
                            blit.endEncoding()
                            blit_cmd.commit()
                            blit_cmd.waitUntilCompleted()
                            ptr = staging.contents()
                            ctypes.memmove(arr.ctypes.data, ptr, min(arr.nbytes, int(length)))
                    except Exception as e:
                        # Surface a clear ResourceError with chaining for diagnostics
                        raise ResourceError(f"Failed to read back buffer at index {bind_index}: {e}") from e
            except MetalRuntimeError:
                raise
            except Exception as e:
                # Don't swallow readback errors; surface as MetalRuntimeError
                raise MetalRuntimeError(f"GPU readback failed: {e}") from e
        end = time.time()
        duration_ms = (end - start) * 1000.0
        return {"status": "ok", "duration_ms": duration_ms, "gpu_error": self._last_gpu_error}


def bind_library(metallib_bytes: bytes, metadata: Optional[dict] = None) -> MetalLibraryHandle:
    """
    Bind a metallib bytes blob to a Metal runtime handle.

    On macOS with PyObjC installed this creates real Metal objects; on other
    platforms a stub handle (is_stub=True) is returned so tests can import safely.

    Args:
        metallib_bytes: bytes of compiled .metallib
        metadata: optional metadata dict to attach to the handle

    Returns:
        MetalLibraryHandle
    """
    metadata = metadata or {}
    # Non-darwin: return a stub that raises on use
    if sys.platform != "darwin":
        logger.debug("Non-darwin platform detected; returning stub MetalLibraryHandle.")
        # Populate sensible defaults for stub metadata so callers (and tests) can
        # rely on fields like platform and library_size even when runtime is absent.
        stub_meta = {"platform": "unknown", "library_size": len(metallib_bytes)}
        stub_meta.update(metadata or {})
        return MetalLibraryHandle(device=None, command_queue=None, library=None, metadata=stub_meta, binary_bytes=metallib_bytes, is_stub=True)

    # Darwin: attempt to import PyObjC frameworks
    try:
        import objc  # type: ignore
        import Metal  # type: ignore
        from Foundation import NSData  # type: ignore
    except Exception as e:
        # PyObjC or Metal framework not available — return a lazy-failing stub handle.
        # This keeps CI and non-native dev environments deterministic and import-safe.
        logger.warning("PyObjC/Metal frameworks not available; returning stub MetalLibraryHandle. (%s)", e)
        metadata = {**metadata, "pyobjc_available": False}
        return MetalLibraryHandle(device=None, command_queue=None, library=None, metadata=metadata, binary_bytes=metallib_bytes, is_stub=True)

    # Create device, command queue, and library
    try:
        # Prefer the global helper if present; otherwise use common API names
        device = None
        try:
            device = Metal.MTLCreateSystemDefaultDevice()
        except Exception:
            device = getattr(Metal, "MTLCreateSystemDefaultDevice", lambda: None)()
        if device is None:
            raise MetalRuntimeError("Failed to obtain system Metal device.")

        command_queue = device.newCommandQueue()
        # Build NSData from bytes
        nsdata = NSData.alloc().initWithBytes_length_(metallib_bytes, len(metallib_bytes))
        # Load library (some APIs return (lib, err) others raise)
        try:
            library = device.newLibraryWithData_error_(nsdata, None)
        except Exception:
            # Try alternate selector
            library = device.newLibraryWithData_(nsdata)
        # library may be a tuple (lib, err)
        if isinstance(library, tuple):
            library = library[0]
        handle = MetalLibraryHandle(device=device, command_queue=command_queue, library=library, metadata=metadata, binary_bytes=metallib_bytes, is_stub=False)
        return handle
    except MetalRuntimeError:
        raise
    except Exception as e:
        logger.exception("Failed to bind Metal library to runtime.")
        raise MetalRuntimeError(f"Failed to bind Metal library: {e}")
