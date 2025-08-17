# Metal backend — runtime & kernel launch (internal)
## Unresolved issues / Fixes

2025-08-17 — sealad886 &lt;155285242+sealad886@users.noreply.github.com&gt;: Symptom: "ld: library 'TritonRuntime' not found" when building Metal unit tests. Fix applied: Added CMake alias TritonRuntime -> TritonTools in lib/Tools/CMakeLists.txt to resolve legacy references.


This document describes the Metal runtime binding and kernel-launch API implemented in [`third_party/metal/backend/runtime.py`](third_party/metal/backend/runtime.py:1). It is intended for engineers and test authors who need to build, run and validate kernels on macOS using the Metal backend.

Status
- Location: docs/_internal/backend/metal.md
- Source code: [`third_party/metal/backend/runtime.py`](third_party/metal/backend/runtime.py:1), [`third_party/metal/backend/compiler.py`](third_party/metal/backend/compiler.py:1)
- Tests: [`python/test/backend/metal_runtime_test.py`](python/test/backend/metal_runtime_test.py:1)

What this doc covers
- Public runtime entrypoint and handle API
- launch_kernel() usage and argument-binding rules
- Supported argument types and packing rules
- Buffer allocation and lifetime semantics
- Exported error classes and when they are raised
- Platforms / prerequisites and stub behaviour
- Minimal copy-paste example (compile → load → launch)
- Testing notes, acceptance criteria and troubleshooting hints

1. Public entrypoint
- bind_library(metallib_bytes, metadata=None) -> MetalLibraryHandle
  - Implemented in [`third_party/metal/backend/runtime.py`](third_party/metal/backend/runtime.py:1:286).
  - On Darwin with PyObjC available this returns a real handle bound to a Metal device/queue/library.
  - On non-Darwin platforms or when PyObjC/frameworks are unavailable a stub handle (is_stub=True) is returned. The stub preserves import-time safety; calling launch_kernel() on a stub raises a descriptive RuntimeError.

2. MetalLibraryHandle (summary)
- The returned object exposes:
  - device, command_queue, library
  - metadata (optional reflection dict)
  - pipeline_cache (lazy pipeline state cache)
  - binary_bytes (original .metallib bytes)
  - is_stub (True for lazy-failing stub)
  - launch_kernel(...) — see below

Refer to the implementation for full behaviour: [`third_party/metal/backend/runtime.py`](third_party/metal/backend/runtime.py:1:45).

3. launch_kernel signature and return
Public signature:
```python
# python
result = handle.launch_kernel(
    name: Optional[str] = None,
    args: Optional[Sequence[Union[numpy.ndarray, memoryview, bytes, int, float]]] = None,
    grid: Optional[Tuple[int, int, int]] = None,
    block: Optional[Tuple[int, int, int]] = None,
    timeout: Optional[float] = None,
    *extra_args,
    **extra_kwargs,
) -> dict
```

- Return: dict with keys {'status': 'ok', 'duration_ms': float, 'gpu_error': Optional[str]} on success.
- The API is intentionally permissive to support stub handles in tests; real callers must pass at least name, args, grid and block.
- If the handle is a stub, calling launch_kernel() raises RuntimeError("Metal runtime unavailable: cannot launch kernel.").

4. Argument binding rules
- The runtime attempts to use reflection metadata if available:
  - `metadata["kernels"]` may contain per-kernel dicts with an "args" list and per-arg entries including "buffer_index".
  - If metadata for the kernel exists and an arg entry contains "buffer_index", that integer is used as the Metal argument index (bind_index).
  - Otherwise the runtime falls back to positional binding: argument i → buffer index i.
- Metadata is best-effort only (parser lives in the compiler module). If metadata parsing fails the binding falls back to positional.

5. Supported argument types and packing rules
- numpy.ndarray (C-contiguous preferred)
  - The runtime converts arrays to bytes (ndarray.tobytes()), creates an MTLBuffer via device.newBufferWithBytes_length_options_, and calls encoder.setBuffer_offset_atIndex_(buf, 0, index).
  - The original numpy array is retained (for testing convenience) when a buffer is created; tests may rely on implicit readback for simple cases but should not depend on it.
- bytes / bytearray / memoryview
  - The runtime first tries encoder.setBytes_length_index_(b, len(b), index) for small POD-like data.
  - If setBytes fails a buffer is created and bound as above.
- int / float scalars
  - Packed as native 64-bit POD using struct.pack("q", int) for ints and struct.pack("d", float) for floats (machine endianness).
  - Packed data is passed via setBytes_length_index_ if available; otherwise via a small buffer.
  - Note: scalars are always packed to 64-bit (q/d). Kernel authors should expect that binding size and alignment may differ from their Metal parameter declarations — verify kernel signatures match these sizes.
- Other buffer-protocol objects
  - The runtime will attempt bytes(obj); if that fails a ResourceError is raised.

6. Buffer allocation semantics and lifetime
- Buffers created by the runtime are allocated via device.newBufferWithBytes_length_options_.
- The runtime records created_buffers as tuples (bind_index, buffer, host_array_or_None) internally for readback expectations and debugging.
- The runtime does not expose an explicit readback API. For tests and callers that require results in host memory:
  - If you pass a numpy.ndarray as the host-side buffer, the runtime creates a device buffer and keeps a reference to the numpy array. Some drivers may implicitly update the numpy array after completion (best-effort). Do not rely on implicit behaviour — perform explicit copy-from-device if deterministic correctness is required.
- Lifetime: buffers live at least until the command buffer completes. Do not free or reuse host memory until after launch_kernel() returns.

7. Error classes (when to expect each)
- MetalRuntimeError
  - Generic runtime error for unexpected failures (device missing, bind failure, execution timeout, GPU error wrap).
- KernelNotFoundError
  - Raised when the named kernel cannot be located in the bound Metal library (lookup/language mismatch).
- PipelineCreationError
  - Raised when creating or setting the compute pipeline state fails.
- ResourceError
  - Raised for argument binding/allocation failures (unsupported arg types, buffer allocation failures, etc).
- Stub behaviour
  - On non-Darwin / missing frameworks the bind returns a stub handle; calling launch_kernel() on it raises a plain RuntimeError with a clear message to preserve historical tests.

8. Platforms and prerequisites
- Full functionality requires:
  - macOS (sys.platform == "darwin")
  - Xcode command-line tools on PATH (xcrun, metal, metallib) for compiler usage
  - PyObjC & access to Metal frameworks (import objc, import Metal)
- On non-Darwin platforms or when PyObjC is unavailable the runtime returns a lazy-failing stub handle. This is intentional to keep CI/test imports deterministic and to let test code skip macOS-only smoke tests cleanly.
- See [`third_party/metal/backend/compiler.py`](third_party/metal/backend/compiler.py:1) for the compilation flow using xcrun/metal/metallib.

9. Minimal copy-pastable example
- Example demonstrates: compile (reflection=True) → load via driver → launch kernel → readback.
- This example uses pytest-style tmp_path for temporary files if needed and assumes a macOS environment with toolchain + PyObjC. Guard with platform checks in real tests.

```python
# python
from third_party.metal.backend import compiler as metal_compiler_mod
from third_party.metal.backend import driver as metal_driver_mod
import numpy as np

# Minimal Metal kernel source (writer_kernel writes 42.0 to buffer[0])
SRC = '''
#include <metal_stdlib>
using namespace metal;
kernel void writer_kernel(device float* out [[buffer(0)]]) { out[0] = 42.0; }
'''

# Compile (requires xcrun/metal/metallib on PATH)
compiler = metal_compiler_mod.MetalCompiler()
binary, metadata = compiler.compile(SRC, options={}, reflection=True)

# Bind library via driver (delegates to runtime.bind_library)
driver = metal_driver_mod.MetalDriver()
handle = driver.load_binary(binary, metadata=metadata)

# Prepare host buffer
out = np.zeros(1, dtype=np.float32)

# Launch kernel (grid/block as 3-tuples)
res = handle.launch_kernel(name="writer_kernel", args=(out,), grid=(1,1,1), block=(1,1,1), timeout=5.0)
assert res["status"] == "ok"

# Best-effort readback: some runtimes update the numpy array in-place; if not, perform explicit readback.
print("duration_ms:", res["duration_ms"], "gpu_error:", res["gpu_error"])
```

Notes:
- The compiler.compile(..., reflection=True) returns (binary, metadata) and the compiler attempts to parse [[ buffer(N) ]] annotations to populate metadata["kernels"].
- Example is copy-paste runnable on macOS with required prerequisites; on other platforms callers should run the tests that guard for platform/tool presence.

10. Testing notes for test authors
- The guarded smoke test to exercise end-to-end behaviour is [`python/test/backend/metal_runtime_test.py`](python/test/backend/metal_runtime_test.py:1).
- Expected skip conditions (the test skips with clear reasons):
  - platform.system() != "Darwin"
  - PyObjC import failure
  - xcrun/metal/metallib not on PATH
  - Compilation failures
  - Runtime bind failures (device==None or library load failed)
  - Kernel launch failures (the test will skip rather than fail when these prerequisites are not satisfied)
- Unit tests should use monkeypatch to substitute a fake handle when simulating specific errors (see existing tests for examples).

11. Minimal troubleshooting (quick checks)
- PyObjC selector mismatch / import errors:
  - Symptom: bind_library returns a stub handle or raises on import.
  - Check: ensure PyObjC installed (pip) and that "import Metal" succeeds in a Python REPL on macOS.
- device == None:
  - Symptom: bind_library raises MetalRuntimeError("Failed to obtain system Metal device.")
  - Check: machine has a Metal-capable GPU; system may run in headless CI image without GPU support.
- Pipeline creation failures:
  - Symptom: PipelineCreationError when creating pipeline for kernel name.
  - Check: kernel name matches compiled symbol; ensure function signature and types match intended arguments.
- Scalar size mismatches:
  - Symptom: kernel reads incorrect values for int/float arguments.
  - Check: runtime packs scalars as 64-bit (q/d); ensure kernel expects matching POD sizes or use buffers for explicit control.
- Verbose logging:
  - The runtime uses loguru. Enable DEBUG logs for the module or test runner to see bind/compile messages and detailed exceptions.

12. Acceptance
- Documentation should allow an engineer to:
  - Locate the runtime API: [`third_party/metal/backend/runtime.py`](third_party/metal/backend/runtime.py:1).
  - Compile a minimal kernel with [`third_party/metal/backend/compiler.py`](third_party/metal/backend/compiler.py:1).
  - Run the guarded smoke test: [`python/test/backend/metal_runtime_test.py`](python/test/backend/metal_runtime_test.py:1) and understand why it may skip.
- Tests and smoke checks that exercise the runtime:
  - python/test/backend/metal_runtime_test.py::test_macos_smoke_launch_guarded (macOS guarded smoke)
  - unit tests that monkeypatch bind_library to validate error propagation (see file for examples).

13. CI recommendation
- Run full Metal backend integration on a macOS runner with:
  - Xcode command-line tools installed (xcrun/metal/metallib).
  - PyObjC available in the runner Python environment.
  - Prefer a runner with a Metal-capable GPU; otherwise accept that smoke tests will be skipped.
- Keep non-mac runners in CI to run import/unit tests — the runtime returns stubs so these environments remain deterministic.

14. TODO / uncertainties
- PyObjC selector differences across macOS / Python versions: runtime uses best-effort fallbacks (alternate selectors) but platform-specific selector mismatches should be validated on CI images. If you encounter selector-related exceptions, capture the exact exception and macOS/PyObjC versions for follow-up.
- Example assumes implicit readback may work on some setups; tests that require deterministic readback should implement explicit device→host copy path.

References
- Implementation: [`third_party/metal/backend/runtime.py`](third_party/metal/backend/runtime.py:1)
- Compiler / reflection parsing: [`third_party/metal/backend/compiler.py`](third_party/metal/backend/compiler.py:1)
- Tests: [`python/test/backend/metal_runtime_test.py`](python/test/backend/metal_runtime_test.py:1)
