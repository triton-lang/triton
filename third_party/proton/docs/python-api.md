# Python Profiling API

Import Proton through Triton's profiler package:

```python
import triton.profiler as proton
```

## Sessions

Use `start`, `activate`, `deactivate`, and `finalize` to control profiling
regions.

```python
session = proton.start("profile_name", context="shadow")

with proton.scope("step"):
    kernel[grid](...)

proton.deactivate(session)

# Work done here is skipped.

proton.activate(session)

with proton.scope("next_step"):
    kernel[grid](...)

proton.finalize(session)
```

`proton.start` accepts:

- `name`: output path without the format suffix. If omitted, Proton writes
  `proton.<suffix>`.
- `context`: `shadow` for Proton scope names or `python` for Python
  file/function/line call paths.
- `data`: `tree` for Hatchet output or `trace` for Chrome trace output.
- `backend`: `cupti`, `rocprofiler`, `roctracer`, `instrumentation`, or `None`
  for automatic selection.
- `mode`: backend-specific mode string or mode object.
- `hook`: `triton` for launch metadata or a custom hook instance.

If multiple sessions are active, `activate()`, `deactivate()`, and `finalize()`
without an explicit session apply to all sessions.

## Function Profiling

`proton.profile` wraps a function and starts a session before calling it.
Finalize the session after the wrapped function returns.

```python
@proton.profile(name="profile_name", context="python")
def run():
    kernel[grid](...)

run()
proton.finalize()
```

The decorator can also be used without arguments:

```python
@proton.profile
def run():
    kernel[grid](...)
```

## Scopes

Use `proton.scope` to define named regions in `shadow` context.

```python
session = proton.start("profile_name", context="shadow")

with proton.scope("test0"):
    with proton.scope("test1"):
        kernel[grid](...)
with proton.scope("test2"):
    kernel[grid](...)

proton.finalize(session)
```

Scopes can also be used as decorators:

```python
@proton.scope("matmul")
def run_matmul():
    matmul_kernel[grid](...)
```

For manual scope control:

```python
proton.enter_scope("load")
kernel[grid](...)
proton.exit_scope("load")
```

`exit_scope` can omit the name, but passing it helps catch mismatched scope
pairs.

## Metrics

Scopes accept a metrics dictionary. Metric values may be integers, floats,
scalar tensors, or vectors of integers/floats.

```python
with proton.scope("matmul", {"flops16": 2.0e12, "bytes": 256_000}):
    matmul_kernel[grid](...)

with proton.scope("candidates", {"latency_samples": [10.0, 12.0, 11.0]}):
    kernel[grid](...)
```

Metric names can include optional unit and type suffixes:

| Suffix | Meaning |
| --- | --- |
| `(inc)` or no suffix | Inclusive metric, accumulated into parent scopes. |
| `(exc)` | Exclusive metric, kept on the current scope only. |
| `(pty)` | Property metric, not accumulated across repeated scopes. |

Examples:

```python
with proton.scope("outer", {"tokens (inc)": 128}):
    with proton.scope("inner", {"cpu_wait (ns)(exc)": 90}):
        kernel[grid](...)

proton.enter_scope("config", metrics={"tile_size (pty)": 128})
proton.exit_scope()
```

Use one consistent value type for each metric name in a profile. Reusing the
same metric name with incompatible scalar types raises an error.

## CPU Timed Scopes

`cpu_timed_scope` records exclusive CPU wall-clock time under the metric
`cpu_time`.

```python
with proton.cpu_timed_scope("host_preprocessing"):
    prepare_inputs()
```

Compare CPU and GPU time in the viewer:

```bash
proton-viewer -m time/ns,cpu_time/ns profile_name.hatchet
```

## States

`proton.state` customizes the call path of GPU operations.

```python
with proton.scope("iteration"):
    with proton.state("optimizer"):
        optimizer_kernel[grid](...)
```

States differ from scopes:

- A GPU operation has at most one active state.
- The innermost state overwrites outer states.
- The state is appended above each kernel in both `shadow` and `python`
  contexts.

Manual state control is also available:

```python
proton.enter_state("state0")
kernel[grid](...)
proton.exit_state()
```

## Launch Metadata Hook

Pass `hook="triton"` to record metadata returned by Triton
`launch_metadata` callbacks.

```python
from typing import NamedTuple
import triton
import triton.language as tl
import triton.profiler as proton

proton.start("profile_name", hook="triton")

def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
    return {
        "name": "copy_kernel",
        "bytes": args["n_elements"] * 4,
        "flops32": 0.0,
    }

@triton.jit(launch_metadata=metadata_fn)
def copy_kernel(x, y, n_elements: tl.constexpr):
    ...
```

Reserved metadata keys include:

```python
name: str
flops8: float
flops16: float
flops32: float
flops64: float
bytes: int
```

Additional scalar or vector numeric metadata is recorded as custom metrics.

Advanced users can pass a configured `LaunchHook` instance:

```python
from triton.profiler.hooks.launch import LaunchHook

hook = LaunchHook()
hook.configure(include=".*matmul.*")
proton.start("profile_name", hook=hook)
```

`include` and `exclude` are regular expressions over kernel names. Only one
filter direction should be used for a hook configuration.
