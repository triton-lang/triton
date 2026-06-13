# Backends and Modes

Proton automatically selects the profiling backend that matches the active
Triton runtime when `backend=None`.

## Backends

| Backend | Platform | Notes |
| --- | --- | --- |
| `cupti` | NVIDIA GPUs | Default NVIDIA backend. Supports regular profiling, `pcsampling`, and `periodic_flushing`. |
| `rocprofiler` | AMD GPUs | Preferred AMD backend when rocprofiler-sdk is available. Supports regular profiling and `periodic_flushing`. |
| `roctracer` | AMD GPUs | **Deprecated** AMD fallback backend. Supports regular profiling and `periodic_flushing`. |
| `instrumentation` | NVIDIA and AMD GPUs | Intra-kernel instrumentation backend for scope-level cycle metrics inside kernels. |

Examples:

```python
proton.start("nvidia_profile", backend="cupti")
proton.start("amd_profile", backend="rocprofiler")
proton.start("instrumented_profile", backend="instrumentation")
```

On AMD GPUs, `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are not supported
by Proton. Use `ROCR_VISIBLE_DEVICES` instead so profiler device IDs can be
mapped correctly.

## Instruction Sampling

NVIDIA instruction sampling is available through CUPTI:

```python
proton.start(
    "profile_name",
    context="shadow",
    backend="cupti",
    mode="pcsampling",
)
```

Proton currently uses the CUPTI backend's default sampling period; the sampling
interval is not configurable through this mode.

Instruction sampling can add significant end-to-end overhead because Proton
transfers and processes sample data on the CPU. Viewer filters such as
`-i <regex>`, `-d <depth>`, and `-t <threshold>` are useful for narrowing the
output.

## Periodic Flushing

`periodic_flushing` splits long profiling sessions into phases and writes
completed phases while the session is still running. It is supported by
`cupti`, `rocprofiler`, and `roctracer`.

See [periodic profiling](periodic-profiling.md) for phase advancement, output
file naming, in-memory phase APIs, and tuning guidance.

## Instrumentation Backend

The instrumentation backend collects fine-grained intra-kernel measurements.
By default it records cycle metrics for each profiled unit.

```python
import triton.profiler as proton
import triton.profiler.mode as pmode

proton.start(
    "profile_name",
    backend="instrumentation",
    mode=pmode.Default(),
)
```

The string form is also accepted:

```python
proton.start(
    "profile_name",
    backend="instrumentation",
    mode="default:buffer_type=global:buffer_size=16384",
)
```

Instrumentation mode options include:

| Option | Values |
| --- | --- |
| `metric_type` | `cycle` |
| `sampling_strategy` | `none`, `selective` |
| `sampling_options` | Comma-separated unit IDs, such as `0,1,2,3` |
| `granularity` | `cta`, `warp`, `warp_2`, `warp_4`, `warp_8`, `warp_group`, `warp_group_2`, `warp_group_4`, `warp_group_8` |
| `buffer_strategy` | `circular`, `flush` |
| `buffer_type` | `shared`, `global` |
| `buffer_size` | Integer byte count; `0` selects the backend default. |
| `optimizations` | Comma-separated `time_shift`, `sched_stores`, `sched_barriers`, `clock32` |

Mode object example:

```python
mode = pmode.Default(
    sampling_strategy="selective",
    sampling_options="0,1,2,3",
    buffer_type="global",
    optimizations="clock32,time_shift",
)
proton.start("profile_name", backend="instrumentation", mode=mode)
```

See [intra-kernel profiling](intra-kernel.md) for end-to-end examples.
