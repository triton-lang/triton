# Proton - A Profiler for Triton

## Introduction

Proton is a lightweight profiler for Triton that captures rich information about program context, metadata, and GPU kernel performance metrics, while keeping both runtime overhead and profile size minimal.

## Installation

The following command installs the latest version of Proton.

```bash
git clone https://github.com/triton-lang/triton
cd triton/python
pip install .
```

To **not build** Proton, you can set the `TRITON_BUILD_PROTON` environment variable to `OFF`:

```bash
TRITON_BUILD_PROTON=OFF pip install .
```

## Usage

### Basic usage

More examples can be found in the [tutorials](tutorials) directory.

Proton can be used to profile *functions* and *regions* in Python code.

- The following examples demonstrate how to use Proton to profile a simple Python function.

```python
import triton.profiler as proton

# name: The path to the profile data
# context: The method used to annotate the context of each GPU kernel. Currently, "shadow" and "python" are supported.
session_id = proton.profile(func, name="profile_name", context="python")(args)
```

- The following examples demonstrate how to use Proton to profile a region in Python code.

```python
session_id = proton.start(name="profile_name", context="python")
...
# Skip a region
proton.deactivate(session_id)
...
# Restart profiling
proton.activate(session_id)
...
# Write out the profile data and finalize the profiler
proton.finalize()
```

### Scope

Unlike the *python* context that provide users with files, functions, and lines where the GPU kernels are invoked, the *shadow* context provides users with the annotated regions in the code. The following example demonstrates how to use the *shadow* context.

```python
import triton.profiler as proton


session_id = proton.start(name="profile_name", context="shadow")

with proton.scope("test0"):
    with proton.scope("test1"):
        foo[1,](x, y)
with proton.scope("test2"):
    foo[1,](x, y)

...
proton.finalize()
```

The *scope* utility also accepts flexible metrics, provided with a dictionary that maps from a string (metric name) to a value (int or float).
Proton will aggregate the metrics for each scope and write them to the profile data.
It is useful for users to understand the performance of the model at a high level.

```python
with proton.scope("test0", {"bytes": 1000}):
    with proton.scope("test1", {"bytes": 2000}):
        foo[1,](x, y)
with proton.scope("test2", {"bytes": 3000}):
    foo[1,](x, y)
```

#### NVTX compatibility

Proton scopes coexist with NVTX ranges.
NVTX pushes and pops (for example, `torch.cuda.nvtx.range_push`) appear as nested scopes in the Proton profile, letting you correlate custom NVTX annotations with Proton's aggregated metrics.


### Backend and mode

Proton supports three profiling backends: `cupti`, `roctracer`, and `instrumentation`.

- **`cupti`**: Used for NVIDIA GPUs. It supports both the default profiling mode and `pcsampling` (instruction sampling).
- **`roctracer`**: Used for AMD GPUs. It supports only the default profiling mode.
- **`instrumentation`**: Available on both NVIDIA and AMD GPUs, this backend enables collection of custom metrics and advanced instrumentation.

By default, Proton automatically selects either `cupti` or `roctracer` as the backend based on your GPU driver. The `instrumentation` backend offers a wide range of mode options for fine-grained profiling, as detailed in the `mode.py` file.

#### Instruction sampling

Proton supports instruction sampling on NVIDIA GPUs.
You may experience ~20x end-to-end overhead when using instruction sampling, although the overhead for each individual GPU kernel is negligible.
The overhead is mostly caused by data transfer and processing on the CPU.
Additionally, the proton-viewer options `-i <regex> -d <depth> -t <threshold>` can be helpful for filtering out GPU kernels that are not of interest.
The following example demonstrates how to use instruction sampling:

```python
import triton.profiler as proton

proton.start(name="profile_name", context="shadow", backend="cupti", mode="pcsampling")
```

#### Instrumentation

The instrumentation backend allows for detailed, fine-grained profiling of intra-kernel behavior, generating trace or tree views similar to those produced by coarse-grained profiling.
By default, if no `mode` is specified, Proton profiles kernel cycles, which may require shared memory. If there is insufficient shared memory, profiling will abort and a warning will be displayed. Future releases will introduce additional instrumentation modes.

**Host-side usage:**

```python
import triton.profiler as proton

proton.start(
    name="profile_name",
    backend="instrumentation",
    mode="<mode0>=<option0>:<mode1>=<option1>:..."
)

# or

import triton.profiler.mode as pmode

proton.start(
    name="profile_name",
    backend="instrumentation",
    mode=pmode.Default(granularity="warp_2") # collect metrics from every 2 warps
)
```

**Kernel-side usage:**

**Caution**: For DSL level instrumentation, **only Gluon** semantic is enabled by default.
Instrumenting kernels written in Triton DSL is disable because Triton's higher-level IR undergoes
aggressive compiler rewrites (loop pipelining, instruction re-ordering, IR duplication, etc.).
These transformations can invalidate naïve instrumentation and lead to misleading results.
To enable instrumentation for Triton DSL, call `pl.enable_semantic("triton")` before `proton.start`.

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import triton.profiler.language as pl

@gluon.jit
def kernel(...):
    pl.enter_scope("scope0")
    for i in range(iters):
        gl.load(...)
    pl.exit_scope("scope0")
    with pl.scope("scope1"):
        for i in range(iters):
            gl.load(...)
```

Advanced users can instrument either the `ttir` or `ttgir` intermediate representations for even finer-grained measurement. The relevant IR instructions are `proton.record start` and `proton.record end`. This can be combined with the environment variable `TRITON_KERNEL_OVERRIDE=1` for custom kernel overrides. For detailed steps, refer to the Triton [documentation](https://github.com/triton-lang/triton?tab=readme-ov-file#tips-for-hacking) under the **Kernel Override Steps** section. We have also assembled a [tutorial](tutorials/ttgir_override) that demonstrates how to use the IR-based instrumentation.


### Hook

```python
import triton.profiler as proton
from typing import NamedTuple

# hook: When hook="triton", it enables proton to invoke launch_metadata function before launching the GPU kernel
proton.start("profile_name", hook="triton")

def metadata_fn(
    grid: tuple,
    metadata: NamedTuple,
    args: dict
):
    return {"name": "<kernel_name>", "flops8": 1.0}

@triton.jit(launch_metadata=metadata_fn)
def foo(x, y):
    tl.store(y, tl.load(x))
```

The `metadata_fn` function is called before launching the GPU kernel to provide metadata for the GPU kernel, which returns a dictionary that maps from a string (metadata name) to a value (int or float).

Currently, **only the launch hook is supported**. In the dictionary returned by the `metadata_fn` function, we can supply the following keys:

```python
name: str  # The name of the kernel
flops8: float  # The number of 8-bit floating-point operations
flops16: float  # The number of 16-bit floating-point operations
flops32: float  # The number of 32-bit floating-point operations
flops64: float  # The number of 64-bit floating-point operations
bytes: int  # The number of bytes expected to be transferred
```

### Command line

Proton can be used as a command-line tool to profile Python scripts and Pytest tests.
The following examples demonstrate how to use Proton command-line.
Detailed options can be found by running `proton -h`.

```bash
proton [options] script.py [script_args] [script_options]
proton [options] pytest [pytest_args] [script_options]
python -m triton.profiler.proton [options] script.py [script_args] [script_options]
proton --instrument=[instrumentation pass] script.py
```

When profiling in the command line mode, the `proton.start` and `proton.finalize` functions are automatically called before and after the script execution. Any `proton.start` and `proton.finalize` functions in the script are ignored. Also, in the command line mode, only a single *session* is supported.
Therefore, `proton.deactivate(session_id=1)` is invalid, while `proton.deactivate(session_id=0)` is valid.

### Visualizing the profile data

By default, proton profiles are in the *json* format and can be read by *Hatchet*. The following command visualizes the profile data on terminal.

```bash
pip install llnl-hatchet
proton-viewer -m time/s <profile.hatchet>
```

NOTE: `pip install hatchet` does not work because the API is slightly different.

If you want to dump the entire trace but not just the aggregated data, you should set the data option to `trace` when starting the profiler.

```python
import triton.profiler as proton

proton.start(name="profile_name", data="trace")
```

The dumped trace will be in the chrome trace format and can be visualized using the `chrome://tracing` tool in Chrome or the [perfetto](https://perfetto.dev) tool.

In addition visualizing the profile data on terminal through Hatchet. A sorted list of the kernels by the first metric can be done using the --print-sorted flag with proton-viewer

```bash
proton-viewer -m time/ns,time/% <profile.hatchet> --print-sorted
```

More options can be found by running the following command.

```bash
proton-viewer -h
```

## Knobs

Triton's runtime has a centralized configuration system called *knobs* that controls various features and behaviors, including the following knobs are defined for Proton:

- `triton.knobs.proton.enable_nvtx` or `TRITON_ENABLE_NVTX` (default: `True`): Whether to enable NVTX ranges in Proton.

- `triton.knobs.proton.cupti_lib_dir` or `TRITON_CUPTI_LIB_DIR` (default: `<triton_root>/backends/nvidia/lib/cupti`): The directory of the CUPTI library.

## Advanced features and knowledge

### Thread management

We guarantee that any call to `libproton.so`, such as `enter_scope`, is synchronized using explicit locks.
For operations that do not trigger calls to libproton.so—including callbacks to CUDA/HIP APIs—we use separated locks to protect data structures that may be accessed concurrently by multiple threads.
For example, the `enter_op` method in `OpInterface` can be invoked by the main thread that involves triton operators, as well as by helper threads that invoke torch operators.

### `cpu_timed_scope`

`cpu_timed_scope` is a utility that wraps `scope` to measure the CPU time of a scope along with other metrics.
The following example demonstrates how to use `cpu_timed_scope`:

```python
import triton.profiler as proton

with proton.cpu_timed_scope("test"):
    foo[1,](x, y)
```

The `cpu_timed_scope` output metric is referred to as `cpu_time`, while `time` represents accelerator (e.g., GPU) time.
The key distinction between `cpu_time` and `time` lies in their inclusivity: `cpu_time` is exclusive, whereas `time` is inclusive.
This difference arises because the time spent on individual kernels represents the smallest measurable time granularity, and each kernel is mutually exclusive.
This exclusivity allows time to be accurately accumulated across parent scopes for `time`.
In contrast, `cpu_time` measures the time within a specific scope.
Since a parent scope encompasses the time spent in its child scopes, summing `cpu_time` from child scope into parent scope would result in double counting.
To visualize both the CPU and GPU time, we can use the following command:

```bash
proton-viewer -m time/ns,cpu_time/ns <proton.hatchet>
```

### Metrics naming

Custom metrics should follow this format: `metric_name (unit) (type)`.
We prefer no space within the metric name.
`unit` and `type` are optional fields.

There are three types of metrics in proton: inclusive, exclusive, and property metrics.
By default, a metric is inclusive.
The metric types are distinguished by the suffix of their names.
The following table shows the suffix for each type and its meaning:

| Suffix | Name | Meaning |
| --- | --- | --- |
| (inc) or "" | Inclusive metric | The metric is accumulated at a scope and can be propagated to the parent scope. |
| (exc) | Exclusive metric | The metric is accumulated at a scope and cannot be propagated to the parent scope. |
| (pty) | Property metric | The metric is a property of the scope and cannot be accumulated or propagated. |

### State annotation

In addition to `proton.scope`, we can also customize the call path of each GPU operation using `proton.state`.

`state` is different from `scope` in several ways:

1. State is not recursive; each operation can have only a single state. Inner most state will overwrite the outer most state.
2. A states is a suffix, meaning that the original call path will append a state above the name of each kernel.
3. State is compatible with both Python and shadow contexts.

The following example demonstrates a basic use of state:

```python
with proton.scope("test"):
    with proton.state("state0"):
        with proton.scope("test0"):
            foo0[1,](x, y)
        with proton.scope("test1"):
            foo1[1,](x, y)
```

The call path of `foo1` will be `test->test1->state0`.

## Proton *vs* Nsight tools

| Aspect | Proton | Nsight Systems | Nsight Compute |
| --- | --- | --- | --- |
| Runtime overhead | Lower overhead | Higher overhead | Higher overhead |
| Profile size | Compact profiles and traces | Large traces | Large traces |
| Portability | Multi vendor | Nvidia only | Nvidia only |
| Triton insights | Metadata hooks | No hooks | No hooks |
| Metric depth | Lightweight metrics | Timeline metrics | Detailed metrics |

**Runtime overhead.** Proton typically keeps slowdown below roughly 1.5×, even for workloads with many short-lived kernels, because it collects fewer metrics and registers fewer callbacks. Nsight Systems and Nsight Compute both impose higher overhead, though they behave similarly to Proton on purely GPU-bound workloads.

**Profile size.** Proton aggregates kernels that share a calling context, so profile files stay compact—sometimes thousands of times smaller than Nsight traces. Both Nsight tools record each GPU kernel individually, which grows traces quickly during long runs.

**Portability.** Proton already runs on AMD and NVIDIA GPUs and has a roadmap to extend instruction sampling to AMD hardware. Nsight Systems and Nsight Compute target NVIDIA GPUs exclusively.

**Triton insights.** Proton can register Triton-specific hooks that surface kernel metadata for richer analysis, at the cost of a small extra overhead. Neither Nsight tool offers comparable Triton integration.

**Metric depth.** Proton emphasizes lightweight metrics and instruction sampling for portability and fast iteration. Nsight Systems focuses on timeline-oriented metrics for NVIDIA GPUs, while Nsight Compute dives deeper into instruction-level details such as memory transactions and access patterns.

## Known issues

- CUDA graph

`hooks` cannot be used to accurately accumulate the number of FLOPs in CUDA graph mode profiling because kernels are captured and launched separately; metrics are not accumulated when kernels are launched in graph mode. This issue can be circumvented by using `scope` to supply FLOPs.

If profiling is initiated after CUDA graph capturing, there may be minor memory leak issues.
This is because the number of kernels in a graph instance (i.e., `cuGraphExec`) is unknown, preventing the deletion of mappings between the kernel ID and the graph ID.

- Instruction sampling

If you encounter permission related problems when using instruction sampling, you can lookup this [page](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) for help.

The overhead of instruction sampling on NVIDIA GPUs is about 20x using Proton because we haven't enabled continuous sampling yet.
Continuous sampling can allow for more runtime optimizations, but it makes it more challenging to attribute performance data back to the GPU kernels because: (1) it enables profiling of concurrent kernels, (2) it doesn't allow profiling of time and instruction samples simultaneously, and (3) it works best if we have a separate thread dedicated to attributing instruction samples to the GPU kernels

- Visible devices on AMD GPUs

Environment variables such as `HIP_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES` are not supported on AMD GPUs. Once it's set, we cannot find a valid mapping between the device ID returned by RocTracer and the physical device ID. Instead, `ROCR_VISIBLE_DEVICES` is recommended to be used.
