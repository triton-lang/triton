# Advanced Features

## CUDA Graphs

Proton supports profiling CUDA graph replay on NVIDIA GPUs.

Start profiling before graph capture. Proton records the call path where the
kernel is captured and the call path where the graph is launched.

```python
import torch
import triton.profiler as proton

session = proton.start("profile_name", context="shadow")

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    with proton.scope("captured_region"):
        kernel[grid](...)

proton.deactivate(session)

proton.activate(session)
with proton.scope("graph_replay"):
    graph.replay()

proton.finalize(session)
```

The replayed call path contains the special `<captured_at>` frame:

```text
graph_replay -> <captured_at> -> captured_region -> kernel_name
```

Flexible metrics from scopes and launch metadata are aggregated for graph
replays the same way they are for individually launched kernels.

## Knobs

Triton's runtime has a centralized configuration system called knobs. Proton
defines these knobs:

| Knob | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `triton.knobs.proton.disable` | `TRITON_PROTON_DISABLE` | `False` | Disable Proton profiling calls. |
| `triton.knobs.proton.enable_nvtx` | `TRITON_ENABLE_NVTX` | `True` | Enable NVTX ranges in Proton. |
| `triton.knobs.proton.cupti_lib_dir` | `TRITON_CUPTI_LIB_PATH` | `<triton_root>/backends/nvidia/lib/cupti` | CUPTI library directory. |
| `triton.knobs.proton.cupti_lib_blackwell_dir` | `TRITON_CUPTI_LIB_BLACKWELL_PATH` | `<triton_root>/backends/nvidia/lib/cupti-blackwell` | CUPTI library directory for Blackwell and newer GPUs. |
| `triton.knobs.proton.profile_buffer_size` | `TRITON_PROFILE_BUFFER_SIZE` | `67108864` | Activity/profile buffer size in bytes. |
| `triton.knobs.proton.profile_metric_buffer_size` | `TRITON_PROFILE_METRIC_BUFFER_SIZE` | `67108864` | GPU metric buffer size for flexible metrics during CUDA graph launches. |

## Thread Management

Calls into `libproton.so`, such as `enter_scope`, are synchronized with
explicit locks. Operations that do not call into `libproton.so`, including
CUDA/HIP callbacks, use separate locks for data structures that may be accessed
by multiple threads.

## Known Issues

### Instruction Sampling Permissions

If NVIDIA instruction sampling reports permission errors, see NVIDIA's
performance counter permission guidance:

https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters

Instruction sampling currently has high end-to-end overhead because Proton does
not yet use continuous sampling. Continuous sampling could reduce overhead, but
it complicates attribution for concurrent kernels and requires separate sample
correlation logic.

### AMD Visible Devices

`HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are not supported by Proton on
AMD GPUs. Use `ROCR_VISIBLE_DEVICES` instead.

## Third-Party Backends

Out-of-tree Triton backends can register Proton support by adding a `proton`
directory to the plugin and using Proton's CMake helper functions.

```text
triton-plugin/
|-- <triton-plugin files>
`-- proton/
    |-- CMakeLists.txt
    |-- MyBackendSourceFile.cpp
    `-- <implementation files>
```

Example `CMakeLists.txt`:

```cmake
add_proton_backend(MyBackend
    MyBackendNamespace
    MyBackendSourceFile.cpp
)

add_proton_device_type(MY_DEVICE)
add_proton_backend_external_lib(MyProfilingApi)
```

The registration source file provides `registerProtonBackend` in the namespace
passed to `add_proton_backend`:

```c++
namespace MyBackendNamespace {

proton::BackendRegistration registerProtonBackend() {
  return {
      proton::ProfilerRegistration{
          "MyBackend", "TritonBackendForThisProfiler",
          []() -> proton::Profiler * {
            return &MyProfilerImplementation::instance();
          }},
      proton::DeviceRegistration{
          "MY_DEVICE", proton::DeviceType::MY_DEVICE,
          [](uint64_t index) { return getMyDevice(index); }},
      proton::RuntimeRegistration{
          "MY_DEVICE",
          []() -> proton::Runtime * {
            return &MyRuntimeImplementation::instance();
          }},
  };
}

}
```

Each registration field is optional, so a backend can register only the Proton
extension points it supports. Backend implementations should follow the existing
`CuptiProfiler` and `RoctracerProfiler` patterns for callback correlation,
runtime events, and metric insertion.
