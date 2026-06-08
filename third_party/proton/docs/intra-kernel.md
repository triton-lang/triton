# Proton Intra-Kernel Profiling

This guide shows how to use Proton's intra-kernel profiler for detailed
performance analysis of GPU kernels written in Triton DSL and Gluon DSL.
Runnable files live in [`../tutorials/intra_kernel/`](../tutorials/intra_kernel/).

## Overview

The Proton intra-kernel profiler captures fine-grained timing information within GPU kernels, enabling performance bottleneck identification and optimization opportunities. This tutorial provides two distinct profiling approaches:

- **TTGIR override approach**: Profile existing Triton DSL kernels by
  injecting instrumentation into dumped TTGIR.
- **Proton DSL approach**: Add embedded profiling scopes directly in Triton or
  Gluon DSL kernels.

## Examples

### 1. TTGIR Override Approach

Example file:
[`../tutorials/intra_kernel/example_override.py`](../tutorials/intra_kernel/example_override.py)

Use this approach to profile existing Triton DSL kernels without modifying
source code. The example is a vector addition kernel with external
instrumentation injection.

Run these commands from `third_party/proton/tutorials/intra_kernel/`.

1. Generate TTGIR dump files:

   ```bash
   ../../scripts/dump_ttgir.sh python3 example_override.py --increase-accuracy
   ```

   Creates original TTGIR files in `ttgir_dump/` directory

2. Insert profiling instrumentation:

   ```bash
   ./insert_proton_records
   ```

   Modifies TTGIR files by adding `proton.record` operators at profiling points

3. Execute with TTGIR override:

   ```bash
   TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=ttgir_dump python3 example_override.py --increase-accuracy
   ```

Environment variables:

- `TRITON_ALWAYS_COMPILE=1` forces recompilation on each run.
- `TRITON_KERNEL_OVERRIDE=1` enables TTGIR override.
- `TRITON_OVERRIDE_DIR=ttgir_dump` points Triton at the modified TTGIR files.

### 2. Proton DSL Approach

Example file:
[`../tutorials/intra_kernel/example_dsl.py`](../tutorials/intra_kernel/example_dsl.py)

Use this approach for native profiling DSL integration in Triton and Gluon DSL
kernels. The example includes Triton vector addition and Gluon matrix
multiplication using NVIDIA Hopper features such as WGMMA and TMA.

```bash
# Timeline trace mode (default)
python3 example_dsl.py

# Operation measurement mode
python3 example_dsl.py --op-measure

# Enable warp sampling with specific warp IDs
python3 example_dsl.py --warp-sampling --warp-ids "0,1,2,3" --gmem_buffer

# High accuracy profiling
python3 example_dsl.py --increase-accuracy
```

Run these commands from `third_party/proton/tutorials/intra_kernel/`.

## DSL Semantics

For DSL-level instrumentation, only Gluon semantic is enabled by default.
Instrumenting Triton DSL kernels is disabled because high-level Triton IR can be
rewritten by loop pipelining, instruction reordering, IR duplication, and other
compiler transformations. Those transformations can move naive instrumentation
and produce misleading measurements.

To enable Triton DSL instrumentation explicitly:

```python
import triton.profiler.language as pl

pl.enable_semantic("triton")
```

Kernel-side instrumentation:

```python
import triton.profiler.language as pl

pl.enter_scope("load")
...
pl.exit_scope("load")

with pl.scope("compute"):
    ...
```

Advanced users can insert `proton.record start` and `proton.record end`
directly in TTIR or TTGIR.

## Understanding Timeline Traces

### Time Representation

- **Scope duration**: Displayed in cycles for precise measurement.
- **Threadblock start times**: Measured in nanoseconds using global timing.
- **Chrome trace format**: Assumes 1GHz GPU frequency for consistent time units.

### Circular Buffer System

- **Backend storage**: Uses a circular buffer for runtime profiling on each CTA.
- **Buffer overflow**: When full, earlier events are dropped and trace
  generation emits warnings.
- **Event window**: Displays the latest recorded event window in the timeline.

### Finalize Time Measurement

- **Definition**: Captures `Finalize Time` when kernel execution completes.
- **Meaning**: Shows the overhead of dumping profiling data from the buffer to
  global memory. It appears as a field in the Chrome trace viewer tab.

## Configuration Options

### Profiling Accuracy

| Option | Description | Use Case |
|--------|-------------|----------|
| `clock32` | Records events in 32-bit clock format for lower overhead | Normal kernels under roughly 4 seconds at 1GHz |
| `time_shift` | Deducts constant profiling overhead from timeline trace | Mitigate Proton runtime overhead for cleaner traces |
| `sched_stores` | Provides more cycle-accurate operation latency measurement | Accurate single operation latency measure |
| `sched_barriers` | Constrains AMD instruction scheduling within proton scopes | AMD GPU profiling |

### Buffer Configuration

| Buffer Type | Options | Default | Description |
|-------------|---------|---------|-------------|
| `buffer_type` | `shared`, `global` | `shared` | Determines whether profiling data is stored in shared or global memory |
| `buffer_size` | Integer | `shared`: maximum size without reducing occupancy; `global`: 16KB x number of profiled units, such as warps | Controls per-block profiling buffer size in bytes |

### Sampling Configuration

| Parameter | Options | Description |
|-----------|---------|-------------|
| `sampling_strategy` | `selective`, `none` | Sampling approach for profiling data collection |
| `sampling_options` | Comma-separated warp IDs | Specific warps to profile (e.g., "0,1,2,3") |

**Sampling Benefits**: Warp sampling captures more events within the same buffer size constraint by focusing on specific warps of interest.

## Output Formats

### Timeline Traces

- **Format**: Chrome trace format (`.chrome_trace` files)
- **Viewer**: Chrome browser at `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/)
- **Content**: Detailed timeline with scope durations

### Operation Measurements

- **Format**: Hatchet format (`.hatchet` files)
- **Viewer**: `proton-viewer -m normalized_cycles <filename>.hatchet`
  (`-m cycles` shows the sum of all cycles across the GPU;
  `normalized_cycles` shows per-warp averaged cycles)
- **Content**: Scope-level performance metrics and statistics
- **Note**: Cycle counts are averaged across warps/CTAs
