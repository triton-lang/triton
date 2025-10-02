# Proton Intra-Kernel Profiler Tutorial

A comprehensive tutorial demonstrating how to use the Proton intra-kernel profiler for detailed performance analysis of GPU kernels written in Triton DSL and Gluon DSL.

## Overview

The Proton intra-kernel profiler captures fine-grained timing information within GPU kernels, enabling performance bottleneck identification and optimization opportunities. This tutorial provides two distinct profiling approaches:

- **TTGIR Override Approach** - For profiling existing Triton DSL kernels by injecting instrumentation
- **Proton DSL Approach** - For native integration with Triton and Gluon DSL kernels using embedded profiling scopes

## Examples

### 1. TTGIR Override Approach (`example_override.py`)

**Use Case**: Profile existing Triton DSL kernels without modifying source code

**Example**: Vector addition kernel with external instrumentation injection

**Workflow**:
1. **Generate TTGIR dump files**:
   ```bash
   ../../scripts/dump_ttgir.sh python3 example_override.py --increase-accuracy
   ```
   Creates original TTGIR files in `ttgir_dump/` directory

2. **Insert profiling instrumentation**:
   ```bash
   ./insert_proton_records
   ```
   Modifies TTGIR files by adding `proton.record` operators at profiling points

3. **Execute with TTGIR override**:
   ```bash
   TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=ttgir_dump python3 example_override.py --increase-accuracy
   ```
   - `TRITON_ALWAYS_COMPILE=1`: Forces recompilation on each run
   - `TRITON_KERNEL_OVERRIDE=1`: Enables TTGIR override mechanism
   - `TRITON_OVERRIDE_DIR=ttgir_dump`: Specifies directory with modified TTGIR files

### 2. Proton DSL Approach (`example_dsl.py`)

**Use Case**: Native profiling DSL integration for Triton and Gluon DSL kernels

**Example**: Triton vector-add and Gluon matrix multiplication using NVIDIA Hopper architecture features (WGMMA, TMA)


**Command Line Options**:
```bash
# Timeline trace mode (default)
python3 example_dsl.py

# Operation measurement mode
python3 example_dsl.py --op-measure

# Enable warp sampling with specific warp IDs
python3 example_dsl.py --warp-sampling --warp-ids "0,1,2,3"

# High accuracy profiling
python3 example_dsl.py --increase-accuracy
```

## Understanding Timeline Traces

### Time Representation
- **Scope Duration**: Displayed in cycles for precise measurement
- **Threadblock Start Times**: Measured in nanoseconds using global timing
- **Chrome Trace Format**: Assumes 1GHz GPU frequency for consistent time units (ns)

### Circular Buffer System
- **Backend Storage**: Uses circular buffer for runtime profiling on each CTA
- **Buffer Overflow**: When full, earlier events are dropped with warnings in trace generation
- **Event Window**: Displays sliding window (the latest window) of recorded events in timeline

### Finalize Time Measurement
- **Definition**: Captures `Finalize Time` when kernel execution completes
- **Meaning**: Shows overhead of dumping profiling data from buffer to global memory (appears as a field in Chrome trace viewer tab)

## Configuration Options

### Profiling Accuracy

| Option | Description | Use Case |
|--------|-------------|----------|
| `clock32` | Records events in 32-bit clock format for lower overhead | normal kernels (<4 seconds @ 1GHz) |
| `time_shift` | Deducts constant profiling overhead from timeline trace | Mitigate Proton runtime overhead for cleaner traces |
| `sched_stores` | Provides more cycle-accurate operation latency measurement | Accurate single operation latency measure |
| `sched_barriers` | Constrains AMD instruction scheduling within proton scopes | AMD GPU profiling |

### Buffer Configuration

| Parameter | Options | Description |
|-----------|---------|-------------|
| `buffer_type` | `shared_mem`| Storage location for profiling buffer |
| `buffer_size` | `N` | Byte size of the profiling buffer (default: infer a small fraction of shared memory) |

### Sampling Configuration

| Parameter | Options | Description |
|-----------|---------|-------------|
| `sampling_strategy` | `selective`, `none` | Sampling approach for profiling data collection |
| `sampling_options` | Comma-separated warp IDs | Specific warps to profile (e.g., "0,1,2,3") |

**Sampling Benefits**: Warp sampling captures more events within the same buffer size constraint by focusing on specific warps of interest.

## Output Formats

### Timeline Traces
- **Format**: Chrome trace format (`.chrome_trace` files)
- **Viewer**: Chrome browser at `chrome://tracing` or [`Perfetto`](https://ui.perfetto.dev/)
- **Content**: Detailed timeline with scope durations

### Operation Measurements
- **Format**: Hatchet format (`.hatchet` files)
- **Viewer**: `proton-viewer -m normalized_cycles <filename>.hatchet`
(with `-m cycles` showing sum of all cycles across the GPU, `normalized_cycles` for per-warp averaged cycles)
- **Content**: Scope-level performance metrics and statistics
- **Note**: Cycle counts are averaged across warps/CTAs
