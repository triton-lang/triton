# Periodic Profiling

Periodic profiling splits one profiling session into numbered phases and flushes
completed phases while the workload is still running. Use it for long-running
jobs where a single profile would become too large, consume too much host
memory, or be unavailable until process exit.

Periodic profiling is enabled with the `periodic_flushing` mode on the regular
GPU profiling backends:

- `cupti`
- `rocprofiler`
- `roctracer`

The instrumentation backend has its own buffering modes; this page covers
host-side periodic flushing for regular GPU activity profiles.

## Why Use It

Without periodic flushing, Proton keeps profile data for the active session
until `finalize()`. That is simple for short scripts, but it is less practical
for training jobs, servers, or benchmark loops with many repeated iterations.

Periodic flushing is useful when you want to:

- Bound the amount of profile data kept in memory.
- Produce partial profile files for each window of work.
- Read or clear older profile phases while newer phases continue collecting.
- Avoid losing all profile output if a long run exits before the end.

## Enable Periodic Flushing

Pass `mode="periodic_flushing"` to `proton.start`.

```python
import triton.profiler as proton

session = proton.start(
    "profile_name",
    mode="periodic_flushing",
)
```

The default periodic output format is `hatchet`.

To choose a format explicitly:

```python
session = proton.start(
    "profile_name",
    mode="periodic_flushing:format=hatchet_msgpack",
)
```

Supported formats are:

| Format | Output |
| --- | --- |
| `hatchet` | JSON Hatchet tree files. |
| `hatchet_msgpack` | MessagePack Hatchet tree files. |
| `chrome_trace` | Chrome trace JSON files. |

## Phases

Each session starts in phase `0`. Calls to `proton.data.advance_phase(session)`
move subsequent profiling records into the next phase and return that new phase
number.

```python
session = proton.start(
    "train_profile",
    mode="periodic_flushing:format=hatchet",
)

for step in range(num_steps):
    with proton.scope(f"step_{step}"):
        train_step()

    if (step + 1) % 100 == 0:
        phase = proton.data.advance_phase(session)
        print(f"advanced to phase {phase}")

proton.finalize(session)
```

In this example, phase `0` contains steps `0..99`, phase `1` contains steps
`100..199`, and so on.

## What Gets Flushed

The current phase is still open, so Proton only treats earlier phases as
complete. Internally, periodic flushing writes phases up to
`current_phase - 1`.

This means:

- Call `advance_phase()` after a logical window of work if you want that window
  to become eligible for periodic flushing before the session ends.
- The currently active phase is written by `finalize()` if it still contains
  data.
- `deactivate(session, flushing=True)` flushes device-side profiler buffers and
  marks previous phases complete, but it does not make the current phase a past
  phase.

Periodic output files are named with the phase number:

```text
<profile_path>.part_<phase>.<format>
```

For example:

```text
train_profile.part_0.hatchet
train_profile.part_1.hatchet
train_profile.part_2.hatchet
```

When periodic flushing writes a completed phase, Proton clears that phase from
the in-memory data store. This keeps memory bounded during long runs.

## Reading Phase Data In Memory

The `triton.profiler.data` module exposes experimental APIs for reading and
clearing phase data directly.

```python
session = proton.start(
    "profile_name",
    mode="periodic_flushing:format=hatchet",
)

phase0 = 0
run_phase_0()
phase1 = proton.data.advance_phase(session)

# Make an explicit flush point. get() and get_msgpack() do not synchronize the
# device by themselves.
proton.deactivate(session, flushing=True)

# Note that you can still call `is_phase_complete()` without flushing so that
# CPU and GPU work can proceed in parallel.
if proton.data.is_phase_complete(session, phase0):
    data = proton.data.get(session, phase0)
    proton.data.clear(session, phase0)

proton.activate(session)
run_phase_1()

proton.finalize(session)
```

Available APIs:

| API | Description |
| --- | --- |
| `proton.data.advance_phase(session)` | Advance the active data phase and return the next phase number. |
| `proton.data.is_phase_complete(session, phase)` | Return whether all device-side records for `phase` have been flushed to the host and the phase will no longer receive new records. |
| `proton.data.get(session, phase=0)` | Return JSON-compatible profile data for a phase. |
| `proton.data.get_msgpack(session, phase=0)` | Return MessagePack profile data for a phase. |
| `proton.data.clear(session, phase=0, clear_up_to_phase=False)` | Clear one phase, or all phases up to and including `phase`. |

Use `clear_up_to_phase=True` after consuming a sequence of completed phases:

```python
if proton.data.is_phase_complete(session, last_phase_to_clear):
    proton.data.clear(
        session,
        phase=last_phase_to_clear,
        clear_up_to_phase=True,
    )
```

## CUDA Graphs

Periodic flushing also works with CUDA graph replay on NVIDIA GPUs. Start
profiling before graph capture, use `hook="triton"` if you need launch metadata,
and advance phases around replay windows.

```python
session = proton.start(
    "graph_profile",
    mode="periodic_flushing:format=hatchet",
    hook="triton",
)

# Warm up and capture the graph after Proton starts.
...

for replay_id in range(num_replays):
    graph.replay()

    if (replay_id + 1) % 50 == 0:
        proton.data.advance_phase(session)

proton.finalize(session)
```

## Output And Viewing

View each Hatchet phase independently:

```bash
proton-viewer -m time/ns train_profile.part_0.hatchet
proton-viewer -m time/ns train_profile.part_1.hatchet
```

For MessagePack output, pass the `.hatchet_msgpack` file to tools that support
Proton's MessagePack Hatchet representation.

For Chrome trace output, open each `.chrome_trace` part in `chrome://tracing` or
[Perfetto](https://ui.perfetto.dev/).

## Tuning

The phase size is a workload choice. Larger phases create fewer files and reduce
flush overhead. Smaller phases bound memory more aggressively and make partial
results available sooner.

The underlying GPU activity buffer size is controlled by:

```text
TRITON_PROFILE_BUFFER_SIZE
```

or:

```python
triton.knobs.proton.profile_buffer_size
```

For diagnostics, set `PROTON_DATA_FLUSH_TIMING=1` to print timing information
for phase serialization, file writes, clearing, and CUDA graph metric peeking.
