# Command Line And Viewer

## `proton`

The `proton` command profiles Python scripts and pytest invocations without
adding `proton.start` and `proton.finalize` calls to the target.

```bash
proton [options] script.py [script_args]
proton [options] pytest [pytest_args]
python -m triton.profiler.proton [options] script.py [script_args]
```

Common options:

| Option | Description |
| --- | --- |
| `-n, --name` | Profile output name without suffix. |
| `-b, --backend` | Profiling backend. |
| `-c, --context` | `shadow` or `python`. |
| `-m, --mode` | Backend-specific mode. |
| `-d, --data` | `tree` or `trace`. |
| `-k, --hook` | Profiling hook, currently `triton`. |

In command-line mode, Proton starts before the target runs and finalizes after
the target exits. `proton.start` and `proton.finalize` calls inside the target
script are ignored. Only session `0` is valid in this mode.

Examples:

```bash
proton -n dynamic_net tutorials/dynamic-net.py
proton -n tests pytest third_party/proton/test/test_cmd.py -s --tb=short
proton -n trace -d trace script.py
proton -n samples -b cupti -m pcsampling script.py
```

## `proton-viewer`

`proton-viewer` prints Hatchet tree profiles in the terminal.

```bash
pip install llnl-hatchet
proton-viewer -m time/s profile_name.hatchet
```

Use `llnl-hatchet`; `pip install hatchet` installs a different package with an
incompatible API.

List available raw and derived metrics:

```bash
proton-viewer --list profile_name.hatchet
```

Display up to two metrics:

```bash
proton-viewer -m time/ns,time/% profile_name.hatchet
```

Common derived metrics include:

- `time/s`, `time/ms`, `time/us`, `time/ns`
- `avg_time/s`, `avg_time/ms`, `avg_time/us`, `avg_time/ns`
- `flop/s`, `gflop/s`, `tflop/s`, and width-specific variants such as
  `tflop16/s`
- `byte/s`, `gbyte/s`, `tbyte/s`
- `util`
- `<metric>/%` for inclusive metric percentages

## Filtering And Formatting

Show paths that pass through frames matching a regular expression:

```bash
proton-viewer -m time/ns -i ".*matmul.*" profile_name.hatchet
```

Exclude matching frames and their children:

```bash
proton-viewer -m time/ns -e ".*warmup.*" profile_name.hatchet
```

Hide frames below a threshold for the first displayed metric:

```bash
proton-viewer -m time/ns -t 1000 profile_name.hatchet
```

Limit tree depth:

```bash
proton-viewer -m time/ns -d 4 profile_name.hatchet
```

Format Python frame names:

```bash
proton-viewer -m time/ns -f file_function_line profile_name.hatchet
```

Supported frame formats are `full`, `file_function_line`, `function_line`, and
`file_function`.

Sort output by metric value instead of chronological order:

```bash
proton-viewer -m time/ns profile_name.hatchet --print-sorted
```

## Diff Profiles

Compare two profiles with `--diff-profile` or `-diff`. The viewer computes:

```text
current_profile[metric] - diff_profile[metric]
```

Example:

```bash
proton-viewer -m time/ns --diff-profile before.hatchet after.hatchet
```

## Trace Visualization

Use `data="trace"` to write Chrome trace output:

```python
proton.start("profile_name", data="trace")
```

Open `profile_name.chrome_trace` in `chrome://tracing` or
[Perfetto](https://ui.perfetto.dev/).

The instrumentation backend also emits trace output when profiling intra-kernel
timelines. See [intra-kernel profiling](intra-kernel.md).
