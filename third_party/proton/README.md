# Proton - A Profiler for Triton

Proton is a lightweight profiler for Triton. It records Python context, user
annotations, Triton launch metadata, and GPU kernel metrics while keeping
runtime overhead and profile size small.

This README is the quick-start entry point. Detailed functionality docs live in
[`docs/`](docs/).

## Installation

Build Triton with Proton enabled:

```bash
git clone https://github.com/triton-lang/triton
cd triton/python
pip install .
```

To build Triton without Proton:

```bash
TRITON_BUILD_PROTON=OFF pip install .
```

## Quick Start

Profile a Python region:

```python
import triton.profiler as proton

session = proton.start("profile_name", context="shadow")

with proton.scope("forward", {"bytes": 1024}):
    kernel[grid](...)

proton.deactivate(session)
proton.finalize(session)
```

Profile a function:

```python
import triton.profiler as proton

@proton.profile(name="profile_name", context="python")
def run():
    kernel[grid](...)

run()
proton.finalize()
```

View the default tree output:

```bash
proton-viewer -m time/s profile_name.hatchet
```

Record a Chrome trace instead:

```python
proton.start("profile_name", data="trace")
```

Open the resulting `.chrome_trace` file in `chrome://tracing` or
[Perfetto](https://ui.perfetto.dev/).

## Common Commands

Profile a script or pytest invocation from the command line:

```bash
proton -n profile_name script.py [script_args]
proton -n profile_name pytest [pytest_args]
python -m triton.profiler.proton -n profile_name script.py
```

Collect NVIDIA instruction samples:

```python
proton.start("profile_name", backend="cupti", mode="pcsampling")
```

Use the intra-kernel instrumentation backend:

```python
import triton.profiler as proton
import triton.profiler.mode as pmode

proton.start(
    "profile_name",
    backend="instrumentation",
    mode=pmode.Default(buffer_type="shared"),
)
```
