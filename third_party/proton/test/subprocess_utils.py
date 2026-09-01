import os


def clean_rocprofiler_env():
    # rocprofiler-sdk updates the native process environment directly. Passing
    # an explicit Python environment prevents that parent-process bootstrap
    # state from initializing the SDK before Proton in a fresh CLI subprocess.
    # TODO: Remove when https://github.com/ROCm/rocm-systems/pull/5348 is
    # available in the minimum supported rocprofiler-sdk version.
    env = os.environ.copy()
    env.pop("ROCPROFILER_REGISTER_FORCE_LOAD", None)
    env.pop("ROCPROFILER_REGISTER_LIBRARY", None)
    env.pop("ROCP_TOOL_LIBRARIES", None)
    return env
