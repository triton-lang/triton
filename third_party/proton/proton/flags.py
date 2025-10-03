"""
Centralized, process-local flags with a minimal interface (no environment variables).

Usage:
    from triton.profiler.flags import flags

    # Toggle
    flags.profiling_on = True
    flags.instrumentation_on = False

    # Check
    if flags.command_line:
            ...
"""
from dataclasses import dataclass


@dataclass
class ProfilerFlags:
    # Whether to enable profiling. Default is False.
    profiling_on: bool = False
    # Whether instrumentation is enabled. Default is False.
    instrumentation_on: bool = False
    # Whether the script is run from the command line. Default is False.
    command_line: bool = False


flags = ProfilerFlags()
