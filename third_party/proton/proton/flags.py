"""
This file contains the global flags used in the proton package.
"""

# Whether to enable profiling. Default is False.
profiling_on = False


def set_profiling_on():
    global profiling_on
    profiling_on = True


def set_profiling_off():
    global profiling_on
    profiling_on = False


def get_profiling_on():
    global profiling_on
    return profiling_on
