"""Run ConSan helper scenarios in a child process for the CDNA4 and CDNA5 ConSan tests.

A device assertion stops the process that launched the kernel, so each scenario
runs in its own process, never in the pytest process.
"""
import subprocess
import sys

import pytest


def run_consan_helper(helper_script, test_name, *args, timeout=120):
    """Run ``test_name`` from ``helper_script`` in a child process and return its stderr."""
    command = [sys.executable, helper_script, test_name] + [str(a) for a in args]
    return run_consan_command(command, timeout=timeout)


def run_consan_command(command, *, timeout):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail(f"ConSan helper {command} timed out after {timeout}s.\n"
                    f"stdout:\n{stdout.decode(errors='replace')}\n"
                    f"stderr:\n{stderr.decode(errors='replace')}")
    stdout = stdout.decode(errors="replace")
    stderr = stderr.decode(errors="replace")
    # A device assertion may stop execution before normal completion;
    # the calling test checks the expected diagnostic.
    # If no assertion was reported, require a successful exit and the
    # completion marker printed after device synchronization.
    if "device assertion failed" not in stderr:
        assert proc.returncode == 0, (f"ConSan helper {command} exited with {proc.returncode}.\n"
                                      f"stdout:\n{stdout}\nstderr:\n{stderr}")
        assert "ConSan helper completed" in stdout.splitlines(), (f"ConSan helper {command} did not complete.\n"
                                                                  f"stdout:\n{stdout}\nstderr:\n{stderr}")
    return stderr
