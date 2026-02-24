"""This plugin runs tests in a new subprocess, so they can fail assertions without breaking the CUDA context.

The implementation is a derivative of pytest-forked, whose license is included below:

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""

import marshal
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from _pytest import runner

_SPAWN_CHILD_ENV = "TRITON_PYTEST_SPAWNED_CHILD"
_SPAWN_REPORT_ENV = "TRITON_PYTEST_SPAWNED_REPORT"
_SPAWNED_REPORTS = None


def _serialize_report(rep):
    data = rep.__dict__.copy()
    if hasattr(rep.longrepr, "toterminal"):
        data["longrepr"] = str(rep.longrepr)
    else:
        data["longrepr"] = rep.longrepr
    for name, value in list(data.items()):
        if hasattr(value, "strpath"):
            data[name] = str(value)
        elif name == "result":
            data[name] = None
    return data


def _spawned_process_crash_report(item, returncode, stdout, stderr):
    from _pytest._code import getfslineno

    path, lineno = getfslineno(item)
    if returncode < 0:
        crash_info = f"{path}:{lineno}: spawned test subprocess crashed with signal {-returncode}"
    else:
        crash_info = f"{path}:{lineno}: spawned test subprocess exited with code {returncode}"

    has_from_call = getattr(runner.CallInfo, "from_call", None) is not None
    if has_from_call:
        call = runner.CallInfo.from_call(lambda: 0 / 0, "???")
    else:
        call = runner.CallInfo(lambda: 0 / 0, "???")
    call.excinfo = crash_info
    rep = runner.pytest_runtest_makereport(item, call)
    if stdout:
        rep.sections.append(("captured stdout", stdout))
    if stderr:
        rep.sections.append(("captured stderr", stderr))
    return rep


def _run_spawned_test(item):
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "pytest_report"
        env = os.environ.copy()
        env.pop("PYTEST_CURRENT_TEST", None)
        for name in list(env):
            if name.startswith("PYTEST_XDIST_"):
                env.pop(name)
        env[_SPAWN_CHILD_ENV] = "1"
        env[_SPAWN_REPORT_ENV] = report_path
        nodeid_suffix = item.nodeid.partition("::")[2]
        item_path = getattr(item, "path", None)
        if item_path is None:
            item_path = getattr(item, "fspath")
        test_target = Path(item_path).resolve()
        if nodeid_suffix:
            test_target = f"{test_target}::{nodeid_suffix}"

        cmd = [sys.executable, "-m", "pytest", "--rootdir", str(item.config.rootpath), test_target]
        device = item.config.getoption("device")
        if device is not None:
            cmd.extend(["--device", device])

        result = subprocess.run(
            cmd,
            capture_output=True,
            cwd=str(item.config.invocation_params.dir),
            env=env,
            errors="replace",
            text=True,
        )
        if report_path.stat().st_size:
            with report_path.open("rb") as report_file:
                reports = marshal.loads(report_file.read())
            return [runner.TestReport(**report) for report in reports]
        return [_spawned_process_crash_report(item, result.returncode, result.stdout, result.stderr)]


def pytest_configure(config):
    config.addinivalue_line("markers", "spawned: run the test in a fresh pytest subprocess")
    global _SPAWNED_REPORTS
    _SPAWNED_REPORTS = [] if os.environ.get(_SPAWN_REPORT_ENV) else None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item):
    if os.environ.get(_SPAWN_CHILD_ENV) or not item.get_closest_marker("spawned"):
        return None

    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    for rep in _run_spawned_test(item):
        ihook.pytest_runtest_logreport(report=rep)
    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def pytest_runtest_logreport(report):
    if _SPAWNED_REPORTS is not None:
        _SPAWNED_REPORTS.append(_serialize_report(report))


def pytest_sessionfinish(session, exitstatus):
    if _SPAWNED_REPORTS is None:
        return

    with open(os.environ[_SPAWN_REPORT_ENV], "wb") as report_file:
        report_file.write(marshal.dumps(_SPAWNED_REPORTS))
