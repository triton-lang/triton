import os
import subprocess
import sys
import sysconfig

import pytest


def _run_python(source, *, force_no_gil=True):
    env = os.environ.copy()
    if force_no_gil:
        env["PYTHON_GIL"] = "0"
    else:
        env.pop("PYTHON_GIL", None)
    return subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(sysconfig.get_config_var("Py_GIL_DISABLED") != 1, reason="requires a free-threaded Python build")
def test_native_specialize_concurrent_first_use():
    result = _run_python("import gc, threading\n"
                         "from concurrent.futures import ThreadPoolExecutor\n"
                         "from triton._C.libtriton import native_specialize_impl\n"
                         "from triton.backends.compiler import BaseBackend\n"
                         "start = threading.Barrier(17)\n"
                         "def specialize(_):\n"
                         "    start.wait(10)\n"
                         "    return native_specialize_impl(BaseBackend, 2, False, False, True)\n"
                         "def collect():\n"
                         "    start.wait(10)\n"
                         "    for _ in range(8):\n"
                         "        gc.collect()\n"
                         "with ThreadPoolExecutor(17) as pool:\n"
                         "    results = [pool.submit(specialize, worker) for worker in range(16)]\n"
                         "    collector = pool.submit(collect)\n"
                         "    assert [future.result(10) for future in results] == [('i32', None)] * 16\n"
                         "    collector.result(10)\n")
    assert result.returncode == 0, result.stdout + result.stderr
