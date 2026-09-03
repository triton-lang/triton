"""Regression tests for `libtriton.extend_with` and interpreter teardown.

These tests do not require a GPU or a built plugin. They guard two invariants:

1. Importing `triton._C.libtriton` and exiting must not segfault. A previous
   refactor stored nanobind Python handles in C++ statics; those were destroyed
   after `Py_Finalize`, causing a use-after-free `Py_DECREF` (SIGSEGV, exit 139)
   at process teardown.

2. `extend_with` walks the module tree (`ir.builder`, `passes`) via
   `py::getattr` at call time. If that structure is renamed/restructured, the
   walk breaks; this test fails loudly so the breakage is caught in CI.
"""

import subprocess
import sys


def test_libtriton_teardown_no_segfault():
    # Import in a fresh subprocess so we can observe the *exit* code: the crash
    # this guards against occurred only during interpreter teardown.
    result = subprocess.run(
        [sys.executable, "-c", "import triton._C.libtriton"],
        capture_output=True,
    )
    assert result.returncode == 0, (f"importing libtriton crashed at teardown (returncode={result.returncode})\n"
                                    f"stderr:\n{result.stderr.decode(errors='replace')}")


def test_extend_with_module_structure():
    # `extendTritonWith` (ir.cc) reaches these objects with `py::getattr`; if the
    # module structure changes, the walk -- and this assertion -- must be updated
    # together.
    from triton._C import libtriton

    assert hasattr(libtriton, "extend_with")
    assert hasattr(libtriton, "ir"), "extend_with walks to `ir`"
    assert hasattr(libtriton.ir, "builder"), "extend_with walks to `ir.builder`"
    assert hasattr(libtriton, "passes"), "extend_with walks to `passes`"
