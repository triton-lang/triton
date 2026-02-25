import ast
import importlib
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Callable

import pytest

from triton.tools.triton_to_gluon_translator.ordered_set import ordered_set
from triton.tools.triton_to_gluon_translator.slice_kernel import get_reference, slice_kernel
from triton.tools.triton_to_gluon_translator.stable_toposort import stable_toposort


@pytest.fixture(autouse=True)
def clean_import_state(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    original = sys.modules.copy()
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original)


def _make_package(tmp_path: Path, files: dict[str, str]) -> tuple[str, Callable[[str], str]]:
    pkg = f"slice_kernel_test_{uuid.uuid4().hex}"
    pkg_dir = tmp_path / pkg
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    for name, source in files.items():
        (pkg_dir / name).write_text(textwrap.dedent(source).strip() + "\n")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    return pkg, lambda mod: f"{pkg}.{mod}"


def _normalize(source: str) -> list[str]:
    return [line.strip() for line in source.strip().splitlines() if line.strip()]


def test_stable_toposort_preserves_component_order():
    graph = {
        0: ordered_set([1]),
        1: ordered_set([2]),
        2: ordered_set([0, 3]),
        3: ordered_set([4]),
        4: ordered_set(),
    }
    assert stable_toposort(graph) == [0, 1, 2, 3, 4]


def test_slice_kernel_basic_module_slicing(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "lib_foo.py": """
                import math

                def some_util() -> int:
                    return 42

                def common_util() -> int:
                    return math.prod([11, 33])
            """,
            "lib_bar.py": """
                def prod(values) -> None:
                    return

                def some_util() -> int:
                    return 55
            """,
            "kernel_mod.py": """
                from .lib_bar import prod as util_bar
                from .lib_bar import some_util
                from .lib_foo import common_util as util_foo
                from .lib_foo import some_util as util

                def prod() -> None:
                    return

                def kernel() -> None:
                    prod()
                    util_foo()
                    util_bar([42, 22])
                    some_util()
                    util()
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"])
    assert "import math" in output
    assert "def lib_foo_some_util() -> int:" in output
    assert "def some_util() -> int:" in output
    assert "def lib_bar_prod(values) -> None:" in output
    assert "def common_util() -> int:" in output
    assert "lib_bar_prod([42, 22])" in output
    assert "lib_foo_some_util()" in output


def test_slice_kernel_supports_injected_decorator_matchers(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py": """
                def mock_kernel(fn=None, *, idle_sms=None):
                    def deco(inner):
                        return inner

                    if fn is None:
                        return deco
                    return deco(fn)

                def keep():
                    def deco(inner):
                        return inner

                    return deco

                def foo() -> None:
                    pass

                @mock_kernel
                @keep()
                def kernel_top() -> None:
                    foo()

                def nested_dep() -> int:
                    return 0

                def get_idle_sms() -> int:
                    return nested_dep()

                @keep()
                @mock_kernel(idle_sms=get_idle_sms())
                def kernel_bottom() -> None:
                    foo()
            """,
        },
    )
    module = importlib.import_module(mod("kernel_mod"))

    def matcher(context, cur_module, decorator):
        func = decorator.func if isinstance(decorator, ast.Call) else decorator
        ref = get_reference(context, cur_module, func)
        return ref is not None and ref[0] is module.mock_kernel

    top = slice_kernel(
        [f"{mod('kernel_mod')}:kernel_top"],
        ["triton", "torch"],
        ignored_decorator_matchers=[matcher],
    )
    assert "@keep()" in top
    assert "@mock_kernel" not in top

    bottom = slice_kernel(
        [f"{mod('kernel_mod')}:kernel_bottom"],
        ["triton", "torch"],
        ignored_decorator_matchers=[matcher],
    )
    assert "@keep()" not in bottom
    assert "@mock_kernel" not in bottom
    assert "def get_idle_sms" not in bottom
    assert "def nested_dep" not in bottom


def test_slice_kernel_translate_to_gluon_keeps_tensor_method_rewrites(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py": """
                import triton
                import triton.language as tl

                @triton.jit(repr=lambda _: "custom_kernel_name")
                def kernel(x):
                    tl.squeeze(x, 0)
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], translate_to_gluon=True)
    assert "import triton.experimental.gluon as gluon" in output
    assert "@gluon.jit(repr=lambda _: 'custom_kernel_name')" in output
    assert "reset_to_default_layout" in output
    assert "squeeze(x, 0)" in output


def test_slice_kernel_translate_to_gluon_inlines_descriptor_adapter(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py": """
                from triton.tools.ragged_tma import create_ragged_descriptor

                def kernel(t):
                    return create_ragged_descriptor(t, [16, 16])
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], translate_to_gluon=True)
    assert "convert_host_descriptor(" in output
    assert "def convert_host_descriptor" in output


def test_slice_kernel_public_imports():
    from triton.tools.triton_to_gluon_translator.slice_kernel import slice_kernel as new_slice_kernel
    from triton.tools.triton_to_gluon_translator.translator import translate_paths
    from triton.tools.triton_to_gluon_translator.translator import convert_triton_to_gluon
    from triton.tools.triton_to_gluon_translator.translator_helpers import convert_host_descriptor

    assert callable(new_slice_kernel)
    assert callable(translate_paths)
    assert callable(convert_triton_to_gluon)
    assert callable(convert_host_descriptor)
