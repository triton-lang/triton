import ast
import importlib
import sys
import sysconfig
import textwrap
import uuid
from pathlib import Path
from typing import Callable

import pytest

from triton.tools.triton_to_gluon_translator.slice_kernel import RewriteSpec, get_reference, slice_kernel
from triton.tools.triton_to_gluon_translator.target import TranslatorTarget


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
        (pkg_dir / name).write_text(textwrap.dedent(source).strip().replace("{pkg}", pkg) + "\n")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    return pkg, lambda mod: f"{pkg}.{mod}"


def assert_code_equal(actual: str, expected: str) -> None:
    lhs = ast.dump(ast.parse(actual))
    rhs = ast.dump(ast.parse(expected))
    assert lhs == rhs


@pytest.mark.parametrize("decorator", ["lru_cache", "lru_cache()", "lru_cache(maxsize=2, typed=True)", "cache"])
@pytest.mark.parametrize("entrypoint", ["cached", "kernel"])
def test_slice_kernel_cached_function(tmp_path, decorator, entrypoint):
    _, mod = _make_package(
        tmp_path, {
            "helpers.py":
            """
            def payload(value):
                return [value + 1]
        """,
            "kernel_mod.py":
            f"""
            from functools import cache, lru_cache
            from .helpers import payload

            @{decorator}
            def cached(value):
                return payload(value)

            def kernel(value):
                return cached(value)
        """,
        })
    source = slice_kernel([f"{mod('kernel_mod')}:{entrypoint}"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    namespace = {}
    exec(source, namespace)
    function = namespace[entrypoint]
    first = function(4)
    assert first == [5]
    assert function(4) is first
    assert function(5) == [6]
    assert namespace["cached"].cache_info().hits == 1
    assert namespace["cached"].cache_info().misses == 2


def test_slice_kernel_basic_module_slicing(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "lib_foo.py":
            """
                import math

                def some_util() -> int:
                    return 42

                def common_util() -> int:
                    return math.prod([11, 33])
            """,
            "lib_bar.py":
            """
                def prod(values) -> None:
                    return

                def some_util() -> int:
                    return 55
            """,
            "kernel_mod.py":
            """
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

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
import math

def lib_foo_some_util() -> int:
    return 42


def some_util() -> int:
    return 55


def lib_bar_prod(values) -> None:
    return


def common_util() -> int:
    return math.prod([11, 33])


def prod() -> None:
    return


def kernel() -> None:
    prod()
    common_util()
    lib_bar_prod([42, 22])
    some_util()
    lib_foo_some_util()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_does_not_treat_site_packages_as_stdlib(tmp_path, monkeypatch):
    fake_stdlib = tmp_path / "venv" / "lib" / "python3.12"
    fake_site_packages = fake_stdlib / "site-packages"
    fake_site_packages.mkdir(parents=True)
    pkg, mod = _make_package(
        fake_site_packages,
        {
            "helpers.py":
            """
                def helper() -> int:
                    return 7
            """,
            "kernel_mod.py":
            """
                from .helpers import helper

                def kernel() -> int:
                    return helper()
            """,
        },
    )

    original_get_paths = sysconfig.get_paths

    def fake_get_paths():
        paths = original_get_paths().copy()
        paths["stdlib"] = str(fake_stdlib)
        paths["platstdlib"] = str(fake_stdlib)
        paths["purelib"] = str(fake_site_packages)
        paths["platlib"] = str(fake_site_packages)
        return paths

    monkeypatch.setattr(sysconfig, "get_paths", fake_get_paths)

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
def helper() -> int:
    return 7


def kernel() -> int:
    return helper()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_does_not_treat_base_python_site_packages_as_stdlib_in_venv(tmp_path, monkeypatch):
    fake_base_stdlib = tmp_path / "base" / "lib" / "python3.12"
    fake_base_site_packages = fake_base_stdlib / "site-packages"
    fake_venv_site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    fake_base_site_packages.mkdir(parents=True)
    fake_venv_site_packages.mkdir(parents=True)
    pkg, mod = _make_package(
        fake_base_site_packages,
        {
            "helpers.py":
            """
                def helper() -> int:
                    return 7
            """,
            "kernel_mod.py":
            """
                from .helpers import helper

                def kernel() -> int:
                    return helper()
            """,
        },
    )

    original_get_paths = sysconfig.get_paths

    def fake_get_paths():
        paths = original_get_paths().copy()
        paths["stdlib"] = str(fake_base_stdlib)
        paths["platstdlib"] = str(fake_base_stdlib)
        paths["purelib"] = str(fake_venv_site_packages)
        paths["platlib"] = str(fake_venv_site_packages)
        return paths

    monkeypatch.setattr(sysconfig, "get_paths", fake_get_paths)

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
def helper() -> int:
    return 7


def kernel() -> int:
    return helper()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_preserves_gluon_builtin_sugar(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                import triton.experimental.gluon.language as gl

                def kernel() -> None:
                    gl.static_assert(True)
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton"], target=TranslatorTarget.GENERIC)
    expected = R"""
import triton.experimental.gluon.language as gl

def kernel() -> None:
    gl.static_assert(True)
    """
    assert_code_equal(output, expected)


def test_slice_kernel_supports_injected_decorator_matchers(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
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
        rewrite_spec=RewriteSpec(ignored_decorator_matchers=[matcher]),
        target=TranslatorTarget.GENERIC,
    )
    expected_top = R"""
def keep():

    def deco(inner):
        return inner
    return deco


def foo() -> None:
    pass


@keep()
def kernel_top() -> None:
    foo()
    """
    assert_code_equal(top, expected_top)

    bottom = slice_kernel(
        [f"{mod('kernel_mod')}:kernel_bottom"],
        ["triton", "torch"],
        rewrite_spec=RewriteSpec(ignored_decorator_matchers=[matcher]),
        target=TranslatorTarget.GENERIC,
    )
    expected_bottom = R"""
def keep():

    def deco(inner):
        return inner
    return deco


def foo() -> None:
    pass


def kernel_bottom() -> None:
    foo()
    """
    assert_code_equal(bottom, expected_bottom)


def test_slice_kernel_translate_to_gluon_keeps_tensor_method_rewrites(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                import triton
                import triton.language as tl

                @triton.jit(repr=lambda _: "custom_kernel_name")
                def kernel(x):
                    tl.squeeze(x, 0)
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], translate_to_gluon=True,
                          target=TranslatorTarget.GENERIC)
    expected = R"""
import triton.experimental.gluon.language as gl
import triton.tools.triton_to_gluon_translator.common_helpers
import triton.experimental.gluon as gluon

@gluon.jit
def squeeze(x, dim: gl.constexpr):
    '''
    Removes a length-1 dimension from :code:`x` at index :code:`dim`.

    The selected dimension must have length 1 (checked at compile time). This is
    the inverse of :func:`unsqueeze`.

    :param x: the input tensor
    :param dim: the index of the dimension to remove
    :type dim: int
    '''
    gl.static_assert(-len(x.shape) <= dim and dim < len(x.shape))
    _dim: gl.constexpr = dim + len(x.shape) if dim < 0 else dim
    gl.static_assert(x.shape[_dim] == 1)
    return triton.tools.triton_to_gluon_translator.common_helpers.reset_to_default_layout(x.reshape(x.shape[:_dim] + x.shape[_dim + 1:]))


@gluon.jit(repr=lambda _: 'custom_kernel_name')
def kernel(x):
    squeeze(x, 0)
    """
    assert_code_equal(output, expected)


def test_slice_kernel_translate_to_gluon_inlines_descriptor_adapter(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                from triton.tools.ragged_tma import create_ragged_descriptor

                def kernel(t):
                    return create_ragged_descriptor(t, [16, 16])
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], translate_to_gluon=True,
                          target=TranslatorTarget.GENERIC)
    expected = R"""
import triton.tools.ragged_tma

def kernel(t):
    return convert_host_descriptor(triton.tools.ragged_tma.create_ragged_descriptor(t, [16, 16]))


def _torch_dtype_to_triton(dtype):
    import torch

    if dtype == torch.float8_e5m2:
        return gl.float8e5
    if dtype == torch.float8_e4m3fn:
        return gl.float8e4nv
    return getattr(gl, str(dtype).split(".")[1])

def convert_host_descriptor(desc):
    from triton.tools.tensor_descriptor import TensorDescriptor

    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    dtype = desc.base.dtype
    tensor = desc.base
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, _torch_dtype_to_triton(dtype))
    return gluon.nvidia.hopper.TensorDescriptor(
        tensor, desc.shape, desc.strides, block_shape, layout
    )
    """
    assert_code_equal(output, expected)


def test_slice_kernel_binds_local_imports(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "helpers.py":
            """
                def local_helper() -> int:
                    return 3
            """,
            "kernel_mod.py":
            """
                def kernel() -> int:
                    from .helpers import local_helper
                    return local_helper()
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
def local_helper() -> int:
    return 3


def kernel() -> int:
    return local_helper()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_function_import(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                def helper() -> None:
                    import math

                    math.prod([11, 33])

                def kernel() -> None:
                    helper()
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
import math

def helper() -> None:
    math.prod([11, 33])


def kernel() -> None:
    helper()
    """
    assert_code_equal(output, expected)


@pytest.mark.xfail(
    strict=True,
    reason="TODO: handle local import aliases that are used as module values",
)
def test_slice_kernel_function_import_module_value(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                def helper():
                    import math

                    return math

                def kernel():
                    return helper()
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
import math

def helper():
    return math


def kernel():
    return helper()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_function_relative_import(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "lib_foo.py":
            """
                import math

                def common_util() -> int:
                    return math.prod([11, 33])
            """,
            "kernel_mod.py":
            """
                def helper() -> None:
                    from .lib_foo import common_util as util_foo

                    util_foo()

                def kernel() -> None:
                    helper()
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
import math

def common_util() -> int:
    return math.prod([11, 33])


def helper() -> None:
    common_util()


def kernel() -> None:
    helper()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_module_level_imported_value_alias(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "lib_foo.py":
            """
                VALUE = {"key": 7}
            """,
            "kernel_mod.py":
            """
                from .lib_foo import VALUE as ALIASED_VALUE

                def kernel() -> int:
                    return ALIASED_VALUE["key"]
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R'''
VALUE = {"key": 7}


def kernel() -> int:
    return VALUE["key"]
    '''
    assert_code_equal(output, expected)


@pytest.mark.xfail(
    strict=True,
    reason="TODO: preserve origin metadata for local from-import values",
)
def test_slice_kernel_function_from_import_value(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                def helper():
                    from math import pi

                    return pi

                def kernel():
                    return helper()
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
import math

def helper():
    return math.pi


def kernel():
    return helper()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_function_absolute_import(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "lib_foo.py":
            """
                import math

                def common_util() -> int:
                    return math.prod([11, 33])
            """,
            "kernel_mod.py":
            """
                import {pkg}.lib_foo

                _PRELOADED_LIB_FOO = {pkg}.lib_foo

                def helper() -> None:
                    from {pkg}.lib_foo import common_util as util_foo

                    util_foo()

                def kernel() -> None:
                    helper()
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
import math

def common_util() -> int:
    return math.prod([11, 33])


def helper() -> None:
    common_util()


def kernel() -> None:
    helper()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_function_module_relative_import(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "lib_foo.py":
            """
                import math

                def common_util() -> int:
                    return math.prod([11, 33])
            """,
            "kernel_mod.py":
            """
                def helper() -> None:
                    from . import lib_foo

                    lib_foo.common_util()

                def kernel() -> None:
                    helper()
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
import math

def common_util() -> int:
    return math.prod([11, 33])


def helper() -> None:
    common_util()


def kernel() -> None:
    helper()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_function_module_relative_import_leaf(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "lib_foo.py":
            """
                import math

                def common_util() -> int:
                    return math.prod([11, 33])
            """,
            "kernel_mod.py":
            """
                def helper() -> None:
                    from . import lib_foo

                    lib_foo.common_util()

                def kernel() -> None:
                    helper()
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch", mod("lib_foo")],
                          target=TranslatorTarget.GENERIC)
    expected = f"""
import {pkg}.lib_foo

def helper() -> None:
    {pkg}.lib_foo.common_util()


def kernel() -> None:
    helper()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_treats_assign_targets_as_locals(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                def helper() -> int:
                    return 1

                def kernel() -> int:
                    helper = lambda: 2
                    return helper()
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
def kernel() -> int:
    helper = lambda: 2
    return helper()
    """
    assert_code_equal(output, expected)


def test_slice_kernel_treats_annassign_targets_as_locals(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                value = 7

                def kernel() -> int:
                    value: int = 3
                    return value
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
def kernel() -> int:
    value: int = 3
    return value
    """
    assert_code_equal(output, expected)


def test_slice_kernel_treats_assign_and_annassign_targets_as_locals(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                import triton

                def global_assign() -> int:
                    return 1

                def global_annassign() -> int:
                    return 2

                @triton.jit
                def kernel() -> tuple[int, int]:
                    global_assign = 3
                    copied = global_assign
                    global_annassign: int = copied + 1
                    return copied, global_annassign
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    expected = R"""
import triton

@triton.jit
def kernel() -> tuple[int, int]:
    global_assign = 3
    copied = global_assign
    global_annassign: int = copied + 1
    return (copied, global_annassign)
    """
    assert_code_equal(output, expected)


def test_slice_kernel_translate_to_gluon_avoids_double_descriptor_wrap(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                from triton.tools.tensor_descriptor import TensorDescriptor

                def convert_host_descriptor(desc):
                    return desc

                def kernel(t):
                    return convert_host_descriptor(TensorDescriptor.from_tensor(t, [16, 16]))
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], translate_to_gluon=True,
                          target=TranslatorTarget.GENERIC)
    expected = R"""
from triton.tools.tensor_descriptor import TensorDescriptor

def convert_host_descriptor(desc):
    return desc


def kernel(t):
    return convert_host_descriptor(TensorDescriptor.from_tensor(t, [16, 16]))
    """
    assert_code_equal(output, expected)


def test_translate_to_gluon_explicit_expand_dims_rewrites_layout(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                import triton
                import triton.language as tl

                @triton.jit
                def kernel(out_ptr, BLOCK: tl.constexpr):
                    offsets = tl.arange(0, BLOCK)
                    expanded = tl.expand_dims(offsets, 0)
                    tl.store(out_ptr + expanded, expanded)
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton"], translate_to_gluon=True,
                          target=TranslatorTarget.GENERIC)
    expected = R"""
import triton.experimental.gluon.language as gl
import triton.tools.triton_to_gluon_translator.common_helpers
import triton.experimental.gluon as gluon

@gluon.jit
def kernel(out_ptr, BLOCK: gl.constexpr):
    offsets = triton.tools.triton_to_gluon_translator.common_helpers.tl_arange(0, BLOCK)
    expanded = gl.expand_dims(triton.tools.triton_to_gluon_translator.common_helpers.convert_to_expand_dims_layout(offsets, [0]), 0)
    gl.store(out_ptr + expanded, expanded)
    """
    assert_code_equal(output, expected)


def test_translate_to_gluon_member_fn_expand_dims_rewrites_layout(tmp_path):
    pkg, mod = _make_package(
        tmp_path,
        {
            "kernel_mod.py":
            """
                import triton
                import triton.language as tl

                @triton.jit
                def kernel(out_ptr, BLOCK: tl.constexpr):
                    offsets = tl.arange(0, BLOCK)
                    expanded = offsets.expand_dims(0)
                    tl.store(out_ptr + expanded, expanded)
            """,
        },
    )

    output = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton"], translate_to_gluon=True,
                          target=TranslatorTarget.GENERIC)
    expected = R"""
import triton.experimental.gluon.language as gl
import triton.tools.triton_to_gluon_translator.common_helpers
import triton.experimental.gluon as gluon

@gluon.jit
def kernel(out_ptr, BLOCK: gl.constexpr):
    offsets = triton.tools.triton_to_gluon_translator.common_helpers.tl_arange(0, BLOCK)
    expanded = triton.tools.triton_to_gluon_translator.common_helpers.convert_to_expand_dims_layout(offsets, [0]).expand_dims(0)
    gl.store(out_ptr + expanded, expanded)
    """
    assert_code_equal(output, expected)


def test_slice_kernel_public_imports():
    from triton.tools.triton_to_gluon_translator.slice_kernel import slice_kernel as new_slice_kernel
    from triton.tools.triton_to_gluon_translator.translator import translate_paths
    from triton.tools.triton_to_gluon_translator.translator import convert_triton_to_gluon
    from triton.tools.triton_to_gluon_translator.nvidia_helpers import convert_host_descriptor

    assert callable(new_slice_kernel)
    assert callable(translate_paths)
    assert callable(convert_triton_to_gluon)
    assert callable(convert_host_descriptor)


@pytest.mark.parametrize("value", ["32", "None", "True", "'shared'"])
@pytest.mark.parametrize("imported", [False, True])
def test_slice_kernel_global_aliases(tmp_path, value, imported):
    _, mod = _make_package(
        tmp_path, {
            "constants.py":
            f"BASE = {value}",
            "kernel_mod.py":
            f"""
            {"from .constants import BASE as SOURCE" if imported else f"SOURCE = {value}"}
            ALIAS: object = SOURCE
            FINAL = ALIAS

            def kernel(value=FINAL):
                return value, SOURCE, ALIAS, FINAL
        """,
        })
    original = importlib.import_module(mod("kernel_mod"))
    source = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    namespace = {}
    exec(source, namespace)
    assert namespace["kernel"]() == original.kernel()
    assert namespace["kernel"]("override") == original.kernel("override")


def test_slice_kernel_global_alias_leaf(tmp_path):
    _, mod = _make_package(
        tmp_path, {
            "constants.py":
            "VALUE = 32",
            "kernel_mod.py":
            """
            from .constants import VALUE
            COPIED = VALUE

            def kernel():
                return VALUE, COPIED
        """,
        })
    source = slice_kernel([f"{mod('kernel_mod')}:kernel"], ["triton", "torch"],
                          leaf_paths=[f"{mod('constants')}:VALUE"], target=TranslatorTarget.GENERIC)
    namespace = {}
    exec(source, namespace)
    assert namespace["kernel"]() == (32, 32)
    importlib.import_module(mod("constants")).VALUE = 64
    assert namespace["kernel"]() == (64, 32)


@pytest.mark.parametrize("construction", ["assignment", "factory", "reassignment"])
@pytest.mark.parametrize("entrypoint", ["cached", "kernel"])
def test_slice_kernel_assigned_cache(tmp_path, construction, entrypoint):
    definitions = {
        "assignment":
        """
            def raw(value):
                return [value + 1]
            cached = functools.cache(raw)
        """,
        "factory":
        """
            def make_cached(offset):
                @functools.cache
                def nested(value):
                    return [value + offset]
                return nested
            cached = make_cached(1)
        """,
        "reassignment":
        """
            def cached(value):
                return [value + 1]
            cached = functools.cache(cached)
        """,
    }
    source = "import functools\n" + textwrap.dedent(definitions[construction])
    source += "\ndef kernel(value):\n    return cached(value)\n"
    _, mod = _make_package(tmp_path, {"kernel_mod.py": source})
    if construction == "reassignment":
        with pytest.raises(ValueError, match="cannot extract cache wrapper rebinding"):
            slice_kernel([f"{mod('kernel_mod')}:{entrypoint}"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
        return
    sliced = slice_kernel([f"{mod('kernel_mod')}:{entrypoint}"], ["triton", "torch"], target=TranslatorTarget.GENERIC)
    namespace = {}
    exec(sliced, namespace)
    result = namespace[entrypoint](4)
    assert result == [5]
    assert namespace[entrypoint](4) is result
    assert namespace["cached"].cache_info().hits == 1
