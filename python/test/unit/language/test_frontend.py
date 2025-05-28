import sys
import os
import io
import inspect
from typing import List, Tuple

from filecheck.options import Options
from filecheck.finput import FInput
from filecheck.parser import Parser, pattern_for_opts
from filecheck.matcher import Matcher

import triton
import triton.language as tl
from triton.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir
from triton.language.core import base_type, base_value

import pytest

# ===-----------------------------------------------------------------------===#
# filecheck_test
# ===-----------------------------------------------------------------------===#

# Stub target for testing the frontend.
stub_target = GPUTarget("cuda", 100, 32)
stub_backend = make_backend(stub_target)

llvm_bin_dir = os.path.join(os.path.dirname(sys.executable), "bin")
filecheck_path = os.path.join(llvm_bin_dir, "FileCheck")


def run_filecheck(name, module_str, check_template):
    options = Options(match_filename=name)
    fin = FInput(name, module_str)
    ops = io.StringIO(check_template)
    parser = Parser(options, ops, *pattern_for_opts(options))
    matcher = Matcher(options, fin, parser)
    matcher.stderr = io.StringIO()
    if matcher.run() != 0:
        raise ValueError(matcher.stderr.getvalue())


def run_parser(kernel_fn):
    sigkeys = [x.name for x in kernel_fn.params]
    sigvals = [f"arg{i}" for i in range(len(sigkeys))]
    signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
    src = ASTSource(fn=kernel_fn, signature=signature)

    context = ir.context()
    ir.load_dialects(context)
    stub_backend.load_dialects(context)

    extra_options = src.parse_options()
    options = stub_backend.parse_options(dict(**extra_options))
    codegen_fns = stub_backend.get_codegen_implementation(options)
    module_map = stub_backend.get_module_map()
    return src.make_ir(options, codegen_fns, module_map, context)


def run_filecheck_test(kernel_fn):
    assert isinstance(kernel_fn, triton.runtime.JITFunction)
    check_template = inspect.getsource(kernel_fn.fn)
    if check_template is None:
        raise ValueError("kernel function must have a docstring with FileCheck template")
    mlir_module = run_parser(kernel_fn)

    run_filecheck("placeholder", str(mlir_module), check_template)


@triton.jit
def anchor(v):
    pass


# Smoke test to make sure filecheck is working correctly.
def test_filecheck_positive():

    @triton.jit
    def test_kernel():
        # CHECK-LABEL: test_kernel
        scalar = 42
        # CHECK: %c42_i32 = arith.constant 42 : i32
        # CHECK-NEXT: call @anchor{{.*}}(%c42_i32) : (i32) -> ()
        anchor(scalar)

    run_filecheck_test(test_kernel)


def test_filecheck_negative():

    @triton.jit
    def test_kernel():
        # CHECK-LABEL: test_kernel
        scalar = 11
        # CHECK: %c42_i32
        anchor(scalar)

    with pytest.raises(ValueError, match="Couldn't match \"%c42_i32\""):
        run_filecheck_test(test_kernel)


def filecheck_test(fn):

    def test_fn():
        run_filecheck_test(fn)

    return test_fn


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


class pair_type(base_type):

    def __init__(self, first_type, second_type):
        self.first_type = first_type
        self.second_type = second_type

    def __eq__(self, other) -> bool:
        return self.first_type == other.first_type and self.second_type == other.second_type

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[ir.value, int]:
        first, cursor = self.first_type._unflatten_ir(handles, cursor)
        second, cursor = self.second_type._unflatten_ir(handles, cursor)
        return pair_value(first, second), cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        self.first_type._flatten_ir_types(builder, out)
        self.second_type._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        return f"pair<{self.first_type.mangle()}, {self.second_type.mangle()}>"


class pair_value(base_value):

    def __init__(self, first, second):
        self.first = first
        self.second = second
        self.type = pair_type(first.type, second.type)

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        self.first._flatten_ir(handles)
        self.second._flatten_ir(handles)

    @triton.jit
    def get_first(self):
        return self.first

    def get_second(self, _builder=None):
        return self.second

    @triton.jit
    def unpack(self):
        return self.get_first(), self.get_second()


@tl.core.builtin
def pair_value_ctor(first, second, _builder=None):
    return pair_value(first, second)


@filecheck_test
@triton.jit
def test_assign_attribute():
    # CHECK-LABEL: assign_attribute
    # CHECK: %c11_i32 = arith.constant 11 : i32
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    scalar = 11
    pair = pair_value_ctor(tl.arange(0, 4), scalar)
    # CHECK: %c42_i32 = arith.constant 42 : i32
    # CHECK-NEXT: call @"anchor{{.*}}"([[RANGE]], %c42_i32)
    pair.second = 42
    anchor(pair)


@filecheck_test
@triton.jit
def test_jit_method():
    # CHECK-LABEL: test_jit_method
    # CHECK: %c11_i32 = arith.constant 11 : i32
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    scalar = 11
    # CHECK: [[V:%.*]]:2 = tt.call @"unpack{{.*}}"([[RANGE]], %c11_i32)
    pair = pair_value_ctor(tl.arange(0, 4), scalar)
    a, b = pair.unpack()
    # CHECK: call @anchor{{.*}}([[V]]#0)
    anchor(a)
    # CHECK: call @anchor{{.*}}([[V]]#1)
    anchor(b)
