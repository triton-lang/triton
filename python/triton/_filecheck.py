import sys
import os
import io
import inspect

from filecheck.options import Options
from filecheck.finput import FInput
from filecheck.parser import Parser, pattern_for_opts
from filecheck.matcher import Matcher

import triton
from triton.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
from triton._C.libtriton import ir

# ===-----------------------------------------------------------------------===#
# filecheck_test
# ===-----------------------------------------------------------------------===#

# Stub target for testing the frontend.
stub_target = GPUTarget("cuda", 100, 32)
stub_backend = make_backend(stub_target)

llvm_bin_dir = os.path.join(os.path.dirname(sys.executable), "bin")
filecheck_path = os.path.join(llvm_bin_dir, "FileCheck")


class MatchError(ValueError):

    def __init__(self, message, module_str):
        super().__init__(message)
        self.module_str = module_str

    def __str__(self):
        return f"{super().__str__()}\n{self.module_str}"


def run_filecheck(name, module_str, check_template):
    options = Options(match_filename=name)
    fin = FInput(name, module_str)
    ops = io.StringIO(check_template)
    parser = Parser(options, ops, *pattern_for_opts(options))
    matcher = Matcher(options, fin, parser)
    matcher.stderr = io.StringIO()
    if matcher.run() != 0:
        raise MatchError(matcher.stderr.getvalue(), module_str)


def run_parser(kernel_fn):
    sigkeys = [x.name for x in kernel_fn.params]
    sigvals = [f"arg{i}" for i in range(len(sigkeys))]
    signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
    source_cls = GluonASTSource if kernel_fn.is_gluon() else ASTSource
    src = source_cls(fn=kernel_fn, signature=signature)

    context = ir.context()
    ir.load_dialects(context)
    stub_backend.load_dialects(context)

    extra_options = src.parse_options()
    options = stub_backend.parse_options(dict(**extra_options))
    codegen_fns = stub_backend.get_codegen_implementation(options)
    module_map = stub_backend.get_module_map()
    module = src.make_ir(options, codegen_fns, module_map, context)
    assert module.verify()
    return module


def run_filecheck_test(kernel_fn):
    assert isinstance(kernel_fn, triton.runtime.JITFunction)
    check_template = inspect.getsource(kernel_fn.fn)
    if check_template is None:
        raise ValueError("kernel function must have a docstring with FileCheck template")
    mlir_module = run_parser(kernel_fn)

    run_filecheck("placeholder", mlir_module.str_nodebug(), check_template)


def filecheck_test(fn):

    def test_fn():
        run_filecheck_test(fn)

    return test_fn
