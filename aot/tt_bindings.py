from dataclasses import dataclass
from typing import Any, Mapping
import sys
import importlib.util

from triton.code_gen import JITFunction

from aot_kernel import AOTKernel

from _types import ModuleScope


def execute_module(fpath):
    # TODO: allow module key specification
    MODULE_KEY = "jitted_funcs"
    spec = importlib.util.spec_from_file_location(MODULE_KEY, fpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[MODULE_KEY] = module
    return module


def filter_jit_funcs(scope: ModuleScope) -> Mapping[str, JITFunction]:
    return {k: v for k, v in scope.items() if isinstance(v, JITFunction)}


def get_scope(module_) -> ModuleScope:
    script_scope = dict(
        (att, getattr(module_, att))
        for att in (v for v in vars(module_) if not v.startswith("__"))
    )
    return script_scope


@dataclass
class TritonContext:
    module_scope: ModuleScope

    @classmethod
    def build_from_script(cls, script_path: str):
        mod_ = execute_module(script_path)
        ctx = cls(get_scope(mod_))
        return ctx

    def init_kernels(self) -> Mapping[str, AOTKernel]:
        jit_funcs = filter_jit_funcs(self.module_scope)
        return {fname: AOTKernel(jit_fn) for fname, jit_fn in jit_funcs.items()}
