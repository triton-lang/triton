from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

from triton.tools.aot.parsers import KernelLinkerMeta

from .templates import (
    DEFAULT_ALGO_DECL_TEMPLATE,
    DEFAULT_GLOBAL_DECL_TEMPLATE,
    DEFAULT_HEADER_INCLUDES,
    DEFAULT_SOURCE_INCLUDES,
)


class SignatureGenerator:
    """Generates function signatures for kernels with meta-parameter and constant values

    Ported from `python/triton/tools/link.py`
    """

    @staticmethod
    def gen_signature_with_full_args(m):
        return ", ".join([f"{ty} {arg}" for ty, arg in zip(m.arg_ctypes, m.arg_names)])

    @staticmethod
    def gen_signature(m):
        arg_types = [ty for ty, hint in zip(m.arg_ctypes, m.sizes) if hint != 1]
        arg_names = [arg for arg, hint in zip(m.arg_names, m.sizes) if hint != 1]
        sig = ", ".join([f"{ty} {arg}" for ty, arg in zip(arg_types, arg_names)])
        return sig


class HeaderGenerator(ABC):
    """Interface for generating header for dispatcher code for from compiled triton kernels"""

    SIGNATURE_GENERATOR = SignatureGenerator
    DEFAULT_ALGO_DECL_TEMPLATE: str
    DEFAULT_GLOBAL_DECL_TEMPLATE: str
    HEADER_INCLUDES: List[str]

    def __init__(self, kernels: Dict[str, KernelLinkerMeta]) -> None:
        self.kernels = kernels
        meta_lists = [meta for _, meta in self.kernels.items()]
        self.meta = meta_lists[0][0]

    @abstractmethod
    def generate(self):
        ...


class C_CUDA_HeaderGenerator(HeaderGenerator):
    """Concrete implementation for C CUDA triton kernels"""

    ALGO_DECL_TEMPLATE = DEFAULT_ALGO_DECL_TEMPLATE
    GLOBAL_DECL_TEMPLATE = DEFAULT_GLOBAL_DECL_TEMPLATE
    HEADER_INCLUDES = DEFAULT_HEADER_INCLUDES

    def _make_algo_decl(self, name: str, metas: List[KernelLinkerMeta]):
        """Declarations for kernels"""

        args = self.SIGNATURE_GENERATOR.gen_signature_with_full_args(metas[-1])

        return self.ALGO_DECL_TEMPLATE.format(
            name=name,
            args=args,
        )

    def _make_algo_decls(self) -> str:
        """Generate declarations of kernels with meta-parameter and constant values"""

        algo_decls = []

        for name, meta in self.kernels.items():
            algo_decls.append(self._make_algo_decl(name, meta))

        return "\n".join(algo_decls).lstrip()

    def _make_get_num_algos_decl(self, meta: Optional[KernelLinkerMeta] = None) -> str:
        meta = meta or self.meta
        src = f"int {meta.orig_kernel_name}_get_num_algos(void);"
        return src

    def _make_global_decl(self, meta: Optional[KernelLinkerMeta] = None) -> str:
        """Generate declarations of global / default kernel launch"""
        meta = meta or self.meta
        return self.GLOBAL_DECL_TEMPLATE.format(
            orig_kernel_name=meta.orig_kernel_name,
            default_args=self.SIGNATURE_GENERATOR.gen_signature_with_full_args(meta),
            full_args=self.SIGNATURE_GENERATOR.gen_signature_with_full_args(meta),
        )

    def generate(self):
        includes = "\n".join(self.HEADER_INCLUDES)

        algo_decls = self._make_algo_decls()
        get_num_algos_decl = self._make_get_num_algos_decl()
        global_decl = self._make_global_decl()
        src = "\n".join(
            [
                algo_decls,
                get_num_algos_decl,
                global_decl,
            ]
        )
        return "\n\n".join([includes, src])


class SourceGenerator(ABC):
    """Interface for generating dispatcher code for from compiled triton kernels"""

    SIGNATURE_GENERATOR = SignatureGenerator
    SOURCE_INCLUDES: List[str]

    def __init__(
        self,
        kernels: Dict[str, KernelLinkerMeta],
        meta: Optional[KernelLinkerMeta] = None,
    ) -> None:
        self.kernels = kernels
        if meta is None:
            meta_lists = [meta for name, meta in self.kernels.items()]
            meta = meta_lists[0][0]
        self.meta = meta

    @abstractmethod
    def generate(self):
        ...


class C_CUDA_SourceGenerator(SourceGenerator):
    """Concrete implementation for C CUDA triton kernels

    TODO: refactor to use templates
    """

    SOURCE_INCLUDES = DEFAULT_SOURCE_INCLUDES

    def _condition_fn(self, val, hint):
        if hint == 16:
            return f"({val} % {hint} == 0)"
        elif hint == 1:
            return f"({val} == {hint})"
        else:
            return None

    def _make_dispatcher_load_defs(self, name, metas):
        src = ""
        for mode in ["load", "unload"]:
            src += f"\n// {mode} for: {name}\n"
            for meta in sorted(metas, key=lambda m: -m.num_specs):
                src += f"void {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n"
            src += f"void {mode}_{name}() {{"
            src += "\n"
            for meta in sorted(metas, key=lambda m: -m.num_specs):
                src += f"  {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n"
            src += "}\n"
        return src

    def _make_kernel_hints_dispatcher(
        self, name: str, metas: Sequence[KernelLinkerMeta]
    ) -> str:
        # generate dispatcher function for kernels with different integer value hints
        docs_str = f"// launcher for: {name}\n"
        fn_sig = ""
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            fn_sig += f"CUresult {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(CUstream stream, {self.SIGNATURE_GENERATOR.gen_signature(meta)});\n"

        kernel_sig = f"CUresult {name}(CUstream stream, {self.SIGNATURE_GENERATOR.gen_signature_with_full_args(metas[-1])}){{"

        src = "\n".join([docs_str + fn_sig, kernel_sig]) + "\n"
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            conds = " && ".join(
                [
                    self._condition_fn(val, hint)
                    for val, hint in zip(meta.arg_names, meta.sizes)
                    if hint is not None
                ]
            )
            src += (
                f"  if ({conds})\n" if any(meta.sizes) else "if (1)\n"
            )  # Edge case where no special
            #        dispatcher_conds = self._make_dispatcher_conditions(metas)

            arg_names = [
                arg for arg, hint in zip(meta.arg_names, meta.sizes) if hint != 1
            ]
            src += f"    return {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(stream, {', '.join(arg_names)});\n"
        src += "\n"
        src += "  return CUDA_ERROR_INVALID_VALUE;\n"
        src += "}\n"

        load_defs = self._make_dispatcher_load_defs(name, metas)
        src += "\n" + load_defs

        return src

    def _make_defs(self):
        defs = []
        for name, metas in self.kernels.items():
            defs.append(self._make_kernel_hints_dispatcher(name, metas))
        return "\n".join(defs)

    def _make_func_pointers(self) -> str:
        # the table of hint dispatchers
        src = f"typedef CUresult (*kernel_func_t)(CUstream stream, {self.SIGNATURE_GENERATOR.gen_signature_with_full_args(self.meta)});\n"
        src += f"kernel_func_t {self.meta.orig_kernel_name}_kernels[] = {{\n"
        for name in self.kernels.keys():
            src += f"  {name},\n"
        src += "};\n"

        return src

    def _make_kernel_meta_const_dispatcher(
        self,
        meta: KernelLinkerMeta = None,
    ) -> str:
        meta = meta or self.meta
        src = f"CUresult {meta.orig_kernel_name}(CUstream stream, {self.SIGNATURE_GENERATOR.gen_signature_with_full_args(meta)}, int algo_id){{\n"
        src += f"  assert (algo_id < (int)sizeof({meta.orig_kernel_name}_kernels));\n"
        src += f"  return {meta.orig_kernel_name}_kernels[algo_id](stream, {', '.join(meta.arg_names)});\n"
        src += "}\n"
        return src

    def _make_kernel_load_defs(self, meta: KernelLinkerMeta = None) -> str:
        meta = meta or self.meta
        src = ""
        for mode in ["load", "unload"]:
            src += f"void {mode}_{meta.orig_kernel_name}(void){{\n"
            for name in self.kernels.keys():
                src += f"  {mode}_{name}();\n"
            src += "}\n\n"
        return src

    def _make_get_num_algos_def(self, meta: KernelLinkerMeta = None) -> str:
        meta = meta or self.meta
        src = f"int {meta.orig_kernel_name}_get_num_algos(void){{\n"
        src += f"  return (int)sizeof({meta.orig_kernel_name}_kernels);\n"
        src += "}\n"
        return src

    def _make_default_algo_kernel_def(self, meta: KernelLinkerMeta = None) -> str:
        meta = meta or self.meta
        src = f"CUresult {meta.orig_kernel_name}_default(CUstream stream, {self.SIGNATURE_GENERATOR.gen_signature_with_full_args(meta)}){{\n"
        src += f"  return {meta.orig_kernel_name}(stream, {', '.join(meta.arg_names)}, 0);\n"
        src += "}\n"
        return src

    def generate(self):
        includes = "\n".join(self.SOURCE_INCLUDES)

        defs = self._make_defs()
        func_pointers_def = self._make_func_pointers()
        meta_const_def = self._make_kernel_meta_const_dispatcher()
        get_num_algos_def = self._make_get_num_algos_def()
        load_unload_def = self._make_kernel_load_defs()
        default_algo_kernel = self._make_default_algo_kernel_def()
        src = "\n".join(
            [
                defs,
                func_pointers_def,
                get_num_algos_def,
                meta_const_def,
                load_unload_def,
                default_algo_kernel,
            ]
        )
        return "\n\n".join([includes, src])
