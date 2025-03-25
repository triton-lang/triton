# Reuse Across Files
import os
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import (
    IndentedBuffer,
    SizeArg,
    PythonPrinter,
)

from torch._dynamo import config as dynamo_config

# Reuse Within The Same File
from torch._inductor.codegen.triton import (
    signature_of,
    config_of,
)
from torch._inductor.codegen import triton


# ===-------------------- For XPytorch Inductor -----------------------===
# Modifed Base Pytorch(v2.0.1) torch/_inductor/codegen/triton.py::TritonPrinter
class XPUTritonPrinter(triton.TritonPrinter):

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.floor({self.paren(self._print(expr.args[0]))})"


triton.TritonPrinter = XPUTritonPrinter

# ===------------------------------------------------------------------===


# ===-------------------- For XPytorch Inductor -----------------------===
# Modifed Base Pytorch(v2.0.1) torch/_inductor/codegen/triton.py::TritonOverrides
class XPUTritonOverrides(triton.TritonOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    def libdevice_abs(x):
        return f"libdevice.abs({x})"

    @staticmethod
    def libdevice_exp(x):
        return f"libdevice.exp({x})"

    @staticmethod
    def exp2(x):
        return f"libdevice.exp2({x})"

    @staticmethod
    def expm1(x):
        return f"libdevice.expm1({x})"

    @staticmethod
    def libdevice_sqrt(x):
        return f"libdevice.sqrt({x})"

    @staticmethod
    def libdevice_cos(x):
        return f"libdevice.cos({x})"

    @staticmethod
    def libdevice_sin(x):
        return f"libdevice.sin({x})"

    @staticmethod
    def lgamma(x):
        return f"libdevice.lgamma({x})"

    @staticmethod
    def erf(x):
        return f"libdevice.erf({x})"

    @staticmethod
    def cosh(x):
        return f"libdevice.cosh({x})"

    @staticmethod
    def sinh(x):
        return f"libdevice.sinh({x})"

    @staticmethod
    def acos(x):
        return f"libdevice.acos({x})"

    @staticmethod
    def acosh(x):
        return f"libdevice.acosh({x})"

    @staticmethod
    def asin(x):
        return f"libdevice.asin({x})"

    @staticmethod
    def asinh(x):
        return f"libdevice.asinh({x})"

    @staticmethod
    def atan2(x, y):
        return f"libdevice.atan2({x}, {y})"

    @staticmethod
    def atan(x):
        return f"libdevice.atan({x})"

    @staticmethod
    def atanh(x):
        return f"libdevice.atanh({x})"

    @staticmethod
    def copysign(x, y):
        return f"libdevice.copysign({x}, {y})"

    @staticmethod
    def erfc(x):
        return f"libdevice.erfc({x})"

    @staticmethod
    def hypot(x, y):
        return f"libdevice.hypot({x}, {y})"

    @staticmethod
    def log10(x):
        return f"libdevice.log10({x})"

    @staticmethod
    def nextafter(x, y):
        return f"libdevice.nextafter({x}, {y})"

    @staticmethod
    def rsqrt(x):
        return f"libdevice.rsqrt({x})"

    @staticmethod
    def log1p(x):
        return f"libdevice.log1p({x})"

    @staticmethod
    def tan(x):
        return f"libdevice.tan({x})"

    @staticmethod
    def tanh(x):
        return f"libdevice.tanh({x})"

    @staticmethod
    def libdevice_sigmoid(x):
        return f"1/(1 + libdevice.exp(-({x})))"

    @staticmethod
    def signbit(x):
        # XX: This is wrong for the value -0.0 in floating point
        return f"libdevice.signbit({x}) if ({x}).dtype is tl.float32 else {x} < 0"

    @staticmethod
    def fmod(a, b):
        return f"libdevice.fmod({a}, {b})"

    @staticmethod
    def pow(a, b):
        return f"libdevice.pow({a}, {b})"

    @staticmethod
    def libdevice_log(x):
        return f"libdevice.log({x})"

    @staticmethod
    def isinf(x):
        return f"libdevice.isinf({x})"

    @staticmethod
    def isnan(x):
        return f"libdevice.isnan({x})"

    @staticmethod
    def round(x):
        return f"libdevice.nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"libdevice.floor({x})"

    @staticmethod
    def trunc(x):
        return f"libdevice.trunc({x})"

    @staticmethod
    def ceil(x):
        return f"libdevice.ceil({x})"


triton.TritonOverrides = XPUTritonOverrides
# ===------------------------------------------------------------------===

# ===-------------------- For XPytorch Inductor -----------------------===
# Modifed Base Pytorch(v2.0.1) torch/_inductor/codegen/triton.py::TritonKernel
from xpu.backend.driver import get_xpu_spec
import math

pexpr = PythonPrinter().doprint
xpu_hasAtomic = False


class XPUTritonKernel(triton.TritonKernel):
    overrides = triton.TritonOverrides
    sexpr = pexpr

    def store(self, name, index, value, mode=None):
        var = self.args.output(name)
        index, mask_vars, mask = self.indexing(index, dense_indexing=True)
        if mode is None:
            line = f"tl.store({var} + ({index}), {value}, {mask})"
        elif mode == "atomic_add":
            line = f"tl.atomic_add({var} + ({index}), {value}, {mask})"
            global xpu_hasAtomic
            xpu_hasAtomic = True
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(name, line)
        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

    def codegen_kernel(self, name=None):
        from triton import next_power_of_2

        # ===-------------------- For Triton XPU -----------------------===
        from xpu.backend.driver import get_xpu_spec
        arch = int(os.environ.get('TRITON_XPU_ARCH', '3'))
        cluster_num = get_xpu_spec(arch)[0]
        core_num_per_cluster = get_xpu_spec(arch)[1]
        core_num = cluster_num * core_num_per_cluster

        def next_multiply_of_num(n: int, m: int) -> int:
            res = math.ceil(n / m) * m
            return res

        def get_xpu_1d_hint(numel_0: int, cluster_num: int) -> int:
            size_hint_0 = math.ceil(numel_0 / cluster_num)
            size_hint_0 = next_power_of_2(size_hint_0)
            size_hint_0 = size_hint_0 * cluster_num
            return size_hint_0

        def get_xpu_2d_hint(numel_0: int, numel_1: int, cluster_num: int, core_num: int) -> int:
            if numel_0 < core_num:
                size_hint_0 = math.ceil(numel_0 / cluster_num)
                size_hint_0 = next_power_of_2(size_hint_0)
                size_hint_0 = size_hint_0 * cluster_num
            else:
                size_hint_0 = next_multiply_of_num(numel_0, core_num)
            size_hint_1 = next_power_of_2(numel_1)
            return [size_hint_0, size_hint_1]

        code = IndentedBuffer()
        # size_hints = [
        #     next_power_of_2(V.graph.sizevars.size_hint(numel)) for numel in self.numels
        # ]
        # vvv
        numel_0 = V.graph.sizevars.size_hint(self.numels[0])
        if len(self.numels) == 1:
            size_hints = [
                get_xpu_1d_hint(numel_0, cluster_num),
            ]
        elif len(self.numels) == 2:
            numel_1 = V.graph.sizevars.size_hint(self.numels[1])
            if numel_1 != 1:
                size_hints = get_xpu_2d_hint(numel_0, numel_1, cluster_num, core_num)
            else:
                size_hints = [
                    get_xpu_1d_hint(numel_0, cluster_num),
                    1,
                ]
        else:
            raise AssertionError(f"invalid size for numels {len(self.numels)}")
        # ===-----------------------------------------------------------===

        if self.persistent_reduction:
            assert self.inside_reduction
            heuristics = "persistent_reduction"
        elif self.inside_reduction:
            heuristics = "reduction"
        else:
            size_hints.pop()
            heuristics = "pointwise"

        if name is None:
            code.splice(f"""
                    import triton
                    import triton.language as tl
                    from torch._inductor.ir import ReductionHint
                    from torch._inductor.ir import TileHint
                    from torch._inductor.triton_ops.autotune import {heuristics}
                    from torch._inductor.utils import instance_descriptor
                    # ===-------------------- For Triton XPU -----------------------===
                    # Borrowed From Pytorch(v2.5.0-rc9) torch/_inductor/runtime/triton_helpers.py
                    # In the latest triton, math functions were shuffled around into different modules:
                    # https://github.com/openai/triton/pull/3172
                    try:
                        from triton.language.extra import libdevice

                        libdevice = tl.extra.libdevice  # noqa: F811
                        math = tl.math
                    except ImportError:
                        if hasattr(tl.extra, "xpu") and hasattr(tl.extra.xpu, "libdevice"):
                            libdevice = tl.extra.xpu.libdevice
                            math = tl.math
                        elif hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "libdevice"):
                            libdevice = tl.extra.cuda.libdevice
                            math = tl.math
                        elif hasattr(tl.extra, "intel") and hasattr(tl.extra.intel, "libdevice"):
                            libdevice = tl.extra.intel.libdevice
                            math = tl.math
                        else:
                            libdevice = tl.math
                            math = tl
                    # ===-----------------------------------------------------------===
                """)

        argdefs, _, signature = self.args.python_argdefs()
        # maps actual expression to SizeArg if its in sizevars replacements
        for i, arg in enumerate(signature):
            if (isinstance(arg, SizeArg) and arg.expr in V.graph.sizevars.inv_precomputed_replacements):
                signature[i] = SizeArg(arg.name, V.graph.sizevars.inv_precomputed_replacements[arg.expr])

        mutated_args = set()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if mutation in self.args.inplace_buffers:
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)

        global xpu_hasAtomic
        triton_meta = {
            "signature": dict(enumerate(map(signature_of, signature))),
            "device": V.graph.scheduler.current_device.index,
            "constants": {},
            "mutated_arg_names": mutated_args,
            "hasAtomic": xpu_hasAtomic,
        }
        xpu_hasAtomic = False  # reset global var

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
                signature.append(sizearg)
                triton_meta["signature"][len(argdefs)] = signature_of(sizearg)
                argdefs.append(f"{tree.prefix}numel")
                # constexpr version causes issues, see
                # https://github.com/pytorch/torchdynamo/pull/1362
                # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
                #     tree.numel
                # )
                # argdefs.append(f"{tree.prefix}numel: tl.constexpr")
        triton_meta["configs"] = [config_of(signature)]

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f"""
                @{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    meta={triton_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                if len(signature) == 4:  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @{heuristics}(size_hints={size_hints!r}, {tile_hint}filename=__file__, meta={triton_meta!r})
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(f"def {name or 'KERNEL_NAME'}({', '.join(argdefs)}):")
        self.codegen_body()
        with code.indent():
            if not dynamo_config.dynamic_shapes:
                self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if name is not None:
            return code.getvalue()

        wrapper = IndentedBuffer()
        wrapper.writeline("async_compile.triton('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''')")
        return wrapper.getvalue()


triton.TritonKernel = XPUTritonKernel

# ===------------------------------------------------------------------===
