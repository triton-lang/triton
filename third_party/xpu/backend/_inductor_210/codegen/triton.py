import os
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
from typing import Dict, Iterable, List, Set

import sympy

import torch

import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges

from torch._inductor import ir
from torch._inductor.codegen import triton
from torch._inductor.utils import (
    DeferredLineBase,
    get_fused_kernel_name,
    get_kernel_metadata,
    green_text,
    is_welford_reduction,
    next_power_of_2,
    sympy_product,
    sympy_subs,
    sympy_symbol,
    unique,
    yellow_text,
)
from torch._inductor import config
from torch._inductor.virtualized import ops, V
from torch._inductor.codegen.common import (
    CSEVariable,
    DeferredLine,
    free_symbol_startswith,
    IndentedBuffer,
    index_prevent_reordering,
    Kernel,
    OpOverrides,
    PythonPrinter,
    SizeArg,
)

from torch._inductor.codegen.triton import triton_compute_type, triton_constant, texpr
from torch._dynamo.utils import counters

from torch._inductor.codegen.triton_utils import (
    config_of,
    signature_of,
    signature_to_meta,
)

CLUSTER_NUM = 8
CORE_NUM = 64


class TritonXPUPrinter(triton.TritonPrinter):

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return f"tl.libdevice.floor({self.paren(self._print(expr.args[0]))})"

    def _helper_sqrt(self, expr):
        return f"tl.libdevice.sqrt({self.paren(self._print(expr))}.to(tl.float32))"

    def _print_Min(self, expr):
        nargs = len(expr.args)
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        a = self._print(sympy.Min(*expr.args[:mid]))
        b = self._print(sympy.Min(*expr.args[mid:]))
        return f"tl.libdevice.min({a}, {b})"

    def _print_Max(self, expr):
        nargs = len(expr.args)
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        a = self._print(sympy.Max(*expr.args[:mid]))
        b = self._print(sympy.Max(*expr.args[mid:]))
        return f"tl.libdevice.max({a}, {b})"


triton.TritonPrinter = TritonXPUPrinter
texpr = TritonXPUPrinter().doprint


class XPUIterationRangesEntry(triton.IterationRangesEntry):

    def _codegen(self):
        self.writeline(f"{self.name} = " + texpr(V.kernel.rename_indexing(self.expr)))
        return self.name


triton.IterationRangesEntry = XPUIterationRangesEntry


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


class IterationXPURangesRoot(triton.IterationRangesRoot):
    # Remove no_x_dim mode(test_bilibili_layernorm.py)
    def codegen_header(self, code, no_x_dim=False):
        x = self.prefix
        if self.is_loop():
            code.writeline(f"{self.name} = {x}offset + {x}base")
        elif x == "r" and self.kernel.persistent_reduction:
            # no need to "roffset = "
            code.writeline(f"{self.name} = {self.ranges_code()}", )
        else:
            line = f"{x}offset + {self.ranges_code()}"
            code.writelines([
                f"{x}offset = {self.get_pid()} * {x.upper()}BLOCK",
                f"{self.name} = {line}",
            ])
        code.writeline(f"{x}mask = {self.name} < {x}numel")


triton.IterationRangesRoot = IterationXPURangesRoot

xpu_hasAtomic = False


class TritonXPUKernel(triton.TritonKernel):

    overrides = XPUTritonOverrides

    # Remove evict_last flag, it is only used in cuda cache(test_softmax.py perf)
    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        indirect_indexing = self.is_indirect_indexing(index)
        original_index = index
        index, mask_vars, mask, expand_str = self.indexing(index)

        ep = ""
        # "other" below is a workaround for https://github.com/openai/triton/issues/737
        # for bool, even though it's likely subject to the same bug, setting `other` leads
        # to LLVM errors so we are skipping it for now
        if ("tmp" in mask or "rmask" in mask) and V.graph.get_dtype(name) != torch.bool:
            other = ", other=0"
        else:
            other = ""

        append_broadcast = None
        if V.graph.is_unspec_arg(name):
            line = var
        else:
            if isinstance(original_index, sympy.Integer):
                line = f"tl.load({var} + ({original_index}))"
                append_broadcast = expand_str
            else:
                line = f"tl.load({var} + ({index}), {mask}{ep}{other})"
            if V.graph.get_dtype(name) in (torch.float16, torch.bfloat16):
                line += ".to(tl.float32)"

        if "tmp" in mask:
            # Masked loads must come after the mask is computed
            load_buffer = self.compute
        elif (self.inside_reduction and not self.persistent_reduction and "rmask" not in mask
              and not indirect_indexing):
            # can lift a common load outside of reduction loop
            # One exception is when this is an indirect_load.
            load_buffer = self.body
        else:
            load_buffer = self.loads

        result_var = self.cse.generate(load_buffer, line)
        result_var.mask_vars = mask_vars

        if append_broadcast:
            line = f"tl.broadcast_to({result_var}, {append_broadcast})"
            result_var = self.cse.generate(load_buffer, line)

        if not self.inside_reduction or "rmask" not in mask:
            self.outside_loop_vars.add(result_var)

        return result_var

    # Remove tl.debug_barrier()(test_ks_batchnorm_online.py)
    def store(self, name, index, value, mode=None):
        var = self.args.output(name)
        indirect_indexing = self.is_indirect_indexing(index)
        original_index = index
        index, mask_vars, mask, expand_str = self.indexing(index, dense_indexing=True)

        if mode is None:
            line = f"tl.store({var} + ({index}), {value}, {mask})"
        elif mode == "atomic_add":
            line = f"tl.atomic_add({var} + ({index}), {value}, {mask})"
            global xpu_hasAtomic
            xpu_hasAtomic = True
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(DeferredLine(name, line))
        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        assert self.inside_reduction
        masks = {f"{tree.prefix}mask" for tree in self.range_trees}
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
        reduction_range_prefix = self.range_trees[-1].prefix
        reduction_sizes = ["None" for _ in self.range_trees]
        reduction_sizes[-1] = ":"

        # Say we have
        #     tmp0 = ops.constant(1, torch.int64)
        #     tmp1 = ops.reduction(torch.int64, torch.int64, "sum", tmp0)
        # tmp0 in the triton code is either a scalar, or single-element tensor
        # so if we emit tl.sum directly, it will only give 1 instead of RBLOCK * 1
        # To avoid this, we broadcast to the expected shape first.
        dense_size_str = self.dense_size_str()
        value = self._map_tuple_or_scalar(
            lambda v: self.cse.generate(self.compute, f"tl.broadcast_to({v}, {dense_size_str})"),
            value,
        )

        def final_reduction(value):
            module = "tl"
            return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})")

        def final_argreduce(buffer, result_var, value, index):
            buffer.splice(f"""\
                _, {result_var}_tmp = triton_helpers.{root_op}_with_index({value}, {index}, {dim})
                {result_var} = {self.reduction_resize(f'{result_var}_tmp')}
                """)

        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        dim = len(self.range_trees) - 1 - int(bool(self.no_x_dim))
        acc_type = triton.triton_acc_type(src_dtype)
        result_var = self.cse.newvar()
        result_var.mask_vars = {var for var in masks if var[0] != "r"}
        cond = " & ".join(masks)

        if self.persistent_reduction:
            default = ir.Reduction.default_value(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(triton.triton_constant, default)

            def _mask_value(value, default):
                return self.cse.generate(self.compute, f"tl.where({cond}, {value}, {default})")

            if isinstance(value, tuple):
                masked_value = [_mask_value(v, d) for v, d in zip(value, default)]
            else:
                masked_value = _mask_value(value, default)

            if reduction_type in {"argmax", "argmin"}:
                accumulator_index = self.cse.generate(
                    self.compute,
                    f"tl.broadcast_to({reduction_range_prefix}index, {masked_value}.shape)",
                )
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
                final_argreduce(self.compute, result_var, masked_value, accumulator_index)
            elif reduction_type == "welford_reduce":
                # For persistent reductions, don't bother with
                # welford's algorithm since it uses more registers, and
                # taking two reductions doesn't increase memory usage.
                sum_ = ops.reduction(dtype, dtype, "sum", value)
                self.inside_reduction = False
                rnumel = ops.index_expr(self.numels[-1], dtype)
                mean = ops.div(sum_, rnumel)

                self.inside_reduction = True
                dx = ops.sub(value, mean)
                dx2 = ops.mul(dx, dx)
                m2 = ops.reduction(dtype, dtype, "sum", dx2)
                result_var = (mean, m2, rnumel)
            elif reduction_type == "welford_combine":
                mean, m2, weight = masked_value
                welford = f"triton_helpers.welford({mean}, {m2}, {weight}, {dim})"
                mean, m2, weight = (self.cse.newvar() for _ in range(3))
                self.compute.writeline(f"{mean}, {m2}, {weight} = {welford}")

                result_var = tuple(
                    self.cse.generate(self.compute, self.reduction_resize(var_name)) for var_name in (mean, m2, weight))
            else:
                result_var = self.cse.generate(self.compute, final_reduction(masked_value))
        else:
            accumulator = f"_{result_var}"
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(triton.triton_constant, default)
            if not isinstance(default, tuple):
                self.body.writeline(f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})")

            if reduction_type in {"argmax", "argmin"}:
                accumulator_index = f"_{result_var}_index"
                long_max = torch.iinfo(torch.int64).max
                self.body.writeline(f"{accumulator_index} = tl.full({self.dense_size_str()}, {long_max}, tl.int64)")
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]

                self.compute.splice(f"""\
                {accumulator}_next, {accumulator_index}_next = triton_helpers.{root_op}imum_with_index(
                    {accumulator}, {accumulator_index}, {value}, {reduction_range_prefix}index
                )
                {accumulator} = tl.where({cond}, {accumulator}_next, {accumulator})
                {accumulator_index} = tl.where({cond}, {accumulator_index}_next, {accumulator_index})
                """)
                final_argreduce(self.suffix, result_var, accumulator, accumulator_index)
            elif is_welford_reduction(reduction_type):
                accumulator = f"{result_var}_mean"
                accumulator_m2 = f"{result_var}_m2"
                accumulator_weight = f"{result_var}_weight"
                self.body.writeline(f"{accumulator} = tl.zeros({self.dense_size_str()}, {acc_type})")
                self.body.writeline(f"{accumulator_m2} = tl.zeros({self.dense_size_str()}, {acc_type})")
                self.body.writeline(f"{accumulator_weight} = tl.zeros({self.dense_size_str()}, {acc_type})")

                if reduction_type == "welford_combine":
                    mean, m2, weight = value
                    self.compute.splice(f"""\
                    {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_combine(
                        {accumulator}, {accumulator_m2}, {accumulator_weight},
                        {mean}, {m2}, {weight}
                    )
                    """)
                else:
                    assert reduction_type == "welford_reduce"
                    self.compute.splice(f"""\
                    {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_reduce(
                        {value}, {accumulator}, {accumulator_m2}, {accumulator_weight},
                    )
                    """)

                self.compute.splice(f"""\
                {accumulator} = tl.where({cond}, {accumulator}_next, {accumulator})
                {accumulator_m2} = tl.where({cond}, {accumulator_m2}_next, {accumulator_m2})
                {accumulator_weight} = tl.where({cond}, {accumulator_weight}_next, {accumulator_weight})
                """)

                result_mean = result_var
                result_m2 = self.cse.newvar()
                result_weight = self.cse.newvar()
                self.suffix.splice(f"""\
                {result_mean}_tmp, {result_m2}_tmp, {result_weight}_tmp = triton_helpers.welford(
                    {accumulator}, {accumulator_m2}, {accumulator_weight}, {dim}
                )
                {result_mean} = {self.reduction_resize(f'{result_mean}_tmp')}
                {result_m2} = {self.reduction_resize(f'{result_m2}_tmp')}
                {result_weight} = {self.reduction_resize(f'{result_weight}_tmp')}
                """)
                result_var = result_mean, result_m2, result_weight
            else:
                combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
                updated = combine_fn(accumulator, value)
                self.compute.writeline(f"{accumulator} = tl.where({cond}, {updated}, {accumulator})")

                if src_dtype == torch.bool:
                    # This is only really used for aten.any. It changes the
                    # final reduction of a non-persistent reduction from
                    #     tmp5 = triton_helpers.max(_tmp5, 1)[:, None]
                    # to
                    #     tmp5 = triton_helpers.max(_tmp5.to(tl.int8), 1)[:, None].to(tl.int1)
                    # which is needed because tl.reduce doesn't support tl.int1
                    accumulator = f"{accumulator}.to(tl.int8)"
                    result_type = triton_compute_type(dtype)
                    self.suffix.writeline(f"{result_var} = {final_reduction(accumulator)}.to({result_type})")
                else:
                    self.suffix.writeline(f"{result_var} = {final_reduction(accumulator)}")

        self.cse.reduction_cache[cache_key] = result_var

        if isinstance(result_var, tuple):
            self.outside_loop_vars |= set(result_var)
        else:
            self.outside_loop_vars.add(result_var)

        return result_var

    # Remove tl.where(test_gather.py)
    def indirect_indexing(self, var, size, check=True):
        # TODO(lezcano) This code should be lifted to codegen/common.py.
        # This should be easy, as now CSE variables carry bounds info
        class IndirectAssertLine(DeferredLineBase):

            def __init__(self, line, var, mask, size_map):
                self.var = var
                self.mask = mask
                self.line = line
                self.size_map = size_map

            def __call__(self):
                size, size_str = self.size_map[(self.var, self.mask)]

                # We assert if we've not been able to prove the bound
                assert_min = (self.var.bounds.lower >= 0) != sympy.true
                assert_max = (self.var.bounds.upper < size) != sympy.true

                # FooBar interview question
                if not (assert_min or assert_max):
                    return None
                elif assert_min and assert_max:
                    # The conditions need to be in parens because of Python's operator precedence.
                    # It'd be less error-prone to use and/or/not, which is suported by triton
                    cond = f"(0 <= {self.var}) & ({self.var} < {size_str})"
                    cond_print = f"0 <= {self.var} < {size_str}"
                elif assert_min:
                    cond = f"0 <= {self.var}"
                    cond_print = cond
                else:
                    assert assert_max
                    cond = f"{self.var} < {size_str}"
                    cond_print = cond

                if self.mask:
                    cond = f"({cond}) | ~{self.mask}"
                return self.line.format(cond=cond, cond_print=cond_print)

            def _new_line(self, line):
                return IndirectAssertLine(line, self.var, self.mask, self.size_map)

        generate_assert = ((check or config.debug_index_asserts) and config.triton.assert_indirect_indexing
                           and torch.version.hip is None)
        if generate_assert:
            mask_vars = set(var.mask_vars)
            if self._load_mask:
                mask_vars.add(self._load_mask)

            mask = ""
            if mask_vars:
                mask = (f"{list(mask_vars)[0]}"
                        if len(mask_vars) == 1 else f"({' & '.join(str(v) for v in mask_vars)})")

            # An assertion line may have been written already, if so just
            # update the max size.
            map_key = (var, mask)
            existing_size, _ = self.indirect_max_sizes.get(map_key, (None, None))
            if existing_size is not None:
                size = sympy.Min(size, existing_size)

            self.indirect_max_sizes[map_key] = (size, self.index_to_str(size))

        return sympy_symbol(str(var))

    def index_to_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expr to a string that can be used in triton code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the triton kernel.

        Index expressions often need to be passed in as arguments to the triton kernel.
        Rename_indexing and codegen_indexing keep track of the needed indices and add
        new parameters to the function signature.
        """
        return texpr(self.rename_indexing(self.codegen_indexing(index)))

    def codegen_kernel(self, name=None):

        from triton import next_power_of_2

        def next_multiply_of_512(n: int) -> int:
            m = CLUSTER_NUM * CORE_NUM  # 8 cluster 64 core
            if n < m:
                res = next_power_of_2(n)
            else:
                res = math.ceil(n / m) * m
            return res

        code = IndentedBuffer()

        if len(self.numels) == 1:
            size_hints = [next_power_of_2(V.graph.sizevars.size_hint(numel)) for numel in self.numels]
        elif len(self.numels) == 2:
            if self.numels[1] != 1:
                size_hints = [
                    next_multiply_of_512(V.graph.sizevars.size_hint(self.numels[0])),
                    next_power_of_2(V.graph.sizevars.size_hint(self.numels[1])),
                ]
            else:
                size_hints = [next_power_of_2(V.graph.sizevars.size_hint(numel)) for numel in self.numels]
        else:
            raise AssertionError(f"invalid size for numels {len(self.numels)}")

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
                    from torch._inductor.triton_heuristics import AutotuneHint, {heuristics}
                    from torch._inductor.utils import instance_descriptor
                    from torch._inductor import triton_helpers
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
            if config.benchmark_kernel:
                code.splice("""
                        from torch._dynamo.testing import rand_strided
                        from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
                        import torch
                        from torch._inductor.triton_heuristics import grid
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
            if (mutation in self.args.inplace_buffers and mutation not in V.graph.removed_buffers):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)

        global xpu_hasAtomic
        triton_meta = {
            "signature": signature_to_meta(signature, size_dtype=self.index_dtype),
            "device": V.graph.scheduler.current_device.index,
            "device_type": V.graph.scheduler.current_device.type,
            "constants": {},
            "mutated_arg_names": mutated_args,
            "autotune_hints": set(self.autotune_hints),
            "kernel_name": "DESCRIPTIVE_KRNL_NAME",
            "hasAtomic": xpu_hasAtomic,
        }
        xpu_hasAtomic = False  # reset global var

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
                signature.append(sizearg)
                triton_meta["signature"][len(argdefs)] = signature_of(sizearg, size_dtype=self.index_dtype)
                argdefs.append(f"{tree.prefix}numel")
                # constexpr version causes issues, see
                # https://github.com/pytorch/torchdynamo/pull/1362
                # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
                #     tree.numel
                # )
                # argdefs.append(f"{tree.prefix}numel: tl.constexpr")
        triton_meta["configs"] = [config_of(signature)]

        for tree in self.range_trees:
            if tree.prefix == "r" and (not self.inside_reduction or self.persistent_reduction):
                continue
            if tree.prefix == "x" and self.no_x_dim:
                continue
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
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark())

        if name is not None:
            return code.getvalue()

        return code.getvalue()


triton.TritonKernel = TritonXPUKernel

import torch._inductor.scheduler as scheduler
from torch._inductor.codegen.triton import EnableReduction

perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")


class TritonXPUScheduling(triton.TritonScheduling):

    @classmethod
    def select_tiling(cls, node_schedule, numel, reduction_numel=sympy.Integer(1)):
        """
        Heuristics to decide how to tile kernels.
        Currently, we tile based on stride-1 dimensions.

        Returns:
            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`

        """
        if reduction_numel != 1 or config.triton.max_tiles <= 1:
            # TODO(jansel): should we tile reductions?
            # do perf hint here if stride-1 dim is not being reduced
            if perf_hint_log.level <= logging.WARNING:
                for node in EnableReduction.filter(node_schedule):
                    if len(cls.candidate_tilings(node)) > 0:
                        perf_hint_log.info("reduction over non-contiguous dims")
                        break
            return (numel, reduction_numel)

        seen_names = set()
        candidate_tiles = collections.Counter()
        for node in EnableReduction.filter(node_schedule):
            for tiling in cls.candidate_tilings(node):
                if tiling.name in seen_names:
                    continue
                seen_names.add(tiling.name)
                candidate_tiles[tiling.tiling] += tiling.score

        ranked_tilings = [tiling for tiling, score in candidate_tiles.most_common()]

        if config.triton.max_tiles >= 3:
            # Consider adding a third dimension of tiling, but only
            # when a1 is a multiple of b1; otherwise, you have a lot
            # of stragglers which is annoying to generate code for.
            #
            # NB: More than three max tiles is not enabled by default.

            # Add one 3D tiling choice
            for i in range(1, len(ranked_tilings)):
                a0, a1 = ranked_tilings[0]
                b0, b1 = ranked_tilings[i]
                if V.graph.sizevars.size_hint(a1 - b1) == 0:
                    continue
                if V.graph.sizevars.size_hint(a1 - b1) < 0:
                    # swap so a0 is bigger
                    a0, a1 = ranked_tilings[i]
                    b0, b1 = ranked_tilings[0]
                assert V.graph.sizevars.size_hint(a1 - b1) > 0
                if V.graph.sizevars.statically_known_multiple_of(a1, b1):
                    tiling = (a0, FloorDiv(a1, b1), b1)
                    ranked_tilings = [tiling] + ranked_tilings
                    break  # only 1 choice for now

        if len(ranked_tilings) > 1:
            perf_hint_log.info("possibly bad tiling: %s", ranked_tilings)

        for tiled_groups in ranked_tilings:
            # [TODO]: Remove this tiling limit
            if (len(tiled_groups) != 1) and all(isinstance(node.node.data, ir.Pointwise) for node in node_schedule):
                print(f"[TORCH_XPU Warning] Pointwise Kernel Limit To 1D-Tiling")
                return (numel, reduction_numel)
            new_groups = (*tiled_groups, reduction_numel)
            if all(
                    triton.TritonKernel.is_compatible(new_groups, node.get_ranges())
                    for node in node_schedule
                    if isinstance(node, scheduler.SchedulerNode)):
                return new_groups

        return (numel, reduction_numel)


triton.TritonScheduling = TritonXPUScheduling
