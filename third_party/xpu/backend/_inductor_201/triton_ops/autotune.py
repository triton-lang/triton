import os
import copy
import torch
from torch._inductor.triton_ops import has_triton
from torch._inductor.utils import ceildiv

from .runtime_utils import (
    get_first_attr, )

if has_triton():
    import triton
    from triton import cdiv, Config, next_power_of_2
    from triton.runtime.jit import get_cuda_stream, KernelInterface
    try:
        from triton.compiler.compiler import ASTSource
    except ImportError:
        ASTSource = None

    try:
        from triton.backends.compiler import GPUTarget
    except ImportError:
        GPUTarget = None
else:
    cdiv = None
    Config = object
    get_cuda_stream = None
    KernelInterface = object
    next_power_of_2 = None
    triton = None
    ASTSource = None
    GPUTarget = None

from torch._inductor.triton_ops import autotune

from xpu.backend.driver import get_xpu_spec

arch = int(os.environ.get('TRITON_XPU_ARCH', '3'))
CLUSTER_NUM = get_xpu_spec(arch)[0]
CORE_NUM = get_xpu_spec(arch)[1]


# ===-------------------- For XPytorch Inductor -----------------------===
# Base Pytorch(v2.0.1) torch/_inductor/triton_ops/autotune.py
# vvv
# Target Pytorch(v2.5.0-rc9) torch/_inductor/runtime/triton_heuristics.py
class XPUCachingAutotuner(autotune.CachingAutotuner):

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: int):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.meta)
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages

        compile_meta["device_type"] = "xpu"
        compile_meta["cc"] = 3
        compile_meta["debug"] = True

        if ASTSource:
            compile_args = (ASTSource(
                self.fn,
                compile_meta["signature"],
                compile_meta["constants"],
                compile_meta["configs"][0],
            ), )

            cc_str = str(compile_meta["cc"])
            if "gfx10" in cc_str or "gfx11" in cc_str:
                rocm_warp_size = 32
            else:
                rocm_warp_size = 64

            if GPUTarget:
                target = GPUTarget(
                    compile_meta["device_type"],
                    compile_meta["cc"],
                    rocm_warp_size if torch.version.hip else 32,
                )
            else:
                target = ((compile_meta["device_type"], compile_meta["cc"]) if not torch.version.hip else [
                    compile_meta["device_type"],
                    compile_meta["cc"],
                    rocm_warp_size,
                ])

            options = {
                "num_warps": compile_meta["num_warps"],
                "num_stages": compile_meta["num_stages"],
                "debug": compile_meta["debug"],
            }
            # if self.device_props.type != "hip":
            #     if "waves_per_eu" in compile_meta:
            #         options["waves_per_eu"] = compile_meta["waves_per_eu"]
            #     if "matrix_instr_nonkdim" in compile_meta:
            #         options["matrix_instr_nonkdim"] = compile_meta[
            #             "matrix_instr_nonkdim"
            #         ]
            compile_kwargs = {
                "target": target,
                "options": options,
            }
        else:
            compile_args = (self.fn, )
            compile_kwargs = compile_meta

        if warm_cache_only_with_cc:
            triton.compile(*compile_args, **compile_kwargs)
            return

        # load binary to the correct device
        with torch.cuda.device(compile_meta["device"]):
            # need to initialize context
            torch.cuda.synchronize(torch.cuda.current_device())
            binary = triton.compile(*compile_args, **compile_kwargs)
            binary._init_handles()

        call_args = [arg for i, arg in enumerate(self.fn.arg_names) if i not in self.fn.constexprs]
        def_args = list(self.fn.arg_names)
        while def_args and def_args[-1] in cfg.kwargs:
            def_args.pop()

        binary_shared = (binary.shared if hasattr(binary, "shared") else binary.metadata.shared)

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "torch": torch,
            "set_device": torch.cuda.set_device,
            "current_device": torch.cuda.current_device,
            "metadata": binary.packed_metadata,
            "launch_enter_hook": binary.launch_enter_hook,
            "launch_exit_hook": binary.launch_exit_hook,
            "shared": binary_shared,
        }

        scope["num_warps"] = (binary.num_warps if hasattr(binary, "num_warps") else binary.metadata.num_warps)

        scope["cta_args"] = ((binary.num_ctas, *get_first_attr(binary, "cluster_dims", "clusterDims")) if hasattr(
            binary, "num_ctas") else ((binary.metadata.num_ctas,
                                       *binary.metadata.cluster_dims) if hasattr(binary, "metadata") else ()))

        scope["function"] = get_first_attr(binary, "function", "cu_function")

        def get_launch_args_without_kernel_launch_metadata(
            grid,
            grid_0,
            grid_1,
            grid_2,
            stream,
            function,
            metadata,
            bin,
            launch_enter_hook,
            launch_exit_hook,
            num_warps,
            shared,
            cta_args,
            args,
        ):
            """
            Construct launch args before CompiledKernel.launch_metadata is added.
            """
            return (
                grid_0,
                grid_1,
                grid_2,
                num_warps,
                *cta_args,
                shared,
                stream,
                function,
                launch_enter_hook,
                launch_exit_hook,
                metadata,
            )

        # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
        # `bin.launch_metadata` is relatively expensive, and returns None unless a
        # `launch_enter_hook` is installed.  So if we don't have that hook installed,
        # we want to burn None in to the launch args with zero overhead.
        # See https://github.com/pytorch/pytorch/issues/123597
        if binary.launch_enter_hook:

            def get_launch_args_with_kernel_launch_metadata(
                grid,
                grid_0,
                grid_1,
                grid_2,
                stream,
                function,
                metadata,
                bin,
                launch_enter_hook,
                launch_exit_hook,
                num_warps,
                shared,
                cta_args,
                args,
            ):
                """
                Construct launch args after CompiledKernel.launch_metadata is added
                by https://github.com/openai/triton/pull/3492 .
                """
                return (
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    bin.launch_metadata(grid, stream, *args),
                    launch_enter_hook,
                    launch_exit_hook,
                )

        else:

            def get_launch_args_with_kernel_launch_metadata(
                grid,
                grid_0,
                grid_1,
                grid_2,
                stream,
                function,
                metadata,
                bin,
                launch_enter_hook,
                launch_exit_hook,
                num_warps,
                shared,
                cta_args,
                args,
            ):
                """
                Construct launch args after CompiledKernel.launch_metadata is added
                by https://github.com/openai/triton/pull/3492 .
                """
                return (
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    None,
                    launch_enter_hook,
                    launch_exit_hook,
                )

        scope["get_launch_args"] = (get_launch_args_with_kernel_launch_metadata if hasattr(binary, "launch_metadata")
                                    else get_launch_args_without_kernel_launch_metadata)

        scope["runner"] = get_first_attr(binary, "run", "c_wrapper")
        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid

                args = {', '.join(call_args)},
                launch_args = get_launch_args(
                    grid, grid_0, grid_1, grid_2, stream, function,
                    metadata, bin, launch_enter_hook, launch_exit_hook,
                    num_warps, shared, cta_args, args
                )
                runner(*launch_args, *args)
                return bin
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        return launcher


autotune.CachingAutotuner = XPUCachingAutotuner
# ===------------------------------------------------------------------===

# ===-------------------- For XPytorch Inductor -----------------------===
from torch._inductor.triton_ops.autotune import cached_autotune, triton_config
from torch._inductor import config

# Modified Pytorch(v2.0.1) torch/_inductor/triton_ops/autotune.py::reduction() && pointwise()


def triton_xpu_config_reduction(size_hints, x, r, num_stages=2) -> Config:
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """

    cfg = {"XBLOCK": x, "RBLOCK": r}
    num_warps = -1  # invalid value, just a placeholder
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def tritonxpu_reduction(size_hints, reduction_hint=False, meta=None, filename=None):
    from torch._inductor.ir import ReductionHint
    import math
    """args to @triton.heuristics()"""
    assert meta is not None

    # ===-------------------- For Triton XPU -----------------------===
    if bool(meta.get("hasAtomic", False)):
        xnumel = math.ceil(size_hints[0] / 1)
    else:
        xnumel = math.ceil(size_hints[0] / CLUSTER_NUM)
    # ===-----------------------------------------------------------===

    rnumel = size_hints[-1]
    if len(size_hints) == 2:
        contiguous_config = triton_xpu_config_reduction(size_hints, xnumel, (rnumel if 0 < rnumel < 8192 else 8192),
                                                        num_stages=1)
        buffersize_config = triton_xpu_config_reduction(size_hints, xnumel, (rnumel if 0 < rnumel < 128 else 128),
                                                        num_stages=1)

        if config.max_autotune:
            pass  # skip all these cases
        elif reduction_hint == ReductionHint.INNER or ReductionHint.DEFAULT:
            return cached_autotune(configs=[contiguous_config], meta=meta)
        else:
            raise NotImplementedError(f"reduction_hint: {reduction_hint}")

    raise NotImplementedError(f"size_hints: {size_hints}")


def tritonxpu_pointwise(size_hints, meta, tile_hint=None, filename=None):
    import functools
    import operator
    """
    Construct @triton.heuristics() based on size_hints.
    """
    # ===-------------------- For Triton XPU -----------------------===
    numel = functools.reduce(operator.mul, size_hints)
    if bool(meta.get("hasAtomic", False)):
        # We need to tile all data in only one cluster for atomic simulation
        bs = max(CORE_NUM, numel // 1)
    else:
        bs = max(CORE_NUM, numel // CLUSTER_NUM)
    # ===-----------------------------------------------------------===

    if len(size_hints) == 1:
        return cached_autotune([triton_config(size_hints, bs)], meta=meta)
    if len(size_hints) == 2:
        raise NotImplementedError(f"[Triton XPU] len(size_hints) == 2 Not Supported")
    if len(size_hints) == 3:
        raise NotImplementedError(f"[Triton XPU] len(size_hints) == 3 Not Supported")
    raise NotImplementedError(f"size_hints: {size_hints}")


from torch._inductor.triton_ops import autotune

autotune.reduction = tritonxpu_reduction
autotune.pointwise = tritonxpu_pointwise

# ===------------------------------------------------------------------===


# ===-------------------- For XPytorch Inductor -----------------------===
def grid(xnumel, ynumel=None, znumel=None):
    """Helper function to compute triton grids"""

    def get_grid_dim(numel, block_name, block):
        if numel is None:
            return 1
        core_nums = CLUSTER_NUM * CORE_NUM
        grid_num = ceildiv(numel, block) if numel < core_nums else CLUSTER_NUM
        return grid_num

    def grid_fn(meta):
        return (
            get_grid_dim(xnumel, "XBLOCK", meta.get("XBLOCK", 1)),
            get_grid_dim(ynumel, "YBLOCK", meta.get("YBLOCK", None)),
            get_grid_dim(znumel, "ZBLOCK", meta.get("ZBLOCK", None)),
        )

    return grid_fn


autotune.grid = grid

# ===------------------------------------------------------------------===
