import os
import copy
import math
import torch
from torch._inductor.utils import (has_triton, ceildiv)
from torch._inductor import config
import torch.autograd.profiler as autograd_profiler

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

from torch._inductor import (
    triton_heuristics, )
from torch._inductor.triton_heuristics import (cached_autotune, HeuristicType)
from torch._inductor.ir import TileHint, ReductionHint

from xpu.backend.driver import get_xpu_spec

arch = int(os.environ.get('TRITON_XPU_ARCH', '3'))
CLUSTER_NUM = get_xpu_spec(arch)[0]
CORE_NUM = get_xpu_spec(arch)[1]


# ===-------------------- For XPytorch Inductor -----------------------===
# Base Pytorch(v2.1.0) torch/_inductor/triton_heuristics.py
# vvv
# Target Pytorch(v2.5.0-rc9) torch/_inductor/runtime/triton_heuristics.py
class XPUCachingAutotuner(triton_heuristics.CachingAutotuner):

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: int):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.meta)
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages

        compile_meta["device_type"] = "xpu"
        compile_meta["cc"] = 3
        compile_meta["debug"] = False

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

    def run(self, *args, grid, stream):
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                self.precompile()
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, grid=grid)

        if (not getattr(self.launchers[0].config, "found_by_coordesc", False) and config.coordinate_descent_tuning):
            self.launchers = [self.coordinate_descent_tuning(self.launchers[0], *args, grid=grid)]

        (launcher, ) = self.launchers

        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook({**dict(zip(self.arg_names, args)), **launcher.config.kwargs})

        # guard the record_function_ctx and only call it if profiling is currently
        # in progress, to reduce latency when profiler is not turned on. Note that
        # the "if" statement (instead of, say, a contextlib.nullcontext) is intentional;
        # it is faster than entering and exiting a context manager, even if the context
        # manager is a nullcontext.
        if autograd_profiler._is_profiler_enabled:
            with self.record_function_ctx:
                return launcher(
                    *args,
                    grid=grid,
                    stream=stream,
                )
        else:
            return launcher(
                *args,
                grid=grid,
                stream=stream,
            )


triton_heuristics.CachingAutotuner = XPUCachingAutotuner
# ===------------------------------------------------------------------===

# ===-------------------- For XPytorch Inductor -----------------------===


def triton_config(size_hints, x, y=None, z=None, num_stages=1, num_elements_per_warp=256) -> Config:

    cfg = {"XBLOCK": x}
    if y:
        cfg["YBLOCK"] = y
    if z:
        cfg["ZBLOCK"] = z

    num_warps = 16  # num_warps represents groups in XPU2/xpu3(16)
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def tritonxpu_pointwise(size_hints, meta, tile_hint=None, filename=None):
    import functools
    import operator
    """
    Construct @triton.heuristics() based on size_hints.
    """
    numel = functools.reduce(operator.mul, size_hints)
    bs = max(256, min(numel // 128, 1024))

    if len(size_hints) == 1:
        # TODO: make it more tunable.
        if bool(meta.get("hasAtomic", False)):
            # We need to tile all data in only one cluster for atomic simulation
            bs = max(CORE_NUM, math.ceil(numel / 1))
        else:
            bs = max(CORE_NUM, math.ceil(numel / CLUSTER_NUM))
        return cached_autotune(
            size_hints=size_hints,
            configs=[triton_config(size_hints, bs)],
            meta=meta,
            heuristic_type=HeuristicType.POINTWISE,
            filename=filename,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def triton_xpu_config_reduction(size_hints, x, r, num_stages=2) -> Config:

    cfg = {"XBLOCK": x, "RBLOCK": r}
    num_warps = 16  # num_warps represents groups in XPU2/XPU3(16)
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def tritonxpu_reduction(size_hints, reduction_hint=False, meta=None, filename=None):
    """args to @triton.heuristics()"""
    assert meta is not None
    if bool(meta.get("hasAtomic", False)):
        xnumel = math.ceil(size_hints[0] / 1)
    else:
        xnumel = math.ceil(size_hints[0] / CLUSTER_NUM)
    rnumel = size_hints[-1]

    if len(size_hints) == 2:
        contiguous_config = triton_xpu_config_reduction(size_hints, xnumel, (rnumel if 0 < rnumel < 8192 else 8192),
                                                        num_stages=1)
        buffersize_config = triton_xpu_config_reduction(size_hints, xnumel, (rnumel if 0 < rnumel < 128 else 128),
                                                        num_stages=1)
        if config.max_autotune:
            pass  # skip all these cases
        elif reduction_hint == ReductionHint.INNER or ReductionHint.DEFAULT:
            return cached_autotune(
                size_hints=size_hints,
                configs=[
                    contiguous_config,
                    # buffersize_config, # TODO: Open autotune
                ],
                meta=meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
    raise NotImplementedError(f"size_hints: {size_hints}")


triton_heuristics.pointwise = tritonxpu_pointwise
triton_heuristics.reduction = tritonxpu_reduction

# ===------------------------------------------------------------------===


# ===-------------------- For XPytorch Inductor -----------------------===
# Base Pytorch(v2.1.0) torch/_inductor/triton_heuristics.py
def grid(*numels):
    """Helper function to compute triton grids"""

    if len(numels) == 1:
        xnumel, ynumel, znumel = numels[0], None, None
    # ===-------------------- For Triton XPU -----------------------===
    # elif len(numels) == 2:
    #     xnumel, ynumel, znumel = numels[1], numels[0], None
    # elif len(numels) == 3:
    #     xnumel, ynumel, znumel = numels[2], numels[1], numels[0]
    # ===-----------------------------------------------------------===
    else:
        raise AssertionError(f"invalid size for numels {len(numels)}")

    def get_grid_dim(numel, block):
        if numel is None:
            return 1
        # return ceildiv(numel, block)
        # ===-------------------- For Triton XPU -----------------------===
        if block is None:
            return numel
        core_num = CLUSTER_NUM * CORE_NUM
        grid_num = ceildiv(numel, block) if numel < core_num else CLUSTER_NUM
        return grid_num
        # ===-----------------------------------------------------------===

    def grid_fn(meta):
        return (
            get_grid_dim(xnumel, meta.get("XBLOCK", 1)),
            get_grid_dim(ynumel, meta.get("YBLOCK", None)),
            get_grid_dim(znumel, meta.get("ZBLOCK", None)),
        )

    return grid_fn


triton_heuristics.grid = grid
# ===------------------------------------------------------------------===
