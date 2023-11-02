from __future__ import annotations

import builtins
import time
from typing import Dict, Any, Sequence, Tuple, Callable
from dataclasses import dataclass

from ..testing import do_bench
from .jit import KernelInterface


class OutOfResources(Exception):
    def __init__(self, required, limit, name):
        self.message = (
            f"out of resource: {name}, Required: {required}, Hardware limit: {limit}. "
            + "Reducing block sizes or `num_stages` may help."
        )
        self.required = required
        self.limit = limit
        self.name = name
        super().__init__(self.message)

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.required, self.limit, self.name))


class Autotuner(KernelInterface):
    def __init__(self, fn, arg_names, configs, key, reset_to_zero, prune_configs_by: Dict = None, warmup=25, rep=100):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It takes configs:List[Config] as its input, and returns pruned configs.
        """
        super().__init__(fn.signature, self.run_with_flags, self.warmup_with_flags)
        if not configs:
            self.configs = [Config({}, num_warps=4, num_stages=2, num_ctas=1)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.cache = {}
        # hook to reset all required tensor to zeros before relaunching a kernel
        self.hook = lambda args: 0
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]

            def _hook(args):
                for i in self.reset_idx:
                    args[i].zero_()

            self.hook = _hook
        self.arg_names = arg_names
        # prune configs
        if prune_configs_by:
            perf_model, top_k = prune_configs_by["perf_model"], prune_configs_by["top_k"]
            if "early_config_prune" in prune_configs_by:
                early_config_prune = prune_configs_by["early_config_prune"]
        else:
            perf_model, top_k, early_config_prune = None, None, None
        self.perf_model, self.configs_top_k = perf_model, top_k
        self.early_config_prune = early_config_prune
        self.fn = fn
        self.rep = rep

    def _bench(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.kwargs)
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.hook(args)
            # TODO: Fixme
            self.fn.run(
                *args,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                num_ctas=config.num_ctas,
                enable_warp_specialization=config.enable_warp_specialization,
                # enable_persistent=False,
                **current,
            )

        try:
            return do_bench(kernel_call, warmup=self.warmup, rep=self.rep, quantiles=(0.5, 0.2, 0.8))
        except OutOfResources:
            return [float("inf"), float("inf"), float("inf")]

    def run_with_flags(self, grid: Tuple[int, int, int], flags: dict[str, Any], kernel_args: dict[str, Any]):
        self.nargs = dict(zip(self.arg_names, kernel_args))
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kernel_args}
            _args = []
            for name in self.arg_names:
                if name in all_args:
                    _args.append(all_args[name])
            key = [_args[i] for i in self.key_idx]
            for arg in _args:
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                # prune configs
                pruned_configs = self.prune_configs(kernel_args)
                bench_start = time.time()
                timings = {config: self._bench(*kernel_args, config=config, **kernel_args) for config in pruned_configs}
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.hook(kernel_args)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        full_nargs = {**self.nargs, **kernel_args, **self.best_config.meta}
        if config.pre_hook is not None:
            config.pre_hook(full_nargs)
        ret = self.fn.run_with_flags(
            {**kernel_args, **config.meta},
            {},
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            num_ctas=config.num_ctas,
            enable_warp_specialization=config.enable_warp_specialization,
            **kernel_kwargs,
            **config.kwargs,
        )
        self.nargs = None
        return ret

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.kwargs,
                        num_stages=config.num_stages,
                        num_warps=config.num_warps,
                        num_ctas=config.num_ctas,
                        enable_warp_specialization=config.enable_warp_specialization,
                        enable_persistent=config.enable_persistent,
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup_with_flags(self, flags: Dict[str, Any], args: Dict[str, Any]):
        self.nargs = dict(zip(self.arg_names, args))
        for config in self.prune_configs(args):
            self.fn.warmup_with_flags(
                {**args, **config.meta},
                flags={**flags, **config.flags()},
            )
        self.nargs = None


@dataclass
class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar meta: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :ivar num_ctas: Number of CTAs in a CGA, i.e. number of blocks in a thread-block cluster.
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type enable_warp_specialization: bool
    :ivar enable_warp_specialization: enable specialization (spatial partitioning) or not. See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#spatial-partitioning-also-known-as-warp-specialization
    :ivar enable_fp_fusion: Whether to allow fp fma instructions to be formed from fp add and mul instructions.
    :type enable_fp_fusion: bool
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    """

    # TODO: Check what this looks like in the documentation.
    # TODO(jlebar): Although these are the only flags that are interesting to
    # autotune, there are other compiler flags the user might still want to set,
    # and that doesn't work with the new kernel launch API.
    meta: dict[str, Any]
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = True
    # TODO(jlebar): It appears this may be unused, and it's certainly undocumented.
    enable_persistent: bool = False
    pre_hook: Callable | None = None

    def flags(self):
        return {
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "num_ctas": self.num_ctas,
            "enable_warp_specialization": self.enable_warp_specialization,
            "enable_persistent": self.enable_persistent,
        }


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, warmup=25, rep=100):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(meta={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(meta={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.
    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param warmup: Warmup time (in ms) to pass to benchmarking, defaults to 25.
    :type warmup: int
    :param rep: Repetition time (in ms) to pass to benchmarking, defaults to 100.
    :type rep: int
    """

    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, prune_configs_by, warmup, rep)

    return decorator


# TODO: This does not handle flags separately from args
class Heuristics(KernelInterface):
    def __init__(self, fn, arg_names, values) -> None:
        super().__init__(fn.signature, self.run_with_flags, warmup_with_flags=None)
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run_with_flags(self, grid: Tuple[int, int, int], flags: Dict[str, Any], kernel_args: Dict[str, Any]):
        kernel_args = dict(kernel_args)  # copy args so we don't modify the user's version
        for v, heur in self.values.items():
            kernel_args[v] = heur({**dict(zip(self.arg_names, kernel_args)), **kernel_args})
        return self.fn.run_with_flags(grid, flags, *args)


def heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitevely expensive, or just not applicable.

    .. highlight:: python
    .. code-block:: python

        @triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size
    :param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a list of positional arguments as input.
    :type values: dict[str, Callable[[list[Any]], Any]]
    """

    def decorator(fn):
        return Heuristics(fn, fn.arg_names, values)

    return decorator
