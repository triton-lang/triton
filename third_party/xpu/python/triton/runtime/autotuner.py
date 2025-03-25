from __future__ import annotations

import builtins
import os
import time
import inspect
from typing import Dict

from ..testing import do_bench, do_bench_cudagraph
from .jit import KernelInterface
from .errors import OutOfResources


class Autotuner(KernelInterface):

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Dict = None,
        warmup=25,
        rep=100,
        use_cuda_graph=False,
        generate_configs=None,
        op_affiliation="",
        row_sign="",
        col_sign="",
        n_elem_sign="",
    ):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It takes configs:List[Config] as its input, and returns pruned configs.
        """
        self.no_configs = False
        self.generate_configs = generate_configs
        self.op_affiliation = op_affiliation
        self.row_sign = row_sign
        self.col_sign = col_sign
        self.n_elem_sign = n_elem_sign
        if not configs:
            self.no_configs = True
            self.configs = [Config({}, num_warps=4, num_stages=2, num_ctas=1)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.cache = {}
        self.arg_names = arg_names

        # Reset to zero or restore values
        self.reset_idx = []
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]
        self.restore_idx = []
        if restore_value is not None:
            self.restore_idx = [arg_names.index(k) for k in restore_value]

        # Hook to reset or restore for required tensors
        self.pre_hook = lambda args, reset_only=False: 0
        self.post_hook = lambda args, exception: 0
        if pre_hook:
            self.pre_hook = pre_hook
        elif (len(self.reset_idx) > 0 or len(self.restore_idx) > 0):

            def _pre_hook(args, reset_only=False):
                for i in self.reset_idx:
                    args[i].zero_()
                if not reset_only:
                    self.restore_copies = [args[i].clone() for i in self.restore_idx]

            self.pre_hook = _pre_hook

        if post_hook:
            self.post_hook = post_hook
        elif len(self.restore_idx) > 0:

            def _post_hook(args, exception):
                for i, j in enumerate(self.restore_idx):
                    args[j].copy_(self.restore_copies[i])
                self.restore_copies = []

            self.post_hook = _post_hook

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get("early_config_prune", self.early_config_prune)

        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn
        self.num_warmups = warmup
        self.num_reps = rep
        # import torch
        self.use_cuda_graph = False  # use_cuda_graph and torch.cuda.is_available()

    def _bench(self, *args, config, **meta):
        from ..compiler.errors import CompileTimeAssertionFailure

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)
            try:
                self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    self.post_hook(args, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            self.post_hook(args, exception=None)

        try:
            if self.use_cuda_graph:
                import torch
                with torch.cuda.stream(torch.cuda.Stream()):
                    bench_res = do_bench_cudagraph(kernel_call, rep=self.num_reps, return_mode="median")
                return bench_res
            return do_bench(kernel_call, warmup=self.num_warmups, rep=self.num_reps, quantiles=(0.5, 0.2, 0.8))
        except (OutOfResources, CompileTimeAssertionFailure):
            return float("inf") if self.use_cuda_graph else [float("inf"), float("inf"), float("inf")]

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        if self.no_configs and self.generate_configs is not None:
            self.configs = block_size_candidates(self.nargs, self.generate_configs, self.op_affiliation, self.row_sign,
                                                 self.col_sign, self.n_elem_sign)
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
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
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.pre_hook(args, reset_only=True)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(f"Triton autotuning for function {self.base_fn.__name__} finished after "
                  f"{self.bench_time:.2f}s; best config selected: {self.best_config};")
        if config.pre_hook is not None:
            config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for config in self.prune_configs(kwargs):
            ret.append(self.fn.warmup(
                *args,
                **kwargs,
                **config.all_kwargs(),
            ))
        self.nargs = None
        return ret


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_ctas: int
    :ivar num_ctas: number of blocks in a block cluster. SM90+ only.
    :type maxnreg: Optional[int]
    :ivar maxnreg: maximum number of registers one thread can use.  Corresponds
                       to ptx .maxnreg directive.  Not supported on all platforms.
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    """

    def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None, pre_hook=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = num_stages
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook

    def all_kwargs(self):
        return {
            **self.kwargs, **{
                k: v
                for (k, v) in (
                    ("num_warps", self.num_warps),
                    ("num_ctas", self.num_ctas),
                    ("num_stages", self.num_stages),
                    ("maxnreg", self.maxnreg),
                ) if v is not None
            }
        }

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_ctas: {self.num_ctas}")
        res.append(f"num_stages: {self.num_stages}")
        res.append(f"maxnreg: {self.maxnreg}")
        return ", ".join(res)


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=25, rep=100, use_cuda_graph=False, generate_configs=None, op_affiliation="sdnn", row_sign=None,
             col_sign=None, n_elem_sign=None):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
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

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

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
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'args': a list of arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'args': a list of arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: Warmup time (in ms) to pass to benchmarking, defaults to 25.
    :type warmup: int
    :param rep: Repetition time (in ms) to pass to benchmarking, defaults to 100.
    :type rep: int
    """

    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, generate_configs=generate_configs,
                         op_affiliation=op_affiliation, row_sign=row_sign, col_sign=col_sign, n_elem_sign=n_elem_sign)

    return decorator


class Heuristics(KernelInterface):

    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


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


def largest_factor(x: int):
    ret = 1
    for i in range(x - 1, 1, -1):
        if x % i == 0:
            ret = i
            break
    return ret


def cdiv(x: int, y: int):
    return (x + y - 1) // y


def floordiv(x: int, y: int):
    return x // y


def aligned(x: int, y: int):
    return cdiv(x, y) * y


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def find_next_multiple_of_12(n):
    """Return the next multiple of 12 greater than n"""
    if n <= 0:
        return 12

    remainder = n % 12

    if remainder == 0:
        return n
    else:
        return n + (12 - remainder)


def append_candidate(candicates: list, target_candicate: Config):
    found = False
    for item in candicates:
        if item.all_kwargs() == target_candicate.all_kwargs():
            found = True
            break
    if not found:
        candicates.append(target_candicate)
    return


def check_out_of_mem(block_size_m, block_size_n, block_size_k, mem, ele_bytes, bias, buffer_num, a_trans, b_trans):
    am_layout = block_size_m
    ak_layout = block_size_k
    bn_layout = block_size_n
    bk_layout = block_size_k

    if a_trans and b_trans:
        am_layout, ak_layout = block_size_k, block_size_m
        bn_layout, bk_layout = block_size_k, block_size_n
    elif a_trans:
        am_layout, ak_layout = block_size_k, block_size_m
    elif b_trans:
        bn_layout, bk_layout = block_size_k, block_size_n

    return ((aligned(ak_layout * ele_bytes, mem[1]) * am_layout + aligned(bn_layout * ele_bytes, mem[1]) * bk_layout)
            > (mem[0] - aligned(block_size_n * ele_bytes, mem[1]) * (block_size_m + bias *
                                                                     (2 + block_size_m))) // buffer_num)


def add_candidate_for_workload_not_balanced(configs: list, block_size_m, block_size_n, block_size_k, buffer_num,
                                            meta_info):
    input_size = meta_info['input_size']
    mem = meta_info['mem']
    ele_bytes = meta_info['ele_bytes']
    bias = meta_info['bias']
    block_names = meta_info['block_names']
    grid_aligned = meta_info['grid_aligned']
    aligned_size = meta_info['aligned_size']
    a_trans = meta_info['a_trans']
    b_trans = meta_info['b_trans']

    grid_m_aligned = cdiv(input_size[0], block_size_m)
    grid_n_aligned = cdiv(input_size[1], block_size_n)

    top_p = 3

    while check_out_of_mem(block_size_m, block_size_n, block_size_k, mem, ele_bytes, bias, buffer_num, a_trans,
                           b_trans):
        if block_size_k % 2 == 0:
            block_size_k = block_size_k // 2
        else:
            block_size_k = largest_factor(block_size_k)

        if block_size_k == 1:
            break

    if (grid_m_aligned * grid_n_aligned) < grid_aligned:
        block_size_m = max(2, min(block_size_m, input_size[0]))
        block_size_n = max(2, min(block_size_n, input_size[1]))

        tmp_grid_m = cdiv(input_size[0], block_size_m)
        tmp_grid_n = cdiv(input_size[1], block_size_n)

        append_candidate(
            configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n, block_names[2]: block_size_k}))
        for i in range(2, 13):
            tmp_block_size_m = block_size_m // i
            if tmp_block_size_m < 2:
                break
            tmp_grid_m = cdiv(input_size[0], tmp_block_size_m)

            if (tmp_grid_m * tmp_grid_n) % grid_aligned == 0:
                append_candidate(
                    configs,
                    Config(
                        {block_names[0]: tmp_block_size_m, block_names[1]: block_size_n, block_names[2]: block_size_k}))

        for i in range(2, 13):
            tmp_block_size_n = block_size_n // i
            if tmp_block_size_n < 2:
                break
            tmp_grid_n = cdiv(input_size[1], tmp_block_size_n)

            if (tmp_grid_m * tmp_grid_n) % grid_aligned == 0:
                append_candidate(
                    configs,
                    Config(
                        {block_names[0]: block_size_m, block_names[1]: tmp_block_size_n, block_names[2]: block_size_k}))
    else:
        append_candidate(
            configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n, block_names[2]: block_size_k}))

        if input_size[0] % block_size_m != 0:
            for i in range(block_size_m, 1, -1):
                _block_size_m = i
                if (cdiv(input_size[0], _block_size_m) * grid_n_aligned) % grid_aligned == 0:
                    append_candidate(
                        configs,
                        Config(
                            {block_names[0]: _block_size_m, block_names[1]: block_size_n, block_names[2]:
                             block_size_k}))
                    break

        elif input_size[1] % block_size_n != 0:
            for i in range(block_size_n, 1, -1):
                _block_size_n = i
                if (cdiv(input_size[1], _block_size_n) * grid_m_aligned) % grid_aligned == 0:
                    append_candidate(
                        configs,
                        Config(
                            {block_names[0]: block_size_m, block_names[1]: _block_size_n, block_names[2]:
                             block_size_k}))
                    break
        else:
            for i in range(block_size_m, (grid_m_aligned - 1) * aligned_size["m_aligned"] + 1, -1):
                _block_size_m = i
                for j in range(block_size_n, (grid_n_aligned - 1) * aligned_size["n_aligned"] + 1, -1):
                    _block_size_n = j
                    tmp_grid_m = cdiv(input_size[0], _block_size_m)
                    tmp_grid_n = cdiv(input_size[1], _block_size_n)

                    if (tmp_grid_m * tmp_grid_n) % grid_aligned == 0:
                        top_p -= 1
                        append_candidate(
                            configs,
                            Config({
                                block_names[0]: _block_size_m, block_names[1]: _block_size_n, block_names[2]:
                                block_size_k
                            }))
                        break

                    if top_p == 0:
                        break

    return


def add_candidate_for_workload_balanced(configs: list, block_size_m, block_size_n, block_size_k, buffer_num, meta_info):
    input_size = meta_info['input_size']
    mem = meta_info['mem']
    ele_bytes = meta_info['ele_bytes']
    bias = meta_info['bias']
    block_names = meta_info['block_names']
    aligned_size = meta_info['aligned_size']
    a_trans = meta_info['a_trans']
    b_trans = meta_info['b_trans']

    grid_m_aligned = cdiv(input_size[0], block_size_m)
    grid_n_aligned = cdiv(input_size[1], block_size_n)

    while check_out_of_mem(block_size_m, block_size_n, block_size_k, mem, ele_bytes, bias, buffer_num, a_trans,
                           b_trans):
        if block_size_k % 2 == 0:
            block_size_k = block_size_k // 2
        else:
            block_size_k = largest_factor(block_size_k)

        if block_size_k == 1:
            break

    append_candidate(configs,
                     Config({block_names[0]: block_size_m, block_names[1]: block_size_n, block_names[2]: block_size_k}))

    if input_size[0] % grid_m_aligned == 0 and input_size[1] % grid_n_aligned == 0:
        block_size_m = max(2, floordiv(input_size[0], grid_m_aligned))
        block_size_n = max(2, floordiv(input_size[1], grid_n_aligned))
    elif input_size[0] % grid_m_aligned == 0:
        block_size_m = max(2, floordiv(input_size[0], grid_m_aligned))
    elif input_size[1] % grid_n_aligned == 0:
        block_size_n = max(2, floordiv(input_size[1], grid_n_aligned))

    append_candidate(configs,
                     Config({block_names[0]: block_size_m, block_names[1]: block_size_n, block_names[2]: block_size_k}))
    return


def get_input_ele_bytes(args):
    ele_bytes = 4

    if "a_ptr" in args.keys():
        A = args["a_ptr"]
    elif "inp" in args.keys():
        A = args["inp"]
    else:
        A = args["A"]

    if A.dtype.__str__() == "torch.float16":
        ele_bytes = 2

    return ele_bytes


def balance_grid(block_size_m, block_size_n, input_size):
    grid_x = cdiv(input_size[0], block_size_m)
    grid_y = cdiv(input_size[1], block_size_n)

    total_grid = grid_x * grid_y

    # simple balance method
    next_multiple_of_12 = find_next_multiple_of_12(total_grid)
    grid_y = cdiv(next_multiple_of_12, grid_x)
    block_size_n = cdiv(input_size[1], grid_y)

    # todo: add more balance method

    return block_size_m, block_size_n


def block_size_candidates_cluster(args, generate_configs, op_affiliation, row_sign, col_sign, n_elem_sign):
    # The result of block_size_candidates
    configs = []

    # 1D Tune
    if "BLOCK_SIZE" in args.keys():  # TODO: add more 1d block_size str to match
        if n_elem_sign == None:
            raise RuntimeError("Failed to tune block size. Miss n_elem_sign")
        n_elements = args[n_elem_sign]

        # max cluster
        block_size = cdiv(n_elements, 12)
        append_candidate(configs, Config({"BLOCK_SIZE": block_size}))

        # max cluster with power2 block_size
        block_size = next_power_of_2(cdiv(n_elements, 12))
        append_candidate(configs, Config({"BLOCK_SIZE": block_size}))

        # Print the result of block_size_candidates
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
            print(f"row: {m}, col: {n}")
            for config in configs:
                print(f"config: {config}")

        return configs

    # 2D Tune
    # collect all useful info
    ele_bytes = get_input_ele_bytes(args)

    grid_aligned = 12

    BLOCK_M = "BLOCK_M"  # TODO: add more 2d block_size m/n str to match
    BLOCK_N = "BLOCK_N"

    block_names = (BLOCK_M, BLOCK_N)

    if row_sign == None or col_sign == None:
        raise RuntimeError("Failed to tune block_m/block_n size. Miss row_sign/col_sign")

    m = args[row_sign]
    n = args[col_sign]

    input_size = (
        m,
        n,
    )

    mem = (8192, 64)  # 8K * 64 cores LM

    aligned_size = {
        "m_aligned": 64,
        "n_aligned": 64,
    }

    meta_info = {
        "ele_bytes": ele_bytes,
        "grid_aligned": grid_aligned,
        "block_names": block_names,
        "input_size": input_size,
        "aligned_size": aligned_size,
        "mem": mem,
    }

    core_num = 64
    buffer_size_upper = 512  # TODO: set to 2048 bytes
    if "buffer_size" in args.keys():
        buffer_size_upper = args["buffer_size"]

    buffer_size_elem_cnt = cdiv(buffer_size_upper, ele_bytes)

    experimental_fine_tune = bool(os.getenv("TRITON_FINE_AUTOTUNE", False))

    # Start To Tune
    block_size_m = input_size[0]
    block_size_n = input_size[1]

    if buffer_size_elem_cnt != next_power_of_2(buffer_size_elem_cnt):
        raise RuntimeError("buffer_size should be power of two")

    # buffer can cache all input_data
    if buffer_size_elem_cnt * core_num >= block_size_n:
        # naive config
        block_size_m = next_power_of_2(cdiv(input_size[0], 12))
        block_size_n = input_size[1]
        append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

        # balance config
        block_size_m = cdiv(input_size[0], 12)
        block_size_n = input_size[1]
        append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

        if experimental_fine_tune:
            # naive config
            block_size_m = next_power_of_2(cdiv(input_size[0], 12))
            block_size_n = next_power_of_2(input_size[1])
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

            # naive config
            block_size_m = cdiv(input_size[0], 12)
            block_size_n = next_power_of_2(input_size[1])
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

        return configs

    # buffer cannot cache all input_data
    block_size_m = input_size[0]
    block_size_n = input_size[1]

    # naive config
    block_size_m = next_power_of_2(cdiv(input_size[0], 12))
    block_size_n = buffer_size_elem_cnt * core_num
    append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

    # TODO: add block_size_m auto tune
    # only support logic: gridX = cdiv(M, BLOCK_M)、gridY = cdiv(N, BLOCK_N)
    for block_size_n in range(buffer_size_elem_cnt * core_num, 0, -aligned_size["n_aligned"]):
        if len(configs) == 5:
            break
        grid_x = cdiv(input_size[0], block_size_m)
        grid_y = cdiv(input_size[1], block_size_n)
        total_grid = grid_x * grid_y
        if total_grid % grid_aligned != 0:
            (block_size_m, block_size_n) = balance_grid(block_size_m, block_size_n, input_size)
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))
        else:
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

    # balance config
    block_size_m = cdiv(input_size[0], 12)
    block_size_n = buffer_size_elem_cnt * core_num
    append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

    # TODO: add block_size_m auto tune
    # only support logic: gridX = cdiv(M, BLOCK_M)、gridY = cdiv(N, BLOCK_N)
    for block_size_n in range(buffer_size_elem_cnt * core_num, 0, -aligned_size["n_aligned"]):
        if len(configs) == 5:
            break
        grid_x = cdiv(input_size[0], block_size_m)
        grid_y = cdiv(input_size[1], block_size_n)
        total_grid = grid_x * grid_y
        if total_grid % grid_aligned != 0:
            (block_size_m, block_size_n) = balance_grid(block_size_m, block_size_n, input_size)
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))
        else:
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

    # Print the result of block_size_candidates
    if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
        print(f"row: {m}, col: {n}")
        for config in configs:
            print(f"config: {config}")

    return configs


def block_size_candidates(args, generate_configs, op_affiliation, row_sign, col_sign, n_elem_sign):
    if op_affiliation == "cluster":
        return block_size_candidates_cluster(args, generate_configs, op_affiliation, row_sign, col_sign, n_elem_sign)

    # Get compile time info
    BLOCK_M = "BLOCK_M"
    BLOCK_N = "BLOCK_N"
    BLOCK_K = "BLOCK_K"
    bias = 0

    if generate_configs == "bmm":
        BLOCK_M = "TILE_M"
        BLOCK_N = "TILE_N"
        BLOCK_K = "TILE_K"
    elif generate_configs == "addmm":
        BLOCK_M = "BLOCK_SIZE_M"
        BLOCK_N = "BLOCK_SIZE_N"
        BLOCK_K = "BLOCK_SIZE_K"
        bias = 1

    # Block names
    block_names = (BLOCK_M, BLOCK_N, BLOCK_K)

    ele_bytes = 4
    if "a_ptr" in args.keys():
        A = args["a_ptr"]
    else:
        A = args["A"]
    if A.dtype.__str__() == "torch.float16":
        ele_bytes = 2

    a_trans = False
    b_trans = False

    if "stride_ak" in args.keys():
        a_trans = args["stride_ak"] != 1
    if "stride_bn" in args.keys():
        b_trans = args["stride_bn"] != 1

    # Input size info
    input_size = (args["M"], args["N"], args["K"])

    mem = (1605632, 128)
    aligned_size = {
        "m_aligned": 80,
        "n_aligned": 64,
        "k_aligned": 128,
    }

    grid_aligned = 12

    meta_info = {
        "ele_bytes": ele_bytes,
        "bias": bias,
        "grid_aligned": grid_aligned,
        "block_names": block_names,
        "input_size": input_size,
        "aligned_size": aligned_size,
        "mem": mem,
        "a_trans": a_trans,
        "b_trans": b_trans,
    }

    max_m_aglined = 4
    max_n_aglined = 7

    # The result of block_size_candidates
    configs = []

    buffer_nums = [2]
    for buffer_num in buffer_nums:
        block_size_m = input_size[0]
        block_size_n = input_size[1]
        block_size_k = input_size[2]

        if block_size_m < 2:
            block_size_m = 2

        if block_size_n < 2:
            block_size_n = 2
        n_loop_num = 2
        for i in range(min(max_m_aglined, cdiv(input_size[0], aligned_size["m_aligned"])), 0, -1):
            if n_loop_num == 0:
                break
            n_loop_num -= 1
            for j in range(min(max_n_aglined, cdiv(input_size[1], aligned_size["n_aligned"])), 0, -1):
                tmp_block_size_m = i * aligned_size["m_aligned"]
                tmp_block_size_n = j * aligned_size["n_aligned"]
                grid_m_aligned = cdiv(input_size[0], tmp_block_size_m)
                grid_n_aligned = cdiv(input_size[1], tmp_block_size_n)
                total_grid = grid_m_aligned * grid_n_aligned
                if total_grid % grid_aligned != 0:
                    add_candidate_for_workload_not_balanced(configs, tmp_block_size_m, tmp_block_size_n, block_size_k,
                                                            buffer_num, meta_info)
                else:
                    add_candidate_for_workload_balanced(configs, tmp_block_size_m, tmp_block_size_n, block_size_k,
                                                        buffer_num, meta_info)

    # Print the result of block_size_candidates
    if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
        print(f"M: {input_size[0]}, N: {input_size[1]}, K: {input_size[2]}")
        for config in configs:
            print(f"config: {config}")

    return configs
