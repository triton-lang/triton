import functools
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl
from . import runtime


def nvsmi(attrs):
    attrs = ','.join(attrs)
    cmd = ['nvidia-smi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
    out = subprocess.check_output(cmd)
    ret = out.decode(sys.stdout.encoding).split(',')
    ret = [int(x) for x in ret]
    return ret


def _summarize_statistics(times, quantiles, return_mode):
    import torch
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(torch, return_mode)(times).item()


def do_bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function.

    :param fn: Function to benchmark
    :type fn: Callable
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all" Default is "mean".
    :type return_mode: str
    """
    import torch
    assert return_mode in ["min", "max", "mean", "median", "all"]

    with torch.cuda.stream(torch.cuda.Stream()):
        # warmup
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None
        # step 1 - we estimate the amount of time the kernel call takes
        # NOTE: this estimate isn't super accurate because the GPU isn't warmed up at this point
        #       but it is probably good enough
        # NOTE: we don't use a graph to estimate the runtime because creating a graph is expensive,
        #       ~300ms on A100, so we default to the same method used in `do_bench` (minus the L2
        #       cache flush).
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        n_repeat = max(1, int(rep / estimate_ms))
        # step 2 - construct a cuda graph with `n_repeat` unrolled function calls to minimize
        # host overhead
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                fn()
        torch.cuda.synchronize()
        # measure time and return
        ret = []
        n_retries = 10
        for _ in range(n_retries):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            ret += [start_event.elapsed_time(end_event) / n_repeat]
        return _summarize_statistics(torch.tensor(ret), quantiles, return_mode)


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all" Default is "mean".    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]
    import torch

    enable_bench_npu = os.getenv("TRITON_BENCH_METHOD", 'default').lower() in ('npu')
    if torch.npu.is_available() and enable_bench_npu:
        return do_bench_npu(fn, warmup=max(5, warmup), active=max(30, rep))

    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    return _summarize_statistics(times, quantiles, return_mode)

def collect_files(base_dir):
    import pandas as pd
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file != 'op_statistic.csv':
                continue
            target_file = os.path.join(root, file)
            df = pd.read_csv(target_file)
            triton_rows = df[df['OP Type'].str.startswith('triton', na=False)]
            if not triton_rows.empty:
                return triton_rows['Avg Time(us)'].values[0]
            return float('inf')
    return float('inf')

def do_bench_npu(fn, warmup=5, active=30):
    import torch
    import torch_npu
    import hashlib
    from datetime import datetime

    stream = torch.npu.current_stream()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    skip_first = 1
    wait = 0
    repeat = 1
    total = skip_first + (wait + warmup + active) * repeat
    md5_hash = hashlib.md5(datetime.now().strftime('%Y-%m-%d').encode('utf-8')).hexdigest()
    torch_path="./profile_result/"+md5_hash
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU
            ],
        schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config) as prof:
        stream.synchronize()

        for i in range(total):
            fn()
            prof.step()
        stream.synchronize()

    time = collect_files(torch_path)

    import shutil
    import os
    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)
    # TODO: use logging
    # print("avg time = ", time, type(time))
    return time

def assert_close(x, y, atol=None, rtol=None, err_msg=''):
    """
    Asserts that two inputs are close within a certain tolerance.

    :param x: The first input.
    :type x: scala, list, numpy.ndarray, or torch.Tensor
    :param y: The second input.
    :type y: scala, list, numpy.ndarray, or torch.Tensor
    :param atol: The absolute tolerance. Default value is 1e-2.
    :type atol: float, optional
    :param rtol: The relative tolerance. Default value is 0.
    :type rtol: float, optional
    :param err_msg: The error message to use if the assertion fails.
    :type err_msg: str
    """
    import numpy as np
    import torch

    # canonicalize arguments to be tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    # absolute tolerance
    if atol is None:
        atol = 1e-2
    atol = atol(x.dtype) if callable(atol) else atol
    # relative tolerance hook
    if rtol is None:
        rtol = 0.
    rtol = rtol(x.dtype) if callable(rtol) else rtol
    # we use numpy instead of pytorch
    # as it seems more memory efficient
    # pytorch tends to oom on large tensors
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16:
            y = y.float()
        y = y.cpu().detach().numpy()
    # we handle size==1 case separately as we can
    # provide better error message there
    if x.size > 1 or y.size > 1:
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
        return
    if not np.allclose(x, y, atol=atol, rtol=rtol):
        raise AssertionError(f'{err_msg} {x} is not close to {y} (atol={atol}, rtol={rtol})')


class Benchmark:
    """
    This class is used by the :code:`perf_report` function to generate line plots with a concise API.
    """

    def __init__(
        self,
        x_names: List[str],
        x_vals: List[Any],
        line_arg: str,
        line_vals: List[Any],
        line_names: List[str],
        plot_name: str,
        args: Dict[str, Any],
        xlabel: str = '',
        ylabel: str = '',
        x_log: bool = False,
        y_log: bool = False,
        styles=None,
    ):
        """
        Constructor.
        x_vals can be a list of scalars or a list of tuples/lists. If x_vals is a list
        of scalars and there are multiple x_names, all arguments will have the same value.
        If x_vals is a list of tuples/lists, each element should have the same length as
        x_names.

        :param x_names: Name of the arguments that should appear on the x axis of the plot.
        :type x_names: List[str]
        :param x_vals: List of values to use for the arguments in :code:`x_names`.
        :type x_vals: List[Any]
        :param line_arg: Argument name for which different values correspond to different lines in the plot.
        :type line_arg: str
        :param line_vals: List of values to use for the arguments in :code:`line_arg`.
        :type line_vals: List[Any]
        :param line_names: Label names for the different lines.
        :type line_names: List[str]
        :param plot_name: Name of the plot.
        :type plot_name: str
        :param args: Dictionary of keyword arguments to remain fixed throughout the benchmark.
        :type args: Dict[str, Any]
        :param xlabel: Label for the x axis of the plot.
        :type xlabel: str, optional
        :param ylabel: Label for the y axis of the plot.
        :type ylabel: str, optional
        :param x_log: Whether the x axis should be log scale.
        :type x_log: bool, optional
        :param y_log: Whether the y axis should be log scale.
        :type y_log: bool, optional
        :param styles: A list of tuples, where each tuple contains two elements: a color and a linestyle.
        :type styles: list[tuple[str, str]]
        """
        self.x_names = x_names
        self.x_vals = x_vals
        self.x_log = x_log
        self.line_arg = line_arg
        self.line_vals = line_vals
        self.line_names = line_names
        self.y_log = y_log
        self.styles = styles
        # plot info
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_name = plot_name
        self.args = args


class Mark:

    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    def _run(self, bench: Benchmark, save_path: str, show_plots: bool, print_data: bool, diff_col=False,
             save_precision=6, **kwrags):
        import os

        import matplotlib.pyplot as plt
        import pandas as pd
        y_mean = bench.line_names
        y_min = [f'{x}-min' for x in bench.line_names]
        y_max = [f'{x}-max' for x in bench.line_names]
        x_names = list(bench.x_names)
        df = pd.DataFrame(columns=x_names + y_mean + y_min + y_max)
        for x in bench.x_vals:
            # x can be a single value or a sequence of values.
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]

            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_mean, row_min, row_max = [], [], []
            for y in bench.line_vals:
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwrags)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            df.loc[len(df)] = list(x) + row_mean + row_min + row_max

        if bench.plot_name:
            plt.figure()
            ax = plt.subplot()
            # Plot first x value on x axis if there are multiple.
            first_x = x_names[0]
            for i, y in enumerate(bench.line_names):
                y_min, y_max = df[y + '-min'], df[y + '-max']
                col = bench.styles[i][0] if bench.styles else None
                sty = bench.styles[i][1] if bench.styles else None
                ax.plot(df[first_x], df[y], label=y, color=col, ls=sty)
                if not y_min.isnull().all() and not y_max.isnull().all():
                    y_min = y_min.astype(float)
                    y_max = y_max.astype(float)
                    ax.fill_between(df[first_x], y_min, y_max, alpha=0.15, color=col)
            ax.legend()
            ax.set_xlabel(bench.xlabel or first_x)
            ax.set_ylabel(bench.ylabel)
            # ax.set_title(bench.plot_name)
            ax.set_xscale("log" if bench.x_log else "linear")
            ax.set_yscale("log" if bench.y_log else "linear")
            if show_plots:
                plt.show()
            if save_path:
                plt.savefig(os.path.join(save_path, f"{bench.plot_name}.png"))
        df = df[x_names + bench.line_names]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df['Diff'] = df[col1] - df[col0]

        if print_data:
            print(bench.plot_name + ':')
            print(df.to_string())
        if save_path:
            df.to_csv(os.path.join(save_path, f"{bench.plot_name}.csv"), float_format=f"%.{save_precision}f",
                      index=False)
        return df

    def run(self, show_plots=False, print_data=False, save_path='', return_df=False, **kwargs):
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            html = open(os.path.join(save_path, "results.html"), "w")
            html.write("<html><body>\n")
        for bench in benchmarks:
            result_dfs.append(self._run(bench, save_path, show_plots, print_data, **kwargs))
            if save_path:
                html.write(f"<image src=\"{bench.plot_name}.png\"/>\n")
        if save_path:
            html.write("</body></html>\n")
            html.close()
        if return_df:
            if has_single_bench:
                return result_dfs[0]
            else:
                return result_dfs
        return None


def perf_report(benchmarks):
    """
    Mark a function for benchmarking. The benchmark can then be executed by using the :code:`.run` method on the return value.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper


def get_dram_gbps(device=None):
    ''' return DRAM bandwidth in GB/s '''
    import torch

    from .runtime import driver
    if not device:
        device = torch.cuda.current_device()
    mem_clock_khz = driver.active.utils.get_device_properties(device)["mem_clock_rate"]  # in kHz
    bus_width = driver.active.utils.get_device_properties(device)["mem_bus_width"]
    bw_gbps = mem_clock_khz * bus_width * 2 / 1e6 / 8  # In GB/s
    return bw_gbps


def get_max_tensorcore_tflops(dtype, clock_rate, device=None):
    import torch

    from .runtime import driver
    if not device:
        device = torch.cuda.current_device()

    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        assert dtype == torch.float16
        ops_per_sub_core = 256  # 2 4x4x4 Tensor Cores
    else:
        if dtype in [torch.float32, torch.int32]:
            ops_per_sub_core = 256
        elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
            ops_per_sub_core = 512
        elif dtype in [torch.int8, tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
            ops_per_sub_core = 1024
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops


# create decorator that wraps test function into
# a cuda-memcheck system call


def cuda_memcheck(**target_kwargs):

    def decorator(test_fn):

        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            import psutil
            ppid_name = psutil.Process(os.getppid()).name()
            run_cuda_memcheck = target_kwargs.items() <= kwargs.items()
            if run_cuda_memcheck and ppid_name != "cuda-memcheck":
                path = os.path.realpath(test_fn.__globals__["__file__"])
                # get path of current file
                env = {"PATH": os.environ["PATH"], "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
                assert 'request' in kwargs, "memcheck'ed test must have a (possibly unused) `request` fixture"
                test_id = kwargs['request'].node.callspec.id
                cmd = f"{path}::{test_fn.__name__}[{test_id}]"
                out = subprocess.run(["cuda-memcheck", "pytest", "-vs", cmd], capture_output=True, env=env)
                assert out.returncode == 0, "cuda-memcheck returned an error: bounds checking failed"
                assert "ERROR SUMMARY: 0 errors" in str(out.stdout)
            else:
                test_fn(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def set_gpu_clock(ref_sm_clock=1350, ref_mem_clock=1215):
    try:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "1"])
        subprocess.check_output([
            "nvidia-smi",
            "-i",
            "0",
            f"--lock-gpu-clocks={ref_sm_clock},{ref_sm_clock}",
        ])
        subprocess.check_output([
            "nvidia-smi",
            "-i",
            "0",
            f"--lock-memory-clocks={ref_mem_clock},{ref_mem_clock}",
        ])
        cur_sm_clock = nvsmi(["clocks.current.sm"])[0]
        cur_mem_clock = nvsmi(["clocks.current.memory"])[0]
        assert abs(cur_sm_clock - ref_sm_clock) < 10, f"GPU SMs must run at {ref_sm_clock} MHz"
        assert abs(cur_mem_clock - ref_mem_clock) < 10, f"GPU SMs must run at {ref_mem_clock} MHz"
        tflops = 1e-6 * 2 * 108 * 4 * 256 * ref_sm_clock
        gbps = 640 * 2 * ref_mem_clock * 1e-3
        yield tflops, gbps
    finally:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "0"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rgc"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rmc"])


def get_max_simd_tflops(dtype, clock_rate, device=None):
    import torch

    from .runtime import driver
    if not device:
        device = torch.cuda.current_device()

    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        if dtype == torch.float32:
            ops_per_sub_core = 32  # 2*16
        elif dtype == torch.float16:
            ops_per_sub_core = 64
        else:
            raise RuntimeError("dtype not supported")
    else:
        if dtype == torch.float32:
            ops_per_sub_core = 32
        elif dtype in [torch.float16, torch.bfloat16]:
            ops_per_sub_core = 64
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops

# # Patch the triton language API here because triton's __init__.py
# # import testing in the last stages.

# from .triton_patch.language.core import dot, gather, insert, subview
# from .triton_patch.language.standard import flip, sigmoid, softmax
# from .triton_patch.language.math import (
#     umulhi,
#     exp,
#     exp2,
#     log,
#     log2,
#     cos,
#     sin,
#     sqrt,
#     sqrt_rn,
#     rsqrt,
#     div_rn,
#     erf,
#     tanh,
#     floor,
#     ceil,
# )
# from .triton_patch.language.semantic import (
#     arange,
#     floordiv,
#     atom_red_typechecking_impl,
#     atomic_max,
#     atomic_min,
#     _load_legacy,
#     maximum,
#     minimum,
# )
# from . import language

# language.dot = dot
# language.flip = flip
# language.sigmoid = sigmoid
# language.softmax = softmax
# language.gather = gather
# language.insert = insert
# language.subview = subview

# # from .triton_patch.language.core import dtype, pointer_type, block_type, function_type
# # language.core.dtype = dtype
# # language.core.pointer_type = pointer_type
# # language.core.block_type = block_type
# # language.core.function_type = function_type
# language.semantic.arange = arange
# language.semantic.floordiv = floordiv
# language.semantic.atom_red_typechecking_impl = atom_red_typechecking_impl
# language.semantic.atomic_max = atomic_max
# language.semantic.atomic_min = atomic_min
# language.semantic._load_legacy = _load_legacy
# language.semantic.maximum = maximum
# language.semantic.minimum = minimum

# language.umulhi = umulhi
# language.exp = exp
# language.exp2 = exp2
# language.log = log
# language.log2 = log2
# language.cos = cos
# language.sin = sin
# language.sqrt = sqrt
# language.sqrt_rn = sqrt_rn
# language.rsqrt = rsqrt
# language.div_rn = div_rn
# language.erf = erf
# language.tanh = tanh
# language.floor = floor
# language.ceil = ceil
# language.math.umulhi = umulhi
# language.math.exp = exp
# language.math.exp2 = exp2
# language.math.log = log
# language.math.log2 = log2
# language.math.cos = cos
# language.math.sin = sin
# language.math.sqrt = sqrt
# language.math.sqrt_rn = sqrt_rn
# language.math.rsqrt = rsqrt
# language.math.div_rn = div_rn
# language.math.erf = erf
# language.math.tanh = tanh
# language.math.floor = floor
# language.math.ceil = ceil
# language.math.isnan = language.extra.ascend.libdevice.isnan
# language.math.isinf = language.extra.ascend.libdevice.isinf
