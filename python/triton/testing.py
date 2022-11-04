import functools
import os
import subprocess
import sys
from contextlib import contextmanager

import torch

import triton._C.libtriton.triton as _triton
from .compiler import OutOfResources

try:
    import triton._C.libtriton.cutlass as _cutlass
    has_cutlass = True
except ImportError:
    _cutlass = None
    has_cutlass = False


def catch_oor(kernel, pytest_handle=None):
    try:
        res = kernel()
    except OutOfResources as e:
        if pytest_handle:
            pytest_handle.skip(str(e))
        return None
    return res


def sparsify_tensor(x, mask, block):
    ret = torch.empty((x.size(0), mask.sum(), block, block), dtype=x.dtype, device=x.device)
    for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
        ret[:, idx, :, :] = x[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block]
    return ret


def make_pair(shape, device="cuda", alpha=1e-2, beta=0., trans=False, data=None):
    if data is None:
        data = torch.randn(shape, dtype=torch.float32, device=device)
    ref_ret = data
    ref_ret = ref_ret * alpha + beta
    ref_ret = ref_ret.half().float()
    if trans:
        ref_ret = ref_ret.t().requires_grad_()
    ref_ret = ref_ret.detach().requires_grad_()
    tri_ret = ref_ret.clone().detach().requires_grad_()
    return ref_ret, tri_ret


def cutlass_matmul(a, b):
    if _cutlass is None:
        raise RuntimeError("Cannot find cutlass library")
    M, N = a.shape[0], b.shape[1]
    Ka, Kb = a.shape[1], b.shape[0]
    assert Ka == Kb
    assert a.dtype == b.dtype
    assert a.device == b.device
    # allocate output
    c = torch.empty_strided((M, N), (1, M), dtype=a.dtype, device=a.device)
    # run function
    dtype = str(a.dtype).split('.')[-1]
    _cutlass.matmul(a.data_ptr(), b.data_ptr(), c.data_ptr(),
                    M, N, Ka,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    dtype, dtype, dtype,
                    a.device.index, torch.cuda.current_stream(a.device).cuda_stream)

    return c


def mask_tensor(x, mask, block, value=0):
    ret = x.clone()
    for h, i, j in zip(*(mask == 0).nonzero(as_tuple=True)):
        ret[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block] = value
    return ret


def assert_almost_equal(x, y, decimal=2, err_msg=''):
    import numpy.testing as npt
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16:
            y = y.float()
        y = y.cpu().detach().numpy()
    npt.assert_array_almost_equal(x, y, err_msg=err_msg, decimal=decimal)


def allclose(x, y, tol=1e-2):
    if x.dtype != y.dtype:
        raise RuntimeError(f'{x.dtype} did not match with {x.dtype}')
    if x.shape != y.shape:
        raise RuntimeError(f'{x.shape} did not match with {y.shape}')
    if x.dtype == torch.bool:
        return torch.sum(x ^ y) == 0
    if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        tol = 0
    diff = abs(x - y)
    x_max = torch.max(x)
    y_max = torch.max(y)
    err = torch.max(diff) / torch.max(x_max, y_max)
    return err <= tol


def nvsmi(attrs):
    attrs = ','.join(attrs)
    cmd = ['nvidia-smi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
    out = subprocess.check_output(cmd)
    ret = out.decode(sys.stdout.encoding).split(',')
    ret = [int(x) for x in ret]
    return ret


def do_bench(fn, warmup=25, rep=100, grad_to_none=None,
             percentiles=(0.5, 0.2, 0.8),
             record_clocks=False, fast_flush=False):
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
    :param percentiles: Performance percentile to return in addition to the median.
    :type percentiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """

    # Estimate the runtime of the function
    fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
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
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    if percentiles:
        percentiles = torch.quantile(times, torch.tensor(percentiles)).tolist()
        return tuple(percentiles)
    else:
        return torch.mean(times).item()


class Benchmark:
    """
    This class is used by the :code:`perf_report` function to generate line plots with a concise API.
    """

    def __init__(
        self,
        x_names,
        x_vals,
        line_arg,
        line_vals,
        line_names,
        plot_name,
        args,
        xlabel='',
        ylabel='',
        x_log=False,
        y_log=False,
        color=None,
        styles=None,
    ):
        """
        Constructor

        :param x_names: Name of the arguments that should appear on the x axis of the plot. If the list contains more than one element, all the arguments are assumed to have the same value.
        :type x_names: List[str]
        :param x_vals: List of values to use for the arguments in :code:`x_names`.
        :type x_vals: List[Any]
        :param line_arg: Argument name for which different values correspond to different lines in the plot.
        :type line_arg: str
        :param line_vals: List of values to use for the arguments in :code:`line_arg`.
        :type line_vals: List[str]
        :param line_names: Label names for the different lines.
        :type line_names: List[str]
        :param plot_name: Name of the plot.
        :type plot_name: str
        :param args: List of arguments to remain fixed throughout the benchmark.
        :type args: List[str]
        :param xlabel: Label for the x axis of the plot.
        :type xlabel: str, optional
        :param ylabel: Label for the y axis of the plot.
        :type ylabel: str, optional
        :param x_log: Whether the x axis should be log scale.
        :type x_log: bool, optional
        :param y_log: Whether the y axis should be log scale.
        :type y_log: bool, optional
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

    def _run(self, bench, save_path, show_plots, print_data):
        import os

        import matplotlib.pyplot as plt
        import pandas as pd
        y_mean = bench.line_names
        y_min = [f'{x}-min' for x in bench.line_names]
        y_max = [f'{x}-max' for x in bench.line_names]
        df = pd.DataFrame(columns=[bench.x_names[0]] + y_mean + y_min + y_max)
        for x in bench.x_vals:
            x_args = {x_name: x for x_name in bench.x_names}
            row_mean, row_min, row_max = [], [], []
            for y in bench.line_vals:
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            df.loc[len(df)] = [x] + row_mean + row_min + row_max
        if bench.plot_name:
            plt.figure()
            ax = plt.subplot()
            x = bench.x_names[0]
            for i, y in enumerate(bench.line_names):
                y_min, y_max = df[y + '-min'], df[y + '-max']
                col = bench.styles[i][0] if bench.styles else None
                sty = bench.styles[i][1] if bench.styles else None
                ax.plot(df[x], df[y], label=y, color=col, ls=sty)
                if y_min is not None and y_max is not None:
                    ax.fill_between(df[x], y_min, y_max, alpha=0.15, color=col)
            ax.legend()
            xlabel = bench.xlabel if bench.xlabel else " = ".join(bench.x_names)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(bench.ylabel)
            # ax.set_title(bench.plot_name)
            ax.set_xscale("log" if bench.x_log else "linear")
            ax.set_yscale("log" if bench.y_log else "linear")
            if show_plots:
                plt.show()
            if save_path:
                plt.savefig(os.path.join(save_path, f"{bench.plot_name}.png"))
        df = df[[bench.x_names[0]] + bench.line_names]
        if print_data:
            print(bench.plot_name + ':')
            print(df)
        if save_path:
            df.to_csv(os.path.join(save_path, f"{bench.plot_name}.csv"), float_format='%.1f', index=False)

    def run(self, show_plots=False, print_data=False, save_path=''):
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        if save_path:
            html = open(os.path.join(save_path, "results.html"), "w")
            html.write("<html><body>\n")
        for bench in benchmarks:
            self._run(bench, save_path, show_plots, print_data)
            if save_path:
                html.write(f"<image src=\"{bench.plot_name}.png\"/>\n")
        if save_path:
            html.write("</body></html>\n")


def perf_report(benchmarks):
    """
    Mark a function for benchmarking. The benchmark can then be executed by using the :code:`.run` method on the return value.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper


def get_dram_gbps(backend=None, device=None):
    ''' return DRAM bandwidth in GB/s '''
    # assert backend == CUDA
    if not backend:
        backend = _triton.runtime.backend.CUDA
    if not device:
        device = torch.cuda.current_device()
    mem_clock_khz = _triton.runtime.memory_clock_rate(backend, device)
    bus_width = _triton.runtime.global_memory_bus_width(backend, device)
    bw_gbps = mem_clock_khz * bus_width * 2 / 1e6 / 8  # In GB/s
    return bw_gbps


def get_max_tensorcore_tflops(dtype: torch.dtype, backend=None, device=None, clock_rate=None):
    if not backend:
        backend = _triton.runtime.backend.CUDA
    if not device:
        device = torch.cuda.current_device()
    num_subcores = _triton.runtime.num_sm(backend, device) * 4  # on recent GPUs
    if not clock_rate:
        clock_rate = _triton.runtime.clock_rate(backend, device)  # in kHz
    cc = _triton.runtime.cc(backend, device)
    if cc < 80:
        assert dtype == torch.float16
        ops_per_sub_core = 256  # 2 4x4x4 Tensor Cores
    else:
        if dtype == torch.float32:
            ops_per_sub_core = 256
        elif dtype in [torch.float16, torch.bfloat16]:
            ops_per_sub_core = 512
        elif dtype == torch.int8:
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


def nvsmi_attr(attrs):
    attrs = ",".join(attrs)
    cmd = [
        "nvidia-smi",
        "-i",
        "0",
        "--query-gpu=" + attrs,
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd)
    ret = out.decode(sys.stdout.encoding).split(",")
    ret = [int(x) for x in ret]
    return ret


@contextmanager
def set_gpu_clock(ref_sm_clock=1350, ref_mem_clock=1215):
    try:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "1"])
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                f"--lock-gpu-clocks={ref_sm_clock},{ref_sm_clock}",
            ]
        )
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                f"--lock-memory-clocks={ref_mem_clock},{ref_mem_clock}",
            ]
        )
        cur_sm_clock = nvsmi_attr(["clocks.current.sm"])[0]
        cur_mem_clock = nvsmi_attr(["clocks.current.memory"])[0]
        assert abs(cur_sm_clock - ref_sm_clock) < 10, f"GPU SMs must run at {ref_sm_clock} MHz"
        assert abs(cur_mem_clock - ref_mem_clock) < 10, f"GPU SMs must run at {ref_mem_clock} MHz"
        tflops = 1e-6 * 2 * 108 * 4 * 256 * ref_sm_clock
        gbps = 640 * 2 * ref_mem_clock * 1e-3
        yield tflops, gbps
    finally:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "0"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rgc"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rmc"])


def get_max_simd_tflops(dtype: torch.dtype, backend=None, device=None):
    if not backend:
        backend = _triton.runtime.backend.CUDA
    if not device:
        device = torch.cuda.current_device()
    num_subcores = _triton.runtime.num_sm(backend, device) * 4  # on recent GPUs
    clock_rate = _triton.runtime.clock_rate(backend, device)  # in kHz
    cc = _triton.runtime.cc(backend, device)
    if cc < 80:
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
