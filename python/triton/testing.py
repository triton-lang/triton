import torch
import os


def sparsify_tensor(x, mask, block):
    ret = torch.empty((x.size(0), mask.sum(), block, block), dtype=x.dtype, device=x.device)
    for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
        ret[:, idx, :, :] = x[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block]
    return ret


def cutlass_matmul(a, b):
    try:
        import triton._C.libtriton.cutlass as _cutlass
    except:
        return None
    M, N = a.shape[0], b.shape[1]
    c = torch.empty_strided((M, N), (1, M), dtype=a.dtype, device=a.device)
    _cutlass.matmul(a, b, c)
    return c


def mask_tensor(x, mask, block, value=0):
    ret = x.clone()
    for h, i, j in zip(*(mask == 0).nonzero(as_tuple=True)):
        ret[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block] = value
    return ret


def allclose(x, y):
    assert x.dtype == y.dtype
    diff = abs(x - y)
    x_max = torch.max(x)
    y_max = torch.max(y)
    tol = 1e-2
    err = torch.max(diff) / torch.max(x_max, y_max)
    return err < tol


def do_bench(fn, warmup=10, rep=50, grad_to_none=None):
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
    for i in range(warmup + rep):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            grad_to_none.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        if i >= warmup:
            start_event[i - warmup].record()
        fn()
        if i >= warmup:
            end_event[i - warmup].record()
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    q = torch.quantile(times, torch.tensor([0.1, 0.5, 0.9]))
    min_ms = q[0].item()
    mean_ms = q[1].item()
    max_ms = q[2].item()
    return mean_ms, min_ms, max_ms


class Benchmark:
    def __init__(self, x_names, x_vals, y_name, y_vals, y_lines, ylabel, loglog, plot_name, args):
        self.x_names = x_names
        self.x_vals = x_vals
        self.y_name = y_name
        self.y_vals = y_vals
        self.y_lines = y_lines
        self.ylabel = ylabel
        self.loglog = loglog
        self.plot_name = plot_name
        self.args = args


class Mark:
    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    def _run(self, bench, result_path, with_plot):
        import matplotlib.pyplot as plt
        import pandas as pd
        import os
        y_mean = bench.y_lines
        y_min = [f'{x}-min' for x in bench.y_lines]
        y_max = [f'{x}-max' for x in bench.y_lines]
        df = pd.DataFrame(columns=[bench.x_names[0]] + y_mean + y_min + y_max)
        for x in bench.x_vals:
            x_args = {x_name: x for x_name in bench.x_names}
            row_mean, row_min, row_max = [], [], []
            for y in bench.y_vals:
                ret = self.fn(**x_args, **{bench.y_name: y}, **bench.args)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            df.loc[len(df)] = [x] + row_mean + row_min + row_max
        if with_plot and bench.plot_name:
            plt.figure()
            ax = plt.subplot()
            xlabel = " = ".join(bench.x_names)
            x = bench.x_names[0]
            for y in bench.y_lines:
                y_min, y_max = df[y + '-min'], df[y + '-max']
                ax.plot(df[x], df[y], label=y)
                if y_min is not None and y_max is not None:
                    ax.fill_between(df[x], y_min, y_max, alpha=0.5)
            ax.legend()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(bench.ylabel)
            ax.set_title(bench.plot_name)
            ax.set_xscale("log" if bench.loglog else "linear")
            ax.set_yscale("log" if bench.loglog else "linear")
            plt.savefig(os.path.join(result_path, f"{bench.plot_name}.png"))
        df = df[[bench.x_names[0]] + bench.y_lines]
        df.to_csv(os.path.join(result_path, f"{bench.plot_name}.csv"), float_format='%.1f', index=False)

    def run(self, result_path, with_plot):
        with open(os.path.join(result_path, "results.html"), "w") as html:
            html.write("<html><body>\n")
            for bench in self.benchmarks:
                self._run(bench, result_path, with_plot)
                html.write(f"<image src=\"{bench.plot_name}.png\"/>\n")
            html.write("</body></html>\n")


def perf_report(benchmarks):
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper
