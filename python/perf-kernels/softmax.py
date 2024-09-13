import argparse
import torch
import sys
import pytest

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def get_cuda_autotune_config():
    return [
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'waves_per_eu': 1}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=16, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=16, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=16, num_stages=1),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def softmax_kernel_online(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                          BLOCK_SIZE: tl.constexpr):

    row_start = tl.program_id(0)
    row_idx = row_start

    #loop 1, find max and sum
    m = -float('inf')  #Initial value of max
    row_sum = 0.0
    row_start_ptr = input_ptr + row_idx * input_row_stride
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(input_ptrs, mask=mask, other=-float('inf'), cache_modifier=".cg")  #load block
        m_p = tl.max(row_block, axis=0)  #find block max
        m_p = tl.maximum(m, m_p)  #Find new max across all blocks so far
        row_sum = row_sum * tl.exp(m - m_p)  #Adjust previous sum
        row_sum += tl.sum(tl.exp(row_block - m_p))  #Add to exponentiated sum of this block
        m = m_p  #save max

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    #Loop 2
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(input_ptrs, mask=mask, other=-float('inf'), cache_modifier=".cg")  #load block
        #subtract, exponentiate and divide by sum
        softmax_output = tl.exp(row_block - m) / row_sum
        #store
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    y = torch.empty_like(x)

    num_programs = n_rows

    grid = lambda meta: (num_programs, )
    softmax_kernel_online[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
    )

    return y


def run_softmax(M, N):
    print(f"Running Softmax on shape ({M},{N})")
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y_triton = softmax(x)

    return y_triton


#pytest
@pytest.mark.parametrize('M, N', [(1823, 781), (1, 1), (128, 1), (1, 128), (8192, 8192), (4096, 8192), (359, 1),
                                  (1, 359), (1, 131072), (1, 89999)])
def test_softmax(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


#Benchmark
arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def run_benchmark(args):
    config = []
    if (args.M_benchmark):
        val = args.M_start
        x_vals_list = []
        while val <= args.M_end:
            x_vals_list.append(val)
            val *= args.M_step
        mn_args = {'N': args.N_start}
        plot_name = str("softmax-performance_" + args.dtype + "_N" + str(args.N_start) + "_M" + str(args.M_start) +
                        "-" + str(args.M_end) + "-" + str(args.M_step))
        x_names = ['M']
    else:
        x_vals_list = [i for i in range(args.N_start, args.N_end, args.N_step)]
        mn_args = {'M': args.M_start}
        plot_name = str("softmax-performance_" + args.dtype + "_M" + str(args.M_start) + "_N" + str(args.N_start) +
                        "-" + str(args.N_end) + "-" + str(args.N_step))
        x_names = ['N']
    dtype = arg_to_torch_dtype[args.dtype]

    print(plot_name)
    config.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=[
                "Triton",
                "Torch",
            ],
            styles=[('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name=plot_name,
            args=mn_args,
        ))

    @triton.testing.perf_report(config)
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=dtype)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: softmax(x))
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Softmax",
        allow_abbrev=False,
    )

    parser.add_argument('-M', "--M_start", default="1", type=int)
    parser.add_argument('-Ms', "--M_step", default="2", type=int)
    parser.add_argument('-Me', "--M_end", default="512", type=int)
    parser.add_argument('-Mb', "--M_benchmark", default=False, type=bool)

    parser.add_argument('-N', "--N_start", default="1024", type=int)
    parser.add_argument('-Ns', "--N_step", default="2048", type=int)
    parser.add_argument('-Ne', "--N_end", default="65536", type=int)

    parser.add_argument('-d', "--dtype", default="fp16")
    parser.add_argument('-nb', "--no_benchmark", default=False, type=bool)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_benchmark:
        run_softmax(args.M_start, args.N_start)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
