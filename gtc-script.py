import torch
import triton
import triton.language as tl
import triton.profiler as proton

fill_2d_tma_descriptor = triton.runtime.driver.active.utils.fill_2d_tma_descriptor

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    BM = args["BLOCK_M"]
    BN = args["BLOCK_N"]
    BK = args["BLOCK_K"]
    nw = kernel.num_warps
    ns = kernel.num_stages
    ret["name"] = f"{kernel.name}_{BM}x{BN}x{BK}x{nw}x{ns} [M={M}, N={N}, K={K}]"
    bytes_per_elem = 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret

@triton.jit
def compute_pids(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M):  
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul1(a_desc_ptr, b_desc_ptr, c_desc_ptr, M, N, K,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
  # prologue
  pid = tl.program_id(0)
  pid_m, pid_n = compute_pids(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
  off_am = pid_m * BLOCK_M
  off_bn = pid_n * BLOCK_N
  acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  # main loop
  for off_k in range(0, K, BLOCK_K):
    a = tl._experimental_descriptor_load(a_desc_ptr, [off_am, off_k], [BLOCK_M, BLOCK_K], tl.float16)
    b = tl._experimental_descriptor_load(b_desc_ptr, [off_bn, off_k], [BLOCK_N, BLOCK_K], tl.float16)
    acc = tl.dot(a, b.T, acc)
  # epilogue
  tl._experimental_descriptor_store(c_desc_ptr, acc.to(tl.float16), [off_am, off_bn])

@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul2(a_desc_ptr, b_desc_ptr, c_desc_ptr, M, N, K,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        # prologue
        pid_m, pid_n = compute_pids(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_M)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # main loop
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for off_k in range(0, K, BLOCK_K):
            a = tl._experimental_descriptor_load(a_desc_ptr, [off_m, off_k], [BLOCK_M, BLOCK_K], tl.float16)
            b = tl._experimental_descriptor_load(b_desc_ptr, [off_n, off_k], [BLOCK_N, BLOCK_K], tl.float16)
            accumulator = tl.dot(a, b.T, accumulator)
        # epilogue
        accumulator = accumulator.to(tl.float16)
        tl._experimental_descriptor_store(c_desc_ptr, accumulator, [off_m, off_n])

def run(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b = torch.randn((N, K), dtype=torch.float16, device="cuda")
    c_tri = torch.zeros((M, N), dtype=torch.float16, device="cuda")
    # reference result
    c_ref = torch.matmul(a, b.T)
    # triton result
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 256, 64, 8
    num_warps, num_stages = 8, 3
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # kernel, grid = matmul2, (NUM_SMS,)
    kernel, grid = matmul1, (triton.cdiv(M, BLOCK_M)*triton.cdiv(N, BLOCK_N), )
    a_desc = torch.empty(128, device="cpu", dtype=torch.int8, pin_memory=True)
    b_desc = torch.empty(128, device="cpu", dtype=torch.int8, pin_memory=True)
    c_desc = torch.empty(128, device="cpu", dtype=torch.int8, pin_memory=True)
    fill_2d_tma_descriptor(a.data_ptr(), M, K, BLOCK_M, BLOCK_K, a.element_size(), a_desc.data_ptr())
    fill_2d_tma_descriptor(b.data_ptr(), N, K, BLOCK_N, BLOCK_K, b.element_size(), b_desc.data_ptr())
    fill_2d_tma_descriptor(c_tri.data_ptr(), M, N, BLOCK_M, BLOCK_N, c_tri.element_size(), c_desc.data_ptr())
    torch.cuda.synchronize()
    tri_fn = lambda: kernel[grid](a_desc, b_desc, c_desc, M, N, K, 
                            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_SMS,
                            num_warps=num_warps, num_stages=num_stages)
    ref_fn = lambda: torch.matmul(a, b.T)
    warmup, reps = 50, 200
    tri_ms = triton.testing.do_bench(tri_fn, warmup=warmup, rep=reps)
    ref_ms = triton.testing.do_bench(ref_fn, warmup=warmup, rep=reps)
    diff = (c_tri - c_ref).abs().max()
    tri_tflops = 2*M*N*K / tri_ms * 1e-9
    ref_tflops = 2*M*N*K / ref_ms * 1e-9
    print(f"({M}, {N}, {K}) | TFLOPS: {tri_tflops} (triton), {ref_tflops} (cuBLAS) ; DIFF: {diff}")

for K in range(256, 8192, 128):
    run(8192, 8192, K)
