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
def matmul1(a_ptr, b_ptr, c_ptr, M, N, K,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
  # prologue
  pid = tl.program_id(0)
  pid_m, pid_n = compute_pids(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
  off_m = pid_m * BLOCK_M
  off_n = pid_n * BLOCK_N
  a_desc = tl._experimental_make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
  b_desc = tl._experimental_make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K])
  c_desc = tl._experimental_make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
  acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  # main loop
  for off_k in range(0, K, BLOCK_K):
    a = a_desc.load([off_m, off_k])
    b = b_desc.load([off_n, off_k])
    acc = tl.dot(a, b.T, acc)
  # epilogue
  c_desc.store([off_m, off_n], acc.to(tl.float16))

@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul2(a_ptr, b_ptr, c_ptr, M, N, K,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    a_desc = tl._experimental_make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl._experimental_make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K])
    c_desc = tl._experimental_make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        # prologue
        pid_m, pid_n = compute_pids(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_M)
        off_m, off_n = (pid_m * BLOCK_M, pid_n * BLOCK_N)
        # main loop
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for off_k in range(0, K, BLOCK_K):
            a = a_desc.load([off_m, off_k])
            b = b_desc.load([off_n, off_k])
            acc = tl.dot(a, b.T, acc)
        # epilogue
        c_desc.store([off_m, off_n], acc.to(tl.float16))

@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul3(a_ptr, b_ptr, c_ptr, M, N, K,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    a_desc = tl._experimental_make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl._experimental_make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K])
    c_desc = tl._experimental_make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N//2])
    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        # prologue
        pid_m, pid_n = compute_pids(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_M)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # main loop
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for off_k in range(0, K, BLOCK_K):
            a = a_desc.load([off_m, off_k])
            b = b_desc.load([off_n, off_k])
            acc = tl.dot(a, b.T, acc)
        # epilogue
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c_desc.store([off_m, off_n], acc0.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N//2], acc1.to(tl.float16))

@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul4(a_ptr, b_ptr, c_ptr, M, N, K,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):

    a_desc = tl._experimental_make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl._experimental_make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K])
    c_desc = tl._experimental_make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N//2])
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)

    tile_id2 = pid - NUM_SMS
    for tile_id1 in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = compute_pids(tile_id1, M, N, BLOCK_M, BLOCK_N, GROUP_M)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for off_k in range(0, K, BLOCK_K):
            a = a_desc.load([off_m, off_k])
            b = b_desc.load([off_n, off_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id2 += NUM_SMS
        pid_m, pid_n = compute_pids(tile_id2, M, N, BLOCK_M, BLOCK_N, GROUP_M)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c_desc.store([off_m, off_n], acc0.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N // 2], acc1.to(tl.float16))
        
        
def run(M, N, K):
    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b = torch.randn((K, N), dtype=torch.float16, device="cuda")
    b = b.T.contiguous()
    c_tri = torch.zeros((M, N), dtype=torch.float16, device="cuda")
    # reference result
    c_ref = torch.matmul(a, b.T)
    # triton result
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 256, 64, 8
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_warps, num_stages = 4, 4 # B200
    # num_warps, num_stages = 8, 3 # H100
    kernel, grid = matmul4, (NUM_SMS,)

    triton.set_allocator(alloc_fn)
    tri_fn = lambda: kernel[grid](a, b, c_tri, M, N, K, 
                            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_SMS,
                            num_warps=num_warps, num_stages=num_stages)
    ref_fn = lambda: torch.matmul(a, b.T)
    tri_fn()
    ref_fn()
    torch.cuda.synchronize()
    tri_ms = triton.testing.do_bench_cudagraph(tri_fn, rep=100)
    ref_ms = triton.testing.do_bench_cudagraph(ref_fn, rep=100)
    diff = (c_tri - c_ref).abs().max()
    tri_tflops = 2*M*N*K / tri_ms * 1e-9
    ref_tflops = 2*M*N*K / ref_ms * 1e-9
    print(f"{K}, {tri_tflops}, {ref_tflops}, {diff}")

for K in range(256, 6144, 128):
    run(8192, 8192, K)
