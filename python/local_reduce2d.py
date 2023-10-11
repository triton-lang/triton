import torch

import triton
import triton.language as tl


@triton.jit
def kernel(X, Y, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_n = tl.num_programs(1)
    local_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    for start_n in range(pid_n, tl.cdiv(N, BLOCK_N), num_pid_n):
        off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        Xs = X + off_m[:, None] * N + off_n[None, :]
        x = tl.load(Xs)
        local_sum += tl.sum(x, axis=1)
    tl.store(Y + off_m * num_pid_n + pid_n, local_sum)


torch.manual_seed(0)
BLOCK_M = 32
BLOCK_N = 128
num_pid_n = 2
N = 1024
x = torch.randn((BLOCK_M, N), dtype=torch.float32, device="cuda")
y = torch.empty((BLOCK_M, num_pid_n), dtype=torch.float32, device="cuda")
h = kernel[(1, num_pid_n)](x, y, N, BLOCK_M, BLOCK_N)
print(x.sum(1))
print(y.sum(1))
print(h.asm["ttgir"])
