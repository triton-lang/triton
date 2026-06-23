import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr,
           K: gl.constexpr):
    BM: gl.constexpr = 128
    BN: gl.constexpr = 128
    BK: gl.constexpr = 16

    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BM, BK], gl.float16, cga_layout=((0, 0), ))
    b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BK, BN], gl.float16, cga_layout=((0, 1), ))
    c_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BM, BN], gl.float16, cga_layout=((0, 1), ))

    a_desc = tma.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK],
                                        a_layout)
    b_desc = tma.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN],
                                        b_layout)
    c_desc = tma.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN],
                                        c_layout)

    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BM
    off_n = pid_n * BN

    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(BM, BN // 2),
        col_stride=1,
        cga_layout=((0, 1), ),
        two_ctas=False,
    )
    acc = allocate_tensor_memory(gl.float32, [BM, BN], acc_layout)

    use_acc = False
    for k in range(0, K, BK):
        smem_a = gl.allocate_shared_memory(gl.float16, [BM, BK], a_layout)
        smem_b = gl.allocate_shared_memory(gl.float16, [BK, BN], b_layout)
        a_bar = mbarrier.allocate_mbarrier()
        b_bar = mbarrier.allocate_mbarrier()
        mbarrier.init(a_bar, count=1)
        mbarrier.init(b_bar, count=1)

        mbarrier.expect(a_bar, a_desc.nbytes_per_cta)
        tma.async_load(a_desc, [off_m, k], a_bar, smem_a, multicast=True)
        mbarrier.wait(a_bar, 0)
        mbarrier.invalidate(a_bar)

        mbarrier.expect(b_bar, b_desc.nbytes_per_cta)
        tma.async_load(b_desc, [k, off_n], b_bar, smem_b)
        mbarrier.wait(b_bar, 0)
        mbarrier.invalidate(b_bar)

        # Reuse the B TMA barrier as the MMA completion barrier. Main misses
        # the cluster-visible init sync before this multicast MMA commit.
        mbarrier.init(b_bar, count=2)
        tcgen05_mma(smem_a,
                    smem_b,
                    acc,
                    use_acc=use_acc,
                    multicast=True,
                    mbarriers=[b_bar])
        mbarrier.wait(b_bar, 0)
        mbarrier.invalidate(b_bar)
        use_acc = True

    c_smem = gl.allocate_shared_memory(gl.float16, [BM, BN], c_layout)
    c_smem.store(acc.load().to(gl.float16))
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(0)


def alloc_fn(size, align, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main():
    M, N, K = 1024, 512, 256
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    triton.set_allocator(alloc_fn)
    kernel[(triton.cdiv(M, 128), triton.cdiv(N, 128), 1)](
        a, b, c, M, N, K, num_warps=8, num_ctas=2)
    torch.testing.assert_close(c,
                               torch.matmul(a.float(), b.float()).half(),
                               rtol=1e-3,
                               atol=1e-3)


if __name__ == "__main__":
    main()
