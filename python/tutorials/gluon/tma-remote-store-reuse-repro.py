"""Reproduce synchronization between multicast TMA and a remote shared store."""

import argparse

import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

BLOCK_M = gl.constexpr(256)
BLOCK_N = gl.constexpr(64)


@gluon.jit
def tma_remote_store_reuse_kernel(input_desc, output):
    gl.static_assert(gl.num_ctas() == 4)

    # The input value is row-sharded across all four CTAs.
    src_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, BLOCK_N // 2],
        threads_per_warp=[16, 2],
        warps_per_cta=[4, 1],
        order=[0, 1],
        cga_layout=((1, 0), (2, 0)),
    )
    dst_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, BLOCK_M // 8],
        threads_per_warp=[8, 4],
        warps_per_cta=[4, 1],
        order=[1, 0],
        cga_layout=((0, 1), (0, 2)),
    )
    replacement_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[1, 0],
        cga_layout=((1, 0), (2, 0)),
    )

    # The TMA destination is a view, so its write effect aliases buffers.
    buffers = gl.allocate_shared_memory(
        gl.float16,
        [1, BLOCK_M // 2, BLOCK_N],
        input_desc.layout,
    )
    tile = buffers.index(0)

    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, input_desc.nbytes_per_cta)

    # The TMA layout forms two multicast groups: CTAs 0/2 and CTAs 1/3.
    # Delay the second group so that the first group can reach the reuse store
    # while the second multicast TMA still targets the old allocation.
    cta_rank = gl.inline_asm_elementwise(
        "mov.u32 $0, %cluster_ctarank;",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )
    if (cta_rank & 1) != 0:
        gl.inline_asm_elementwise(
            "nanosleep.u32 100000; mov.b32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )

    tma.async_load(input_desc, [0, 0], bar, tile, multicast=True)
    mbarrier.wait(bar, 0, deps=[tile])
    buffers._keep_alive()

    # This allocation has the same per-CTA size as buffers, so it can reuse the
    # same shared-memory interval.
    replacement = gl.allocate_shared_memory(
        gl.float16,
        [BLOCK_N, BLOCK_M],
        replacement_layout,
    )

    value = gl.full([BLOCK_M, BLOCK_N], 7, gl.float16, src_layout)
    # value is row-sharded. After transposing it is column-sharded, while the
    # destination remains row-sharded. This local_store therefore targets
    # peer-CTA distributed shared memory and lowers to st.shared::cluster.
    replacement.store(value.trans())

    # Stabilize the observation after the potentially racing store.
    gl.barrier(cluster=True)
    result = replacement.load(dst_layout)
    dst_n = gl.arange(0, BLOCK_N)[:, None]
    dst_m = gl.arange(0, BLOCK_M)[None, :]
    dst_ptrs = output + dst_n * BLOCK_M + dst_m
    gl.store(gl.set_auto_layout(dst_ptrs, dst_layout), result)
    mbarrier.invalidate(bar)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    args = parser.parse_args()

    block_m = BLOCK_M.value
    block_n = BLOCK_N.value
    source = torch.full((block_m, block_n), 2, device="cuda", dtype=torch.float16)
    output = torch.empty((block_n, block_m), device="cuda", dtype=torch.float16)
    input_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m // 2, block_n],
        gl.float16,
        cga_layout=((1, 0), (0, 0)),
    )
    input_desc = TensorDescriptor.from_tensor(source, [block_m // 2, block_n], input_layout)

    failed_trials = 0
    for trial in range(args.trials):
        output.fill_(-1)
        tma_remote_store_reuse_kernel[(1, )](
            input_desc,
            output,
            num_warps=4,
            num_ctas=4,
        )
        torch.cuda.synchronize()
        mismatches = int((output != 7).sum().item())
        if mismatches:
            failed_trials += 1
            if failed_trials <= 5:
                print(f"trial {trial}: {mismatches} mismatches")

    print(f"failed trials: {failed_trials}/{args.trials}")


if __name__ == "__main__":
    main()
