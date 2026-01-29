import pytest
import torch
import triton.profiler as proton
from triton_kernels.topk import topk, topk_torch
from triton_kernels.testing import assert_equal, assert_close
from triton_kernels.distributed import SymmetricMemoryPool
import torch.distributed as dist


@pytest.mark.parametrize("n_rows", [1, 7, 256, 300])
@pytest.mark.parametrize("n_cols", [13, 32, 128, 200])
@pytest.mark.parametrize("k", [8])
@pytest.mark.parametrize("apply_softmax", [True, False])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_topk(n_rows, n_cols, k, apply_softmax, dtype):
    device = "cuda"

    torch.manual_seed(0)
    dtype = getattr(torch, dtype)
    x = torch.randn((n_rows, n_cols), dtype=torch.float32, device=device)
    sparse_x_tri = topk(x, k, apply_softmax=apply_softmax)
    sparse_x_ref = topk_torch(x, k, apply_softmax=apply_softmax)
    assert_close(sparse_x_tri.vals, sparse_x_ref.vals)
    assert_equal(sparse_x_tri.indx, sparse_x_ref.indx)
    assert_equal(sparse_x_tri.mask.storage.data, sparse_x_ref.mask.storage.data)
    assert sparse_x_tri.mask.storage.data.stride() == sparse_x_ref.mask.storage.data.stride()
    assert sparse_x_tri.mask.storage.data.shape == sparse_x_ref.mask.storage.data.shape


def bench_topk(n_rows, n_cols, k, apply_softmax, all_gather=False):
    # setup distributed environment
    rank, world_size = 0, 1
    if all_gather:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank)
    # run benchmark
    x = torch.randn((n_rows, n_cols), dtype=torch.float32, device=f"cuda:{rank}")
    symm_mem_pool = SymmetricMemoryPool()
    symm_mem_pool._reserve_region("topk", world_size * x.numel() * x.element_size(), 128, 0)
    symm_mem_pool._initialize(world_size, group=torch.distributed.group.WORLD, device=x.device)
    proton.start(f"profile_{rank}", hook="triton")
    # warmup
    proton.deactivate()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g):
            _ = topk(x, k, apply_softmax=apply_softmax, all_gather=all_gather, symm_mem_pool=symm_mem_pool)
    torch.cuda.synchronize()
    proton.activate()
    for i in range(100):
        g.replay()
    dist.barrier()
    torch.cuda.synchronize()
    proton.finalize()
    symm_mem_pool.release()


if __name__ == "__main__":
    bench_topk(1024, 1024, 8, False, all_gather=True)
