import torch
from torch.autograd.function import FunctionCtx

import triton
from triton_kernels.distributed import SymmetricMemoryPool
from triton_kernels.op import Op
from triton_kernels.tensor import SparseMatrix, Tensor, dtype_to_torch_dtype, wrap_torch_tensor
from triton_kernels.tensor_details.dtype import BIT, DataType
from triton_kernels.tensor_details.sharding import RangeSharding
from triton_kernels.tensor_metadata import extend_sharding, get_sharding_local_if_unset
from triton_kernels.tensor_types import Dim, Sharded, Size, Unsharded
from triton_kernels.topk_details._topk_backward import _topk_backward
from triton_kernels.topk_details._topk_forward import _topk_forward


def make_empty(
    shape: tuple[int, ...],
    dtype: torch.dtype | DataType,
    device: torch.device,
    all_gather: bool,
    symm_mem_pool: SymmetricMemoryPool | None,
    region: str = "topk_vals",
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    dtype = dtype_to_torch_dtype(dtype)
    if all_gather:
        rank_id = symm_mem_pool.process_group.local_rank
        ret_bufs = symm_mem_pool.get_tensors(shape=shape, dtype=dtype, region=region)
        ret = ret_bufs[rank_id]
        return ret_bufs, ret
    ret = torch.empty(shape, dtype=dtype, device=device)
    return (ret, ), ret


def topk_forward(
    x: Tensor,
    k: int,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: torch.Tensor | None = None,
    symm_mem_pool: SymmetricMemoryPool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Tensor]:
    tensor_sharding = get_sharding_local_if_unset(x, 0)
    all_gather = not tensor_sharding.is_local

    cdiv = lambda a, b: (a + b - 1) // b
    BLOCK_M = 32
    BLOCK_N = 32
    use_provided_indx = y_indx is not None
    assert symm_mem_pool is not None or not all_gather
    assert len(x.shape) == 2
    assert x.shape[-1] < 32768
    assert dim == 1

    sharding = tensor_sharding.sharding
    rank = sharding.mesh.local_rank
    sharding_type = sharding.triton_sharding_type
    world_size = sharding.mesh.world_size
    replication_factor = getattr(sharding, "replication_factor", 1)
    n_rows_local_max = x.local_shape_max[0]

    n_rows_out_max, n_cols = x.global_shape_max
    dev = x.storage.data.device

    # scratchpad tensors
    # NOTE: these are not returned
    y_vals_bufs, y_vals = make_empty(
        (n_rows_out_max, k),
        x.dtype,
        dev,
        all_gather=all_gather,
        symm_mem_pool=symm_mem_pool,
        region="topk_vals",
    )
    if y_indx is None:
        y_indx_bufs, y_indx = make_empty(
            (n_rows_out_max, k),
            torch.int16,
            dev,
            all_gather=all_gather,
            symm_mem_pool=symm_mem_pool,
            region="topk_y_indx",
        )
    else:
        y_indx_bufs = (y_indx, )
    # create bitmatrix in transposed memory layout:
    n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix_bufs, bitmatrix_data = make_empty(
        (n_cols_words, cdiv(n_rows_out_max, 32) * 32),
        torch.uint32,
        dev,
        all_gather=all_gather,
        symm_mem_pool=symm_mem_pool,
        region="topk_bitmatrix",
    )
    bitmatrix_data = torch.transpose(bitmatrix_data, 0, 1)[:n_rows_local_max]
    pids = cdiv(n_rows_local_max, BLOCK_M)
    _topk_forward[(pids, )](
        x.storage.data,
        x.storage.data.stride(0),  # inputs
        y_vals_bufs,
        y_indx_bufs,
        y_vals.stride(0),
        use_provided_indx,  # output [topk]
        bitmatrix_bufs,
        bitmatrix_data.stride(0),
        bitmatrix_data.stride(1),  # output [bitmatrix]
        x.shape[0],
        n_cols,  # shapes
        rank,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,  # tunable parameter
        APPLY_SOFTMAX=apply_softmax,
        N_EXPTS_PAD=n_cols_pad,
        N_EXPTS_ACT=k,  # constants
        SHARDING_TYPE=sharding_type,
        REPLICATION_FACTOR=replication_factor,
        WORLD_SIZE=world_size,
    )
    if all_gather:
        symm_mem_pool.process_group.barrier()

    bitmatrix = wrap_torch_tensor(
        bitmatrix_data,
        dtype=BIT,
        shape=x.shape[:],
        local_shape_max=x.global_shape_max[:],
    )
    return y_vals, y_indx, bitmatrix


def topk_backward(
    x: torch.Tensor,
    y_indx: torch.Tensor,
    dy_vals: torch.Tensor,
    k: int,
    n_rows: int | None,
    apply_softmax: bool,
) -> torch.Tensor:
    assert dy_vals.shape[-1] == k
    n_expts_pad = triton.next_power_of_2(x.shape[-1])
    dx = torch.empty_like(x)
    _topk_backward[(dy_vals.shape[0], )](
        y_indx,
        y_indx.stride(0),
        dy_vals,
        dy_vals.stride(0),
        x,
        x.stride(0),  # inputs
        dx,  # outputs
        dx.stride(0),
        x.shape[0],
        n_rows,
        x.shape[-1],
        APPLY_SOFTMAX=apply_softmax,
        N_EXPTS_ACT=k,
        N_EXPTS_PAD=n_expts_pad,
    )
    return dx


@Op(types={
    "x": (Dim(2) & (Size(1) < 32768) & (Unsharded() | Sharded(0, uniform=True))),
}, )
def topk_triton(
    x: Tensor,
    k: int,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: torch.Tensor | None = None,
    symm_mem_pool: SymmetricMemoryPool | None = None,
) -> SparseMatrix:
    y_vals, y_indx, bitmatrix = topk_forward(x, k, apply_softmax, dim, y_indx, symm_mem_pool)
    return SparseMatrix(vals=y_vals, indx=y_indx, mask=bitmatrix)


class TopK(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        k: int,
        apply_softmax: bool,
        dim: int,
        y_indx: torch.Tensor | None,
        n_rows: int | torch.Tensor | None,
        all_gather: bool | None,
        symm_mem_pool: SymmetricMemoryPool | None,
    ) -> tuple[torch.Tensor, torch.Tensor, Tensor]:
        assert isinstance(x, torch.Tensor)

        x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
        x_triton = wrap_torch_tensor(x, shape=x_shape, local_shape_max=x.shape[:])
        if all_gather:
            assert symm_mem_pool is not None
            x_triton = extend_sharding(x_triton, dim=0, sharding=RangeSharding(mesh=symm_mem_pool.process_group))

        m = topk_triton(x_triton, k, apply_softmax, dim, y_indx, symm_mem_pool)
        y_vals, y_indx, bitmatrix = m.vals, m.indx, m.mask

        ctx.save_for_backward(x, y_indx)
        ctx.apply_softmax = apply_softmax
        ctx.k = k
        ctx.n_rows = n_rows
        return y_vals, y_indx, bitmatrix

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        dy_vals: torch.Tensor,
        _0: torch.Tensor | None,
        _1: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None, None, None, None, None, None, None]:
        x, y_indx = ctx.saved_tensors
        dx = topk_backward(x, y_indx, dy_vals, ctx.k, ctx.n_rows, ctx.apply_softmax)
        return dx, None, None, None, None, None, None, None


def topk(
    x: Tensor | torch.Tensor,  # REVIEW: torch.Tensor or triton_kernels.Tensor
    k: int,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: torch.Tensor | None = None,
    n_rows: int | None = None,
    all_gather: bool | None = None,
    symm_mem_pool: SymmetricMemoryPool | None = None,
) -> SparseMatrix:
    """
    Computes the top-k values and indices along a specified dimension of a tensor.
    Note that the input can be either a `Tensor` or a `torch.Tensor`, but the output will always be a `torch.Tensor`.

    Parameters
    ----------
    x : Union[triton_kernels.Tensor, torch.Tensor]
        Input tensor of shape (n_tokens, n_expts).
    k : int
        Number of top elements to retrieve.
    apply_softmax : bool, default True
        Whether to apply softmax to the input tensor before computing top-k.
    dim : int, default 1
        Dimension along which to compute top-k.
    y_indx : torch.Tensor, optional
        Pre-allocated tensor for storing indices of top-k elements with shape (n_tokens, k).
        If provided, we skip the computation of top-k indices and use this tensor instead.
    n_rows : int, optional
        Number of rows to apply top-k on. If None, we consider all rows in `x`.

    Returns
    -------
    SparseMatrix: sparse matrix equal to `x` with non-selected entries set to 0
    """
    if isinstance(x, torch.Tensor):
        # Backwards compatibility
        y_vals, y_indx, bitmatrix = TopK.apply(x, k, apply_softmax, dim, y_indx, n_rows, all_gather, symm_mem_pool)
        return SparseMatrix(vals=y_vals, indx=y_indx, mask=bitmatrix)

    assert all_gather is None
    assert n_rows is None
    return topk_triton(x, k, apply_softmax, dim, y_indx, symm_mem_pool)


def topk_torch(
    x: torch.Tensor,
    k: int,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: torch.Tensor | None = None,
    n_rows: int | None = None,
) -> SparseMatrix:
    if n_rows is None:
        n_rows = x.shape[0]
    has_user_provided_indx = y_indx is not None
    cdiv = lambda a, b: (a + b - 1) // b
    device = x.device
    assert dim == 1
    assert not isinstance(x, Tensor)
    if not has_user_provided_indx:
        y_indx = torch.argsort(-x, dim=1, stable=True)[:, :k]
    y_indx = y_indx.long()
    y_vals = torch.take_along_dim(x[:n_rows, :], y_indx[:n_rows, :], dim=1)
    y_vals = torch.cat([y_vals, x[n_rows:, :k]], dim=0)
    y_indx = y_indx.int()
    # compute bitmatrix
    _, n_cols = x.shape
    bitmatrix_data = torch.zeros((cdiv(n_cols, 32), cdiv(x.shape[0], 32) * 32), dtype=torch.int32, device=device)
    bitmatrix_data = torch.transpose(bitmatrix_data, 0, 1)[:x.shape[0]]
    # fill bitmatrix
    if apply_softmax:
        y_vals = torch.softmax(y_vals.float(), dim=-1).to(x.dtype)
    if not has_user_provided_indx:
        y_vals, sort_indices = torch.sort(y_vals.float(), dim=1, descending=True, stable=True)
        y_indx = torch.gather(y_indx, 1, sort_indices)
    y_indx[n_rows:, :] = -1
    rows = (torch.arange(x.shape[0], device=device).unsqueeze(1).expand(-1, y_indx.shape[1]).reshape(-1))
    cols = y_indx.reshape(-1)  # 64-bit safe for div/mod
    word_idx = torch.div(cols, 32, rounding_mode="floor")
    bit_idx = cols % 32
    masks = torch.ones_like(bit_idx) << bit_idx
    bitmatrix_data.index_put_((rows, word_idx), masks, accumulate=True)
    bitmatrix_data = bitmatrix_data.view(torch.uint32)

    bitmatrix = wrap_torch_tensor(bitmatrix_data, dtype=BIT, shape=x.shape)
    return SparseMatrix(vals=y_vals, indx=y_indx, mask=bitmatrix)
