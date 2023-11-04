import torch

from ... import cdiv, heuristics, jit
from ... import language as tl

# ********************************************************
# --------------------------------------------------------
# Sparse = Dense x Dense (SDD)
# This operation uses super-blocking to make sure that
# it's done efficiently when small blocks can be grouped
# together
# --------------------------------------------------------
# ********************************************************


@heuristics({
    'EVEN_K': lambda nargs: nargs['K'] % nargs['TILE_K'] == 0,
})
@jit
def _sdd_kernel(A, B, C,  #
                stride_za, stride_ha, stride_ma, stride_ak,  #
                stride_zb, stride_hb, stride_bk, stride_nb,  #
                stride_zc, stride_hc, stride_mc, stride_nc,  #
                K, grid_offset, lut,  #
                TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,  #
                BLOCK: tl.constexpr, EVEN_K: tl.constexpr  #
                ):
    # ------------ #
    # - Prologue - #
    # ------------ #
    block_id = tl.program_id(0) + grid_offset
    lut += block_id * 3
    # offsets
    off_z = tl.program_id(2)  # batch
    off_h = tl.load(lut + 0)  # head

    # initialize pointers to A
    start_am = tl.load(lut + 1)
    offs_am = start_am * BLOCK + (tl.arange(0, TILE_M) % BLOCK)
    offs_ak = tl.arange(0, TILE_K)
    a_ptrs = A \
        + off_z * stride_za \
        + off_h * stride_ha \
        + offs_am[:, None] * stride_ma \
        + offs_ak[None, :] * stride_ak
    # initialize pointers to B
    start_bn = tl.load(lut + 2)
    offs_bn = start_bn * BLOCK + (tl.arange(0, TILE_N) % BLOCK)
    offs_bk = tl.arange(0, TILE_K)
    b_ptrs = B \
        + off_z * stride_zb \
        + off_h * stride_hb \
        + offs_bn[None, :] * stride_nb \
        + offs_bk[:, None] * stride_bk
    # ---------------- #
    #    Inner Loop    #
    # ---------------- #
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(K, 0, -TILE_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_ak[None, :] < k, other=0.)
            b = tl.load(b_ptrs, mask=offs_bk[:, None] < k, other=0.)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk
    c = acc.to(C.dtype.element_ty)
    # ---------------- #
    #    Epilogue      #
    # ---------------- #
    offs_cm = tl.arange(0, TILE_M) % BLOCK
    offs_cn = tl.arange(0, TILE_N) % BLOCK
    pc = C \
        + off_z * stride_zc \
        + block_id * stride_hc \
        + offs_cm[:, None] * stride_mc \
        + offs_cn[None, :] * stride_nc
    tl.store(pc, c, mask=True)


def sdd_matmul(a, b, trans_a, trans_b, trans_c, spdims, block, lut, widths, out=None):
    if a.stride(2) != 1 and a.stride(3) != 1:
        a = a.contiguous()
    if b.stride(2) != 1 and b.stride(3) != 1:
        b = b.contiguous()
    # (A * B)^T = B^T * A^T
    if trans_c:
        a, b = b, a
        trans_a, trans_b = not trans_b, not trans_a
    # shape constraints
    a_dim = -2 if trans_a else -1
    b_dim = -1 if trans_b else -2
    Ka, Kb = a.shape[a_dim], b.shape[b_dim]
    if Ka != Kb:
        raise ValueError(f"Inner dimension mismatch (A: {Ka} vs B: {Kb})")
    # allocate output
    if out is None:
        c = torch.empty((a.shape[0], lut.shape[0], block, block), dtype=a.dtype, device=a.device)
    else:
        assert out.shape == (a.shape[0], lut.shape[0], block, block)
        c = out
    grid = [c.shape[1], 1, c.shape[0]]
    _sdd_kernel[grid](
        a, b, c,  #
        a.stride(0), a.stride(1), a.stride(3 if trans_a else 2), a.stride(2 if trans_a else 3),  #
        b.stride(0), b.stride(1), b.stride(3 if trans_b else 2), b.stride(2 if trans_b else 3),  #
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),  #
        Ka, 0, lut,  #
        TILE_M=block, TILE_N=block, TILE_K=32, BLOCK=block, num_stages=4,  #
        num_warps=4  #
    )
    return c


def sdd_lut(layout, block, device):
    lut = layout.nonzero(as_tuple=False).to(device).int()
    lut = lut.contiguous()
    return lut, None


# -----------------------------
# Dense = Sparse x Dense (DSD)
# This operation uses a look-up table that contains pre-computed pointer increments
# in order to minimize computations in the inner loop of the matmul kernel.
# -----------------------------


@jit
def _dsd_kernel(A, B, C,  #
                stride_az, stride_ha, stride_am, stride_ak,  #
                stride_zb, stride_hb, stride_bk, stride_bn,  #
                stride_zc, stride_hc, stride_cm, stride_cn,  #
                DS0, DS1, lut,  #
                TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,  #
                GROUP_SIZE_M: tl.constexpr, BLOCK: tl.constexpr  #
                ):
    # ------------ #
    # - Prologue - #
    # ------------ #
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_SIZE_M)
    pidz = tl.program_id(2)
    header = lut + pid_n * 4
    offset = tl.load(header + 0)
    K = tl.load(header + 1)
    column = tl.load(header + 2)
    off_h = tl.load(header + 3)
    pinc = lut + offset
    # initialize pointers to A (sparse)
    block_id = tl.load(pinc + 1)
    block_id = tl.multiple_of(block_id, 8)  # compiler hint
    offs_am = tl.arange(0, TILE_M)
    offs_ak = tl.arange(0, TILE_K)
    pa = A + pidz * stride_az \
        + block_id * stride_ha \
        + offs_am[:, None] * stride_am \
        + offs_ak[None, :] * stride_ak
    # initialize pointers to B (dense)
    offs_bn = pid_m * TILE_N + tl.arange(0, TILE_N)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn % DS0, TILE_N), TILE_N)
    start_bk = tl.load(pinc)
    start_bk = tl.multiple_of(start_bk, 8)  # compiler hint
    offs_bk = start_bk + tl.arange(0, TILE_K)
    pb = B + pidz * stride_zb \
        + off_h * stride_hb \
        + offs_bn[None, :] * stride_bn \
        + offs_bk[:, None] * stride_bk
    # ---------------- #
    #    Inner Loop    #
    # ---------------- #
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    pinc += 2
    inc_a = tl.load(pinc + 1)
    inc_a = tl.multiple_of(inc_a, 8)
    inc_b = tl.load(pinc)
    inc_b = tl.multiple_of(inc_b, 8)
    for k in range(K, 0, -TILE_K):
        a = tl.load(pa)
        b = tl.load(pb)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        pa += inc_a
        pb += inc_b * stride_bk
        pinc += 2
        inc_a = tl.load(pinc + 1)
        inc_a = tl.multiple_of(inc_a, 8)
        inc_b = tl.load(pinc)
        inc_b = tl.multiple_of(inc_b, 8)
    c = acc.to(C.dtype.element_ty)
    # initialize pointers to C
    offs_cm = column * TILE_M + tl.arange(0, TILE_M)
    offs_cn = pid_m * TILE_N + tl.arange(0, TILE_N)
    pc = C \
        + off_h * stride_hc \
        + pidz * stride_zc \
        + offs_cm[:, None] * stride_cm \
        + offs_cn[None, :] * stride_cn
    tl.store(pc, c, mask=offs_cn[None, :] < DS0)


def dsd_matmul(a, b, trans_a, trans_b, trans_c, spdims, block, lut, width, out=None):
    if a.stride(2) != 1 and a.stride(3) != 1:
        a = a.contiguous()
    if b.stride(2) != 1 and b.stride(3) != 1:
        b = b.contiguous()
    # shapes / dtypes
    AS1 = block * spdims[2 if trans_a else 1]
    BS0 = b.size(0)
    BS1 = b.size(1)
    BS3 = b.size(2 if trans_b else 3)
    dtype = a.dtype
    # allocate output
    CS0 = BS0
    CS1 = BS1
    CS2 = BS3 if trans_c else AS1
    CS3 = AS1 if trans_c else BS3
    if out is None:
        c = torch.empty((CS0, CS1, CS2, CS3), dtype=dtype, device=a.device)
    else:
        assert out.shape == (CS0, CS1, CS2, CS3)
        c = out
    # meta-parameter heuristics
    TILE_N = 128
    # compute output
    grid = lambda meta: [cdiv(BS3, meta['TILE_N']), width, BS0]
    _dsd_kernel[grid](
        a, b, c,  #
        a.stride(0), a.stride(1), a.stride(3 if trans_a else 2), a.stride(2 if trans_a else 3),  #
        b.stride(0), b.stride(1), b.stride(3 if trans_b else 2), b.stride(2 if trans_b else 3),  #
        c.stride(0), c.stride(1), c.stride(3 if trans_c else 2), c.stride(2 if trans_c else 3),  #
        BS3, AS1, lut,  #
        TILE_M=block, TILE_N=TILE_N, TILE_K=min(block, 32), BLOCK=block, num_stages=4,  #
        num_warps=4, GROUP_SIZE_M=4  #
    )
    # exit()
    return c


def dsd_lut(layout, block, step, trans, device):
    """
    Generates the look-up table for incrementing pointers in the DSD/DDS matmul.
    Example (BLOCK=32, STEP=16)
    [[1, 0, 0, 1, 0],
     [0, 1, 1, 0, 1],
     [1, 0, 1, 0, 0]]

    Then the offsets for A are
     [0 , 16, 32, 48] <- row 0
      \\----/  \\----/
      col=0   col=3
     [64, 80, 96, 112, 128, 144] <- row 1
      \\----/   \\----/  \\------/
       col=1    col=2    col=3
     [160, 176, 192, 208]
    which leads to increments table
    [0, 16, 16, 16, || 64, 16, 16, 16, 16, 16, || 160, 16, 16, 16]

    Because B is dense, the offsets are
    [0, 16, 96, 112] <- row 0
    [32, 48, 64, 80]  <- row 1
    [0, 16, 64, 80]   <- row 2
    """
    sizes = torch.sum(layout, 2 if trans else 1)
    head_id, col_id = torch.ones_like(sizes).nonzero(as_tuple=True)
    sizes = sizes.flatten()
    segments = sizes * step
    # pointer increments
    if trans:
        nnz = layout.nonzero(as_tuple=False)
    else:
        nnz = layout.transpose(1, 2).nonzero(as_tuple=False)
    num_blocks = nnz.size(0)
    offsets = torch.zeros_like(sizes)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    offsets = torch.min(offsets, (num_blocks - 1) * torch.ones_like(offsets))
    # -------------------------------
    # dense input pointer increments
    # -------------------------------
    # Note that the inner loop matmul kernel may have a fixed step size (e.g., TILE_K)
    # that is smaller than the block size, so we need to do a bit of extra work
    # to handle this case
    B_idx = nnz[:, 2] * block
    B_incs = B_idx.clone()
    B_incs[1:] -= B_idx[:-1]
    div = block // step
    B_incs = B_incs.view(-1, 1).repeat(1, div)
    B_incs[:, 1:] = step
    B_incs[:, 0] -= (div - 1) * step
    # first increment for each reduction is actually the offset
    B_incs[offsets[segments > 0], 0] = B_idx[offsets[segments > 0]]
    B_incs = B_incs.view(-1)
    # -------------------------------
    # sparse input pointer increments
    # -------------------------------
    # same as above, except that the increments are in the sparse memory layout
    if trans:
        A_idx = torch.arange(num_blocks, device=layout.device)
    else:
        A_idx = torch.tensor([], dtype=torch.int64, device=layout.device)
        current_offset = 0
        for z in range(layout.size(0)):
            layoutw = layout[z, :, :].clone().long()
            msum = layoutw.sum()
            layoutw[layoutw > 0] = 1 + torch.arange(msum, device=layout.device)
            A_idx = torch.cat((A_idx, current_offset + layoutw.T[layoutw.T > 0] - 1))
            current_offset += msum
    A_incs = A_idx * block * block
    A_incs[1:] -= A_idx[:-1] * block * block
    A_incs = A_incs.view(-1, 1).repeat(1, div)
    if trans:
        A_incs[:, 1:] = step
        A_incs[:, 0] -= (div - 1) * step
    else:
        A_incs[:, 1:] = step * block
        A_incs[:, 0] -= (div - 1) * step * block
    A_incs[offsets[segments > 0], 0] = A_idx[offsets[segments > 0]]
    A_incs = A_incs.view(-1)
    # create header
    width = col_id.size(0)
    offsets = offsets * 2 * div + 4 * width
    segments = segments * div
    header = torch.stack((offsets, segments, col_id, head_id), dim=1).view(-1).contiguous()
    # create increments
    incs = torch.stack((B_incs, A_incs), dim=1).view(-1).contiguous()
    # pad by a factor 2*MAX_NUM_STAGES
    # to accommodate pre-fetching inside the kernel
    pad = torch.zeros(20, device=incs.device, dtype=incs.dtype)
    incs = torch.cat((incs, pad))
    # create lut
    lut = torch.cat((header, incs))
    lut = lut.type(torch.int32).to(device)
    # create locks
    return lut, width


# -----------------------------
# Dense = Dense x Sparse (DDS)
# -----------------------------
# AB = (B^T A^T)^T


def dds_matmul(a, b, trans_a, trans_b, trans_c, spdims, block, lut, width, out=None):
    return dsd_matmul(b, a, not trans_b, not trans_a, not trans_c, spdims, block, lut, width, out=out)


##############
#  MAIN API  #
##############


class _matmul(torch.autograd.Function):

    fn = {'sdd': sdd_matmul, 'dsd': dsd_matmul, 'dds': dds_matmul}

    @staticmethod
    def forward(ctx, a, b, trans_a, trans_b, trans_c, mode, spdims, block, c_lut, c_width, da_lut, da_width, db_lut,
                db_width, out):
        c = _matmul.fn[mode](a, b, trans_a, trans_b, trans_c, spdims, block, c_lut, c_width, out=out)
        # save for backward
        ctx.save_for_backward(a, b)
        ctx.da_lut = da_lut
        ctx.da_width = da_width
        ctx.db_lut = db_lut
        ctx.db_width = db_width
        ctx.mode = mode
        ctx.spdims = spdims
        ctx.block = block
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.trans_c = trans_c
        ctx.has_out = out is not None
        return c

    @staticmethod
    def backward(ctx, dc):
        # saved for backward
        a, b = ctx.saved_tensors
        da, db = None, None
        mode = ctx.mode
        # gradients w.r.t. a
        if ctx.needs_input_grad[0]:
            mode_da = mode[1] + mode[0] + mode[2]
            da = _matmul.fn[mode_da](dc, b, ctx.trans_c, not ctx.trans_b, ctx.trans_a, ctx.spdims, ctx.block,
                                     ctx.da_lut, ctx.da_width)
        # gradients w.r.t. b
        if ctx.needs_input_grad[1]:
            mode_db = mode[2] + mode[1] + mode[0]
            db = _matmul.fn[mode_db](a, dc, not ctx.trans_a, ctx.trans_c, ctx.trans_b, ctx.spdims, ctx.block,
                                     ctx.db_lut, ctx.db_width)
        dout = dc if ctx.has_out else None
        return da, db, None, None, None, \
            None, None, None, None, \
            None, None, None, None, None, dout


class matmul:

    def __init__(self, layout, block, mode, device, trans_a=False, trans_b=False, trans_c=False):
        if mode not in ['sdd', 'dsd', 'dds']:
            raise NotImplementedError('Supported modes are: sdd, dsd, dds')
        self.block = block
        self.mode = mode
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.layout = layout
        self.spdims = layout.shape
        step = min(block, 32)
        if self.mode == 'sdd':
            self.c_lut, self.c_width = sdd_lut(layout, block, device)
            self.da_lut, self.da_width = dsd_lut(layout, block, step, True, device)
            self.db_lut, self.db_width = dsd_lut(layout, block, step, False, device)
        if self.mode == 'dsd':
            self.c_lut, self.c_width = dsd_lut(layout, block, step, not self.trans_a, device)
            self.da_lut, self.da_width = sdd_lut(layout, block, device)
            self.db_lut, self.db_width = dsd_lut(layout, block, step, self.trans_a, device)
        if self.mode == 'dds':
            self.c_lut, self.c_width = dsd_lut(layout, block, step, self.trans_b, device)
            self.da_lut, self.da_width = dsd_lut(layout, block, step, not self.trans_b, device)
            self.db_lut, self.db_width = sdd_lut(layout, block, device)

    def __call__(self, a, b, out=None):
        c = _matmul.apply(a, b, self.trans_a, self.trans_b, self.trans_c, self.mode, self.spdims, self.block,  #
                          self.c_lut, self.c_width,  #
                          self.da_lut, self.da_width,  #
                          self.db_lut, self.db_width,  #
                          out)
        return c
