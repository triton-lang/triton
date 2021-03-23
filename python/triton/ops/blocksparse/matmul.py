import triton
import triton._C.libtriton as libtriton
import torch
import os
import math

src = triton.read(os.path.join(os.path.dirname(__file__), 'matmul.c'))


##############
#  MAIN API  #
##############
class _matmul(torch.autograd.Function):

    sdd_cache = dict()
    dsd_cache = dict()
    dds_cache = dict()
    locks = dict()

    # Given an array sizes representing reduction size for each
    # column of a block-mode matrix multiplication,
    # performs load-balancing to achieve more smaller reductions
    # between `seg_size` elements
    @staticmethod
    def load_balance(sizes, block):
        # segment size
        # heuristics taken from OpenAI blocksparse code
        # https://github.com/openai/blocksparse/blob/master/blocksparse/matmul.py#L95
        max_size = sizes.max()
        min_size = sizes[sizes != 0].min()
        #if max_size > min_size * 2.0:
        #  seg_max = max(triton.cdiv(max_size, 4), min_size*2)
        #else:
        #  seg_max = max_size
        seg_max = max_size
        seg_min = max(triton.cdiv(seg_max, 4), 4)
        # split reduction into segments
        div = sizes // seg_max
        rem = sizes % seg_max
        packs = div + (sizes < seg_min).long() + (rem >= seg_min).long()
        width = packs.sum()
        segments = torch.empty(width, dtype=sizes.dtype)
        column = torch.empty_like(segments)
        lockid = torch.zeros_like(segments)
        maxid = torch.zeros_like(segments)
        nlocks = 0
        current = 0
        col_idx = 0
        for i in range(len(sizes)):
            d, r = div[i], rem[i]
            isempty = sizes[i] < seg_min
            last = current + d + (r >= seg_min) + isempty
            # column id
            column[current:last] = col_idx
            # lock id
            if d > 1 or (d == 1 and r >= seg_min):
                nlocks += 1
                lockid[current:last] = nlocks
                maxid[current:last] = last - current
            # segment size
            segments[current:current + d] = seg_max
            if r < seg_min and not isempty:
                segments[current + d - 1] += r
            if r >= seg_min or isempty:
                segments[current + d] = r
            current = last
            col_idx += 1
        offsets = torch.zeros_like(segments)
        offsets[1:] = torch.cumsum(segments[:-1], dim=0)
        return segments, column, lockid, maxid, offsets

    @staticmethod
    def get_locks(size, dev):
        if dev not in _matmul.locks or \
            size > _matmul.locks[dev].size(0):
            _matmul.locks[dev] = torch.zeros(size, dtype=torch.int32, device=dev)
        return _matmul.locks[dev]

    ##########################
    # SPARSE = DENSE x DENSE #
    ##########################

    @staticmethod
    def make_sdd_lut(layout, block, dtype, device):
        start_width = 128 // block
        layout = layout.type(torch.int32)
        superblocks = libtriton.superblock(layout.data_ptr(), layout.shape[0], layout.shape[1], layout.shape[2], start_width)
        luts, widths, packs = [], [], []
        for size, nnz in superblocks:
            nnz = nnz.reshape(-1, 4)
            width = nnz.shape[0] // (size * size)
            luts.append(torch.from_numpy(nnz).type(torch.int32).to(device))
            widths.append(width)
            packs.append(size)
        # create locks
        return luts, None, widths, packs

    @staticmethod
    def _sdd_matmul(a, b, trans_a, trans_b, trans_c, spdims, block, luts, num_locks, widths, packs):

        if trans_c:
            a, b = b, a
            trans_a, trans_b = not trans_b, not trans_a
        AS0 = a.size(0)
        AS1 = a.size(1)
        AS2 = a.size(3 if trans_a else 2)
        AS3 = a.size(2 if trans_a else 3)
        BS0 = b.size(0)
        BS1 = b.size(1)
        BS2 = b.size(3 if trans_b else 2)
        BS3 = b.size(2 if trans_b else 3)
        dtype = a.dtype
        device = a.device
        is_16_multiple = AS3 % 16 == 0
        is_32_multiple = AS3 % 32 == 0
        is_64_multiple = AS3 % 64 == 0
        if not is_16_multiple:
            raise ValueError('Reduction size for SDD must be a multiple of 16')
        # create kernel
        total_width = sum([width * pack * pack for width, pack in zip(widths, packs)])
        c = torch.empty((AS0, total_width, block, block), dtype=dtype, device=device)
        for lut, width, pack in zip(luts, widths, packs):
            num_lock = 1
            key = (block, device, a.dtype, b.dtype, trans_a, trans_b, trans_c, pack, is_32_multiple, is_64_multiple)
            if key not in _matmul.sdd_cache:
                defines = {
                    'TM': block * pack,
                    'TN': block * pack,
                    'TMN': block * block * pack * pack,
                    'BLOCK': block,
                    'TK': 32,
                    'TYPE': dtype,
                    'STRIDE_AM': '1' if trans_a else 'lda',
                    'STRIDE_AK': 'lda' if trans_a else '1',
                    'STRIDE_BN': 'ldb' if trans_b else '1',
                    'STRIDE_BK': '1' if trans_b else 'ldb',
                    'STRIDE_CM': 'ldc',
                    'STRIDE_CN': '1',
                    'SDD': True,
                    'TZ': 1,
                    'NAME': 'sdd_kernel'
                }
                _matmul.sdd_cache[key] = triton.kernel(src, device=device, defines=defines)

            kernel = _matmul.sdd_cache[key]
            # create output
            locks = _matmul.get_locks(2 * width * AS0 * num_lock, a.device)
            # maximum grid size is 65535
            # so operation might be decomposed into multiple
            # kernel calls
            max_width = 49152
            for off_width in range(0, width, max_width):
                kernel(
                    a.data_ptr(),
                    b.data_ptr(),
                    c.data_ptr(),
                    a.stride(2),
                    b.stride(2),
                    block,
                    a.stride(0),
                    b.stride(0),
                    c.stride(0),
                    a.stride(1),
                    b.stride(1),
                    c.stride(0),
                    AS2,
                    AS2,
                    AS3,
                    off_width,
                    lut.data_ptr(),
                    locks.data_ptr(),
                    num_lock,
                    grid=lambda opt: [opt.TZ, min(max_width, width - off_width), AS0]
                )
        # save for backward pass
        return c

    ##########################
    # DENSE = DENSE x SPARSE #
    # DENSE = SPARSE x DENSE #
    ##########################

    # Given a binary layout of 0s and 1s,
    # Construct look-up table for efficient execution on GPUs
    @staticmethod
    def make_dxx_lut(layout, block, step, trans, device, transform=lambda idx: idx):
        # load-balancing
        _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
        segments = _empty.clone()
        column = _empty.clone()
        depth = _empty.clone()
        lockid = _empty.clone()
        maxid = _empty.clone()
        offsets = _empty.clone()
        current_offset = 0
        current_maxid = 0
        for z in range(layout.size(0)):
            if trans:
                sizes = torch.sum(layout[z, :, :], 1)
            else:
                sizes = torch.sum(layout[z, :, :], 0)
            z_segments, z_column, z_lockid, z_maxid, z_offsets = _matmul.load_balance(sizes, block)
            z_depth = z * torch.ones_like(z_segments)
            z_lockid[z_lockid > 0] += current_maxid
            current_maxid = z_lockid.max()
            # concatenate depth
            segments = torch.cat((segments, z_segments))
            column = torch.cat((column, z_column))
            depth = torch.cat((depth, z_depth))
            maxid = torch.cat((maxid, z_maxid))
            offsets = torch.cat((offsets, current_offset + z_offsets))
            lockid = torch.cat((lockid, z_lockid))
            current_offset += layout[z, :, :].sum()
        segments *= step
        # pointer increments
        if trans:
            nnz = layout.nonzero(as_tuple=False)
        else:
            nnz = layout.transpose(1, 2).nonzero(as_tuple=False)
        num_blocks = nnz.size(0)
        offsets = torch.min(offsets, (num_blocks - 1) * torch.ones_like(offsets))
        idx = transform(nnz[:, 2] * block)
        xincs = idx.clone()
        xincs[1:] -= idx[:-1]
        # divide block into multiple steps
        div = block // step
        xincs = xincs.view(-1, 1).repeat(1, div)
        xincs[:, 1:] = step
        xincs[:, 0] -= (div - 1) * step
        # first increment for each reduction is actually the offset
        xincs[offsets[segments > 0], 0] = idx[offsets[segments > 0]]
        xincs = xincs.view(-1)
        # block-mode input increments
        if trans:
            widx = torch.arange(num_blocks)
        else:
            widx = _empty.clone()
            current_offset = 0
            for z in range(layout.size(0)):
                layoutw = layout[z, :, :].clone()
                msum = layoutw.sum()
                layoutw[layoutw > 0] = 1 + torch.arange(msum)
                widx = torch.cat((widx, current_offset + layoutw.T[layoutw.T > 0] - 1))
                current_offset += msum
        widx = widx
        wincs = widx * block * block
        wincs[1:] -= widx[:-1] * block * block
        wincs = wincs.view(-1, 1).repeat(1, div)
        if trans:
            wincs[:, 1:] = step
            wincs[:, 0] -= (div - 1) * step
        else:
            wincs[:, 1:] = step * block
            wincs[:, 0] -= (div - 1) * step * block
        wincs[offsets[segments > 0], 0] = widx[offsets[segments > 0]]
        wincs = wincs.view(-1)
        # adjust offset and segment size
        offsets *= 2 * div
        segments *= div
        # create header
        width = column.size(0)
        offsets += 6 * width
        header = torch.stack((offsets, segments, column, depth, lockid, maxid), dim=1).view(-1).contiguous()
        incs = torch.stack((xincs, wincs), dim=1).view(-1).contiguous()
        incs = torch.cat((incs, torch.zeros(2, device=incs.device, dtype=incs.dtype)))
        # create lut
        lut = torch.cat((header, incs))
        lut = lut.type(torch.int32).to(device)
        # create locks
        num_locks = max(1, lockid.max())
        return lut, num_locks, width, None

    @staticmethod
    def _dds_matmul(a, b, trans_a, trans_b, trans_c, spdims, block, lut, num_locks, width, packs):
        # shapes / dtypes
        AS0 = a.size(0)
        AS1 = a.size(1)
        AS2 = a.size(3 if trans_a else 2)
        AS3 = a.size(2 if trans_a else 3)
        BS0 = spdims[0]
        BS1 = block * spdims[2 if trans_b else 1]
        BS2 = block * spdims[1 if trans_b else 2]
        dtype = a.dtype
        # kernel
        key = (block, a.device, a.dtype, b.dtype, trans_a, trans_b, trans_c)
        if key not in _matmul.dds_cache:
            defines = {
                'TM': 128,
                'TN': block,
                'TK': 16,
                'BLOCK': block,
                'TYPE': dtype,
                'STRIDE_AM': 1 if trans_a else 'lda',
                'STRIDE_AK': 'lda' if trans_a else 1,
                'STRIDE_BN': block if trans_b else 1,
                'STRIDE_BK': 1 if trans_b else block,
                'STRIDE_CM': '1' if trans_c else 'ldc',
                'STRIDE_CN': 'ldc' if trans_c else '1',
                'NAME': 'dds_kernel',
                'DDS': True
            }
            _matmul.dds_cache[key] = triton.kernel(src, device=a.device, defines=defines)
        kernel = _matmul.dds_cache[key]
        # output
        CS0 = AS0
        CS1 = AS1
        CS2 = BS2 if trans_c else AS2
        CS3 = AS2 if trans_c else BS2
        locks = _matmul.get_locks(2 * AS0 * AS2 // 32 * num_locks, a.device)
        c = torch.empty((CS0, CS1, CS2, CS3), dtype=dtype, device=a.device)
        kernel(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            a.stride(2),
            block,
            c.stride(2),
            a.stride(0),
            b.stride(0),
            c.stride(0),
            a.stride(1),
            b.stride(1),
            c.stride(1),
            AS2,
            BS2,
            0,
            0,
            lut.data_ptr(),
            locks.data_ptr(),
            num_locks,
            grid=lambda opt: [width, triton.cdiv(AS2, opt.TM), AS0]
        )
        return c

    @staticmethod
    def _dsd_matmul(a, b, trans_a, trans_b, trans_c, spdims, block, lut, num_locks, width, packs):
        # shapes / dtypes
        AS0 = spdims[0]
        AS1 = block * spdims[2 if trans_a else 1]
        AS2 = block * spdims[1 if trans_a else 2]
        BS0 = b.size(0)
        BS1 = b.size(1)
        BS2 = b.size(3 if trans_b else 2)
        BS3 = b.size(2 if trans_b else 3)
        dtype = a.dtype
        # kernel
        key = (block, a.device, a.dtype, b.dtype, trans_a, trans_b, trans_c)
        if key not in _matmul.dsd_cache:
            defines = {
                'TM': block,
                'TN': 128,
                'TK': 16,
                'BLOCK': block,
                'TYPE': dtype,
                'STRIDE_AM': 1 if trans_a else block,
                'STRIDE_AK': block if trans_a else 1,
                'STRIDE_BN': 'ldb' if trans_b else '1',
                'STRIDE_BK': '1' if trans_b else 'ldb',
                'STRIDE_CM': '1' if trans_c else 'ldc',
                'STRIDE_CN': 'ldc' if trans_c else '1',
                'NAME': 'dsd_kernel',
                'DSD': True
            }
            _matmul.dsd_cache[key] = triton.kernel(src, device=a.device, defines=defines)
        kernel = _matmul.dsd_cache[key]
        # output
        CS0 = BS0
        CS1 = BS1
        CS2 = BS3 if trans_c else AS1
        CS3 = AS1 if trans_c else BS3
        locks = _matmul.get_locks(2 * BS0 * BS3 // 32 * num_locks, a.device)
        c = torch.empty((CS0, CS1, CS2, CS3), dtype=dtype, device=a.device)
        kernel(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            block,
            b.stride(2),
            c.stride(2),
            a.stride(0),
            b.stride(0),
            c.stride(0),
            a.stride(1),
            b.stride(1),
            c.stride(1),
            BS3,
            AS1,
            0,
            0,
            lut.data_ptr(),
            locks.data_ptr(),
            num_locks,
            grid=lambda opt: [width, triton.cdiv(BS3, opt.TN), BS0]
        )
        return c

    fn = {'sdd': _sdd_matmul.__get__(object), 'dsd': _dsd_matmul.__get__(object), 'dds': _dds_matmul.__get__(object)}

    @staticmethod
    def forward(
        ctx, a, b, trans_a, trans_b, trans_c, mode, spdims, block, c_lut, c_num_locks, c_width, c_packs, da_lut, da_num_locks,
        da_width, da_packs, db_lut, db_num_locks, db_width, db_packs
    ):
        c = _matmul.fn[mode](a, b, trans_a, trans_b, trans_c, spdims, block, c_lut, c_num_locks, c_width, c_packs)
        # save for backward
        ctx.save_for_backward(a, b)
        ctx.da_num_locks = da_num_locks
        ctx.da_lut = da_lut
        ctx.da_width = da_width
        ctx.da_packs = da_packs
        ctx.db_lut = db_lut
        ctx.db_num_locks = db_num_locks
        ctx.db_width = db_width
        ctx.db_packs = db_packs
        ctx.mode = mode
        ctx.spdims = spdims
        ctx.block = block
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        return c

    @staticmethod
    def backward(ctx, dc):
        # saved for backward
        a, b = ctx.saved_tensors
        mode = ctx.mode
        # gradients w.r.t. a
        if ctx.needs_input_grad[0]:
            mode_da = mode[1] + mode[0] + mode[2]
            da = _matmul.fn[mode_da](
                dc, b, False, not ctx.trans_b, ctx.trans_a, ctx.spdims, ctx.block, ctx.da_lut, ctx.da_num_locks, ctx.da_width,
                ctx.da_packs
            )
        # gradients w.r.t. b
        if ctx.needs_input_grad[1]:
            mode_db = mode[2] + mode[1] + mode[0]
            db = _matmul.fn[mode_db](
                a, dc, not ctx.trans_a, False, ctx.trans_b, ctx.spdims, ctx.block, ctx.db_lut, ctx.db_num_locks, ctx.db_width,
                ctx.db_packs
            )
        return da, db, None, None, None,\
               None, None, None, None,\
               None, None, None, None, None, None,\
               None, None, None, None, None, None,\
               None, None, None, None, None, None


class matmul:
    def make_lut(self, dtype, device):
        key = (dtype, device)
        if key in self.lut_cache:
            return self.lut_cache[key]
        # C look-up table
        layout, block = self.layout, self.block
        step = 16
        if self.mode == 'sdd':
            c_lut, c_num_locks, c_width, c_packs = _matmul.make_sdd_lut(layout, block, dtype, device)
        elif self.mode == 'dsd':
            c_lut, c_num_locks, c_width, c_packs = _matmul.make_dxx_lut(layout, block, step, not self.trans_a, device)
        elif self.mode == 'dds':
            c_lut, c_num_locks, c_width, c_packs = _matmul.make_dxx_lut(layout, block, step, self.trans_b, device)
        # DA look-up table
        if self.mode == 'sdd':
            da_lut, da_num_locks, da_width, da_packs = _matmul.make_dxx_lut(layout, block, step, True, device)
        elif self.mode == 'dsd':
            da_lut, da_num_locks, da_width, da_packs = _matmul.make_sdd_lut(layout, block, dtype, device)
        elif self.mode == 'dds':
            da_lut, da_num_locks, da_width, da_packs = _matmul.make_dxx_lut(layout, block, step, not self.trans_b, device)
        # DB look-up table
        if self.mode == 'sdd':
            db_lut, db_num_locks, db_width, db_packs = _matmul.make_dxx_lut(layout, block, step, False, device)
        elif self.mode == 'dsd':
            db_lut, db_num_locks, db_width, db_packs = _matmul.make_dxx_lut(layout, block, step, self.trans_a, device)
        elif self.mode == 'dds':
            db_lut, db_num_locks, db_width, db_packs = _matmul.make_sdd_lut(layout, block, dtype, device)
        self.lut_cache[key] = (c_lut, c_num_locks, c_width, c_packs,\
                               da_lut, da_num_locks, da_width, da_packs,\
                               db_lut, db_num_locks, db_width, db_packs)
        return self.lut_cache[key]

    def __init__(self, layout, block, mode, trans_a=False, trans_b=False):
        if mode not in ['sdd', 'dsd', 'dds']:
            raise NotImplementedError('Supported modes are: sdd, dsd, dds')
        # look-up table cache
        self.lut_cache = dict()
        # attributes
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.mode = mode
        self.spdims = layout.shape
        self.block = block
        self.layout = layout

    # pad shapes of a tensor to make it
    # compatible with kernel calls
    @staticmethod
    def _pad_shape(x, is_sparse):
        max_dim = 3 if is_sparse else 4
        for i in range(max_dim - x.dim()):
            x = x.unsqueeze(0)
        return x

    def __call__(self, a, b):
        c_lut, c_num_locks, c_width, c_packs,\
        da_lut, da_num_locks, da_width, da_packs,\
        db_lut, db_num_locks, db_width, db_packs = self.make_lut(a.dtype, a.device)
        # pad shapes with ones
        a = matmul._pad_shape(a, self.mode == 'dsd')
        b = matmul._pad_shape(b, self.mode == 'dds')
        # execute
        c = _matmul.apply(
            a, b, self.trans_a, self.trans_b, False, self.mode, self.spdims, self.block, c_lut, c_num_locks, c_width, c_packs,
            da_lut, da_num_locks, da_width, da_packs, db_lut, db_num_locks, db_width, db_packs
        )
        return c
