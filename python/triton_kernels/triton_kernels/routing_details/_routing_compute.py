import triton
import triton.language as tl


@triton.jit
def _routing_compute_expt_offs(ExpertHist, FinalExpertOffs, hist_size,  # histogram
                               BLOCK_N: tl.constexpr):
    loop_iterations = (hist_size + BLOCK_N - 1) // BLOCK_N
    x = tl.zeros([BLOCK_N], ExpertHist.dtype.element_ty)
    for i in range(loop_iterations):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < hist_size
        hist2 = tl.load(ExpertHist + offs_n, mask=mask_n)
        tok_starts = tl.cumsum(hist2, 0) - hist2 + x
        x += tl.sum(hist2, 0)
        tl.store(FinalExpertOffs + offs_n, tok_starts, mask=mask_n)
        offs_n += BLOCK_N


@triton.jit
def _routing_compute_indx_offs(TokensStart, PartialHist, PartialOffs, shape_pm, stride_pm, BLOCK_M: tl.constexpr):
    expt_id = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    # initialize first row of the output
    start = tl.load(TokensStart + expt_id)
    tl.store(PartialOffs + expt_id, start)
    # iterate over input data
    curr_sum = start
    for _ in range(0, shape_pm, BLOCK_M):
        offs = offs_m * stride_pm + expt_id
        curr = tl.load(PartialHist + offs, mask=offs_m < shape_pm)
        out = tl.cumsum(curr, 0) + curr_sum
        curr_sum += tl.sum(curr, 0)
        offs = (1 + offs_m) * stride_pm + expt_id
        tl.store(PartialOffs + offs, out, mask=offs_m < shape_pm - 1)
        offs_m += BLOCK_M


@triton.jit
def _keyed_add(x, y):

    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xffff0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _routing_compute_indx(GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx, PartialOffs, stride_pm, n_gates,
                          BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr):

    pid_m = tl.program_id(0)

    tl.static_assert(N_EXPTS_ACT * BLOCK_M <= 32768)

    local_offs = tl.arange(0, N_EXPTS_ACT * BLOCK_M)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + local_offs
    expert = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1).to(tl.uint32)

    # stable-sort by expert ID:
    kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + (kv_pairs & 0xffff)
    mask = expert != 0xffff
    gate_scal = tl.load(ExptScal + offs, mask=mask)

    # compute run lengths in expert-sorted order:
    x = (kv_pairs & 0xffff0000 | 0x00000001)
    expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xffff

    gates = tl.load(PartialOffs + pid_m * stride_pm + expert, mask=(expert != 0xffff))
    gates += exclusive_run_lengths

    tl.store(ScatterIndx + offs, gates, mask=mask)
    tl.store(GatherIndx + gates, offs, mask=mask)
    tl.store(GateScal + gates, gate_scal, mask=mask)


@triton.jit
def _routing_clear_bitmatrix(Bitmatrix, stride_bm, shape_bn, cutoff, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    cutoff_word = cutoff // 32
    cutoff_bit = cutoff % 32
    cutoff_mask = (1 << (cutoff_bit)) - 1
    for start_n in range(0, shape_bn, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        values = tl.load(Bitmatrix + pid_m * stride_bm + offs_n, mask=offs_n < shape_bn)
        values = tl.where(offs_n == cutoff_word, values & cutoff_mask, values)
        values = tl.where(offs_n > cutoff_word, 0, values)
        tl.store(Bitmatrix + pid_m * stride_bm + offs_n, values, mask=offs_n < shape_bn)


@triton.jit
def _routing_memset_indx(Indx, size, sentinel, BLOCK: tl.constexpr, ExpertHist, FinalExpertOffs, hist_size,
                         BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)

    if pid == 0:
        _routing_compute_expt_offs(ExpertHist, FinalExpertOffs, hist_size, BLOCK_N)
    else:
        offs = (pid - 1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < size
        tl.store(Indx + offs, sentinel, mask=mask)
