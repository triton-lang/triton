import triton
import triton.language as tl


@triton.jit
def _keyed_add(x, y):

    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xffff0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _count_previous(x):
    """
    Input  x : uint16[..., N]
    Output y : uint32[..., N]
    semantics : y[..., i] = sum_j((x[..., j] == x[..., i]) & (j < i))
    credits: @apgoucher
    """

    BLOCK_N: tl.constexpr = x.shape[-1]  # summation axis
    BATCHES: tl.constexpr = x.numel // BLOCK_N  # number of batches

    # reduce to two-dimensional case:
    y = tl.reshape(x, [BATCHES, BLOCK_N]).to(tl.uint32)

    tl.static_assert(BLOCK_N <= 32768, "compute_run_lengths requires axis to have length <= 32768")

    # sort (expert, position) ordered pairs to perform an argsort:
    kv_pairs = ((y << 16) | tl.arange(0, BLOCK_N)[None, :]).to(tl.uint32)
    sorted_kv_pairs = tl.sort(kv_pairs, 1)

    # compute run lengths in expert-sorted order:
    x = (sorted_kv_pairs & 0xffff0000 | 0x00000001)
    expts_and_inclusive_run_lengths = tl.associative_scan(x, 1, _keyed_add)
    exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xffff

    # undo permutation by doing another sort
    # TODO rewrite this when tl.scatter becomes available
    kv_pairs = ((sorted_kv_pairs << 16) | exclusive_run_lengths).to(tl.uint32)
    unsorted_run_lengths = tl.sort(kv_pairs) & 0xffff

    res = tl.reshape(unsorted_run_lengths, x.shape)
    return res


@triton.jit
def _compute_indx(GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx, PartialOffs, stride_pm, n_gates,
                  BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr):
    pid_m = tl.program_id(0)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + tl.arange(0, N_EXPTS_ACT * BLOCK_M)
    mask = offs < n_gates
    indx = tl.load(ExptIndx + offs, mask=mask)
    gates = tl.load(PartialOffs + pid_m * stride_pm + indx, mask=mask)
    gates += tl.reshape(_count_previous(indx), [BLOCK_M * N_EXPTS_ACT])
    gate_scal = tl.load(ExptScal + offs, mask=mask)
    tl.store(ScatterIndx + offs, gates, mask=mask)
    tl.store(GatherIndx + gates, offs, mask=mask)
    tl.store(GateScal + gates, gate_scal, mask=mask)
