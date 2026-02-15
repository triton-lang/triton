# ruff: noqa: E402
"""
Common utilities for GFX1250 GEMM kernels.

This module contains shared functions, classes, and utilities used by both
persistent and StreamK GEMM implementations.
"""

from triton.experimental import gluon
from triton.language.core import _aggregate as aggregate
import triton.experimental.gluon.language as ttgl


@gluon.jit
def chiplet_transform(pid, num_workgroups, num_xcds: ttgl.constexpr):
    """
    Basic chiplet transformation for multi-XCD AMD GPUs.

    Transforms program ID to distribute work evenly across chiplets (XCDs).
    Each XCD gets a contiguous range of work items.
    """
    xcd = pid % num_xcds
    pos_in_xcd = pid // num_xcds
    min_per_xcd = num_workgroups // num_xcds
    extra_sms = num_workgroups % num_xcds
    offset = xcd * min_per_xcd + min(xcd, extra_sms)
    return offset + pos_in_xcd


@gluon.jit
def chiplet_transform_chunked(pid, num_workgroups, num_xcds: ttgl.constexpr, chunk_size: ttgl.constexpr):
    """
    Chunked chiplet transformation for improved memory locality.

    Groups work items into chunks of size `chunk_size` per XCD, ensuring
    adjacent work items within a chunk are on the same chiplet for better
    cache utilization and memory bandwidth.
    """
    # Simplified to reduce SGPR temporaries
    if pid >= (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid

    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return (local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size)


@gluon.jit
def remap_xcd_chunked(pid, grid_mn, num_xcds: ttgl.constexpr = 8, chunk_size: ttgl.constexpr = 2):
    """
    XCD remapping with chunked distribution (alternative implementation).
    Similar to chiplet_transform_chunked but with default parameters
    """
    # Compute current XCD and local PID
    xcd = pid % num_xcds
    # Distribute the modulo pids in round robin
    if pid > (grid_mn // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid
    local_pid = pid // num_xcds
    # Calculate chunk index and position within chunk
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size
    # Calculate new PID
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


@gluon.constexpr_function
def create_shared_layouts(BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                          TRANSPOSE_B: ttgl.constexpr):

    SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                [1, 0])
    if not TRANSPOSE_B:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K, BLOCK_N],
                                                                                    [1, 0])
    else:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K],
                                                                                    [1, 0])

    return (SHARED_LAYOUT_A, SHARED_LAYOUT_B)


def build_gemm_layouts(BLOCK_M, BLOCK_N, BLOCK_K, cga_layout_a, cga_layout_b, cga_layout_c, WARP_BASES, TRANSPOSE_B):
    """
    Build all layouts for the GEMM kernel.
    """
    # If TRANSPOSE_B we need to transpose each basis vector of the CGALayout for the
    # shared allocation because the permute will transpose the basis vectors before we
    # load them for wmmas.
    if TRANSPOSE_B:
        # Transpose each basis vector: [a, b] -> [b, a]
        cga_layout_b_transposed = tuple([tuple([row[1], row[0]]) for row in cga_layout_b])
    else:
        cga_layout_b_transposed = cga_layout_b

    SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                [1, 0], cga_layout_a)
    if not TRANSPOSE_B:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K, BLOCK_N],
                                                                                    [1, 0], cga_layout_b_transposed)
    else:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K],
                                                                                    [1, 0], cga_layout_b_transposed)

    WMMA_LAYOUT_A = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32], cga_layout_a)
    WMMA_LAYOUT_B = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32], cga_layout_b)
    ACCUMULATOR_LAYOUT = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32], cga_layout_c)
    OPERAND_LAYOUT_A = ttgl.DotOperandLayout(0, WMMA_LAYOUT_A, 8)
    OPERAND_LAYOUT_B = ttgl.DotOperandLayout(1, WMMA_LAYOUT_B, 8)

    return SHARED_LAYOUT_A, SHARED_LAYOUT_B, ACCUMULATOR_LAYOUT, OPERAND_LAYOUT_A, OPERAND_LAYOUT_B


@gluon.jit
def create_tensor_descriptors(a_ptr, b_ptr, off_am, off_bn, stride_am, stride_ak, stride_bn, stride_bk,
                              shared_layout_a: ttgl.constexpr, shared_layout_b: ttgl.constexpr, M: ttgl.constexpr,
                              N: ttgl.constexpr, K: ttgl.constexpr, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                              BLOCK_K: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr + off_am, shape=(M, K),
                                                         strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K),
                                                         layout=shared_layout_a)
    if not TRANSPOSE_B:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + off_bn, shape=(K, N),
                                                             strides=(stride_bk, stride_bn),
                                                             block_shape=(BLOCK_K, BLOCK_N), layout=shared_layout_b)
    else:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + off_bn, shape=(N, K),
                                                             strides=(stride_bn, stride_bk),
                                                             block_shape=(BLOCK_N, BLOCK_K), layout=shared_layout_b)

    return a_desc, b_desc


@gluon.jit
def issue_l2_prefetches(distance, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K: ttgl.constexpr,
                        TRANSPOSE_B: ttgl.constexpr, pred=True):
    """
    Creates L2 prefetch for iteration `producer + distance`.
    """
    if distance == 0:
        return

    prefetch_iteration = producer + distance
    ttgl.amd.gfx1250.tdm.prefetch(a_desc, [off_am, prefetch_iteration * BLOCK_K], pred=pred)
    if not TRANSPOSE_B:
        ttgl.amd.gfx1250.tdm.prefetch(b_desc, [prefetch_iteration * BLOCK_K, off_bn], pred=pred)
    else:
        ttgl.amd.gfx1250.tdm.prefetch(b_desc, [off_bn, prefetch_iteration * BLOCK_K], pred=pred)


@gluon.jit
def issue_l2_prefetches_prologue(distance, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K: ttgl.constexpr,
                                 NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr, pred=True):
    """
    Creates prefetches for iterations [NUM_BUFFERS, distance - NUM_BUFFERS) or no prefetches if distance <= NUM_BUFFERS.
    This skips iterations which are preloaded in the prologue because prefetching them does not make sense for GEMMs.
    """
    if distance <= NUM_BUFFERS:
        return

    for i in ttgl.static_range(NUM_BUFFERS - distance):
        issue_l2_prefetches(distance + NUM_BUFFERS + i, producer, a_desc, b_desc, 0, 0, BLOCK_K, TRANSPOSE_B, pred)


@gluon.jit
def issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K: ttgl.constexpr,
                NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr, pred=1):
    # pred is a hardware predicate passed to async_load for conditional execution without branch divergence
    # Convert boolean pred to i32 for hardware predicate (i1 -> i32)
    pred_i32 = pred.to(ttgl.int32) if hasattr(pred, 'to') else pred
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, producer * BLOCK_K], a_buffer.index(producer % NUM_BUFFERS),
                                    pred=pred_i32)
    if not TRANSPOSE_B:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [producer * BLOCK_K, off_bn], b_buffer.index(producer % NUM_BUFFERS),
                                        pred=pred_i32)
    else:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, producer * BLOCK_K], b_buffer.index(producer % NUM_BUFFERS),
                                        pred=pred_i32)
    producer += 1
    return producer


@gluon.jit
def issue_wmma(consumer, a_buffer, a_layout: ttgl.constexpr, b_buffer, b_layout: ttgl.constexpr, accumulator,
               wait_producers_cnt, NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):
    """
    For multi-CTA configurations, we want warps within the CGA (cluster) to stay temporally aligned so we can
    multicast data to multiple CTAs.
    We do this by signaling the cluster barrier before `async_wait` (which inserts a CTA barrier), then waiting
    for the cluster barrier to complete. This keeps warps of a CGA within one iteration of each other.
    It can also improve latency hiding by overlapping the cluster and CTA barriers.
    """
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    if num_ctas > 1:
        ttgl.amd.gfx1250.cluster.arrive()

    ttgl.amd.gfx1250.tdm.async_wait(wait_producers_cnt)

    if num_ctas > 1:
        ttgl.amd.gfx1250.cluster.wait()

    a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=b_layout)
    else:
        b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=b_layout)

    accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)
    consumer += 1
    return consumer, accumulator


@gluon.jit
def lds_subtile_load(consumer, start, a_buffer, a_layout: ttgl.constexpr, b_buffer, b_layout: ttgl.constexpr,
                     NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr, SUBTILE_LEN: ttgl.constexpr):

    index = consumer % NUM_BUFFERS
    a = a_buffer.index(index).slice(start, SUBTILE_LEN, 1).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(index).slice(start, SUBTILE_LEN, 0).load(layout=b_layout)
    else:
        b = b_buffer.index(index).slice(start, SUBTILE_LEN, 1).permute([1, 0]).load(layout=b_layout)

    return a, b


@gluon.jit
def lds_load(consumer, a_buffer, a_layout: ttgl.constexpr, b_buffer, b_layout: ttgl.constexpr,
             NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):
    """Load A and B tiles from shared memory (LDS) into registers."""
    a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=b_layout)
    else:
        b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=b_layout)

    consumer += 1
    return consumer, a, b


@gluon.jit
def issue_wmma_compute(a, b, accumulator):
    """Perform WMMA computation on pre-loaded operands."""
    accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)
    return accumulator


@aggregate
class TileScheduler:
    """
    Tile Scheduler

    Stores essential tile scheduling state. Values like iters_per_tile
    are recomputed when needed to reduce live register pressure.

    Stored fields (4 SGPRs total):
    - num_pid_m: Number of tiles in M dimension
    - num_pid_n: Number of tiles in N dimension
    - total_full_tiles: Number of tiles processed in persistent mode
    - num_streamk_tiles: Number of tiles for StreamK processing
    """
    num_pid_m: ttgl.tensor
    num_pid_n: ttgl.tensor
    total_full_tiles: ttgl.tensor
    num_streamk_tiles: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, num_pid_m, num_pid_n, total_full_tiles, num_streamk_tiles):
        self.num_pid_m = num_pid_m
        self.num_pid_n = num_pid_n
        self.total_full_tiles = total_full_tiles
        self.num_streamk_tiles = num_streamk_tiles

    @gluon.jit
    def initialize(M, N, K, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                   STREAMK_TILES: ttgl.constexpr):
        """Initialize scheduler - stores essential state for tile scheduling."""
        num_pid_m = ttgl.cdiv(M, BLOCK_M)
        num_pid_n = ttgl.cdiv(N, BLOCK_N)
        total_tiles = num_pid_m * num_pid_n
        total_full_tiles = total_tiles - STREAMK_TILES
        # Convert constexpr to tensor for storage
        num_streamk_tiles = total_tiles - total_full_tiles
        return TileScheduler(num_pid_m, num_pid_n, total_full_tiles, num_streamk_tiles)

    @gluon.jit
    def get_num_tiles(self):
        """Return total number of full tiles for persistent loop."""
        return self.total_full_tiles

    @gluon.jit
    def get_num_full_tiles(self):
        return self.total_full_tiles

    @gluon.jit
    def get_num_streamk_tiles(self):
        return self.num_streamk_tiles

    @gluon.jit
    def get_pid(self):
        """Return current program ID."""
        return ttgl.program_id(axis=0)

    @gluon.jit
    def get_num_sms(self):
        """Return total number of SMs/CUs available."""
        return ttgl.num_programs(axis=0)

    @gluon.jit
    def get_swizzled_tile_coords(self, tile_id, GROUP_SIZE_M: ttgl.constexpr):
        """Get swizzled tile coordinates using stored num_pid_m and num_pid_n."""
        num_pid_in_group = GROUP_SIZE_M * self.num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = ttgl.minimum(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    @gluon.jit
    def apply_chiplet_transform(self, pid, num_sms, num_xcds: ttgl.constexpr):
        """Apply basic chiplet transformation to a program ID."""
        return chiplet_transform(pid, num_sms, num_xcds)

    @gluon.jit
    def apply_chiplet_transform_chunked(self, pid, num_sms, num_xcds: ttgl.constexpr, chunk_size: ttgl.constexpr):
        """Apply chunked chiplet transformation for improved cache locality."""
        return chiplet_transform_chunked(pid, num_sms, num_xcds, chunk_size)
