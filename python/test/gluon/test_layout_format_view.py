import subprocess
from pathlib import Path

import pytest

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size


@pytest.fixture
def ttl_cli():
    repo_root = Path(__file__).parents[3]
    for binary in (repo_root / "build").glob("*/bin/triton-tensor-layout"):
        break
    else:
        pytest.skip("triton-tensor-layout binary not found")

    def run(layout_str: str, shape: list[int], use_hw_view: bool = False) -> str:
        tensor_str = "tensor<" + "x".join(str(s) for s in shape) + "xf16>"
        cmd = [str(binary), "-l", layout_str, "-t", tensor_str]
        if use_hw_view:
            cmd.append("-use-hw-view")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.split("\n", 1)[1]  # remove header

    return run


def fmt(lst):
    return "[" + ", ".join(str(x) for x in lst) + "]"


def fmt_bases(bases):
    return "[" + ", ".join("[" + ", ".join(str(x) for x in b) + "]" for b in bases) + "]"


@pytest.mark.parametrize(
    "size_per_thread,threads_per_warp,warps_per_cta,order,shape,use_hw_view",
    [([4], [32], [4], [0], [128], False),  # 1d
     ([1, 4], [4, 8], [4, 1], [1, 0], [16, 32], False),  # 2d
     ([1, 1, 4], [2, 4, 4], [2, 2, 1], [2, 1, 0], [4, 8, 16], False),  # 3d
     ([1, 4], [4, 8], [4, 1], [1, 0], [16, 32], True),  # use_hw_view
     ],
)
def test_format_view_blocked_layout(size_per_thread, threads_per_warp, warps_per_cta, order, shape, use_hw_view,
                                    ttl_cli):

    def to_ttg_attr(layout):
        return (f"#ttg.blocked<{{sizePerThread = {fmt(layout.size_per_thread)}, "
                f"threadsPerWarp = {fmt(layout.threads_per_warp)}, "
                f"warpsPerCTA = {fmt(layout.warps_per_cta)}, "
                f"order = {fmt(layout.order)}}}>")

    layout = ttgl.BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)
    if use_hw_view:
        assert layout.format_hardware_view(shape) == ttl_cli(to_ttg_attr(layout), shape, use_hw_view=True)
    else:
        assert layout.format_tensor_view(shape) == ttl_cli(to_ttg_attr(layout), shape, use_hw_view=False)


@pytest.mark.parametrize("dim,shape", [(1, [16])])
def test_format_view_slice_layout(dim, shape, ttl_cli):

    def blocked_to_ttg_attr(layout):
        return (f"#ttg.blocked<{{sizePerThread = {fmt(layout.size_per_thread)}, "
                f"threadsPerWarp = {fmt(layout.threads_per_warp)}, "
                f"warpsPerCTA = {fmt(layout.warps_per_cta)}, "
                f"order = {fmt(layout.order)}}}>")

    def to_ttg_attr(layout):
        parent_str = blocked_to_ttg_attr(layout.parent)
        return f"#ttg.slice<{{dim = {layout.dim}, parent = {parent_str}}}>"

    parent = ttgl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    layout = ttgl.SliceLayout(dim, parent)
    assert layout.format_tensor_view(shape) == ttl_cli(to_ttg_attr(layout), shape)


@pytest.mark.parametrize(
    "version,warps_per_cta,instr_shape,shape",
    [([2, 0], [4, 1], [16, 8], [64, 64])],
)
def test_format_view_nvmma_layout(version, warps_per_cta, instr_shape, shape, ttl_cli):

    def to_ttg_attr(layout):
        return (f"#ttg.nvidia_mma<{{versionMajor = {layout.version[0]}, "
                f"versionMinor = {layout.version[1]}, "
                f"warpsPerCTA = {fmt(layout.warps_per_cta)}, "
                f"instrShape = {fmt(layout.instr_shape)}}}>")

    layout = ttgl.NVMMADistributedLayout(version, warps_per_cta, instr_shape)
    assert layout.format_tensor_view(shape) == ttl_cli(to_ttg_attr(layout), shape)


@pytest.mark.parametrize("operand_index,shape", [(0, [64, 64]), (1, [32, 128])])
def test_format_view_dot_operand_layout(operand_index, shape, ttl_cli):

    def nvmma_to_ttg_attr(layout):
        return (f"#ttg.nvidia_mma<{{versionMajor = {layout.version[0]}, "
                f"versionMinor = {layout.version[1]}, "
                f"warpsPerCTA = {fmt(layout.warps_per_cta)}, "
                f"instrShape = {fmt(layout.instr_shape)}}}>")

    def to_ttg_attr(layout):
        parent_str = nvmma_to_ttg_attr(layout.parent)
        return f"#ttg.dot_op<{{opIdx = {layout.operand_index}, parent = {parent_str}, kWidth = {layout.k_width}}}>"

    parent = ttgl.NVMMADistributedLayout([2, 0], [4, 1], [16, 8])
    layout = ttgl.DotOperandLayout(operand_index, parent, 2)
    assert layout.format_tensor_view(shape) == ttl_cli(to_ttg_attr(layout), shape)


@pytest.mark.parametrize(
    "vec,per_phase,max_phase,order,shape",
    [(8, 4, 2, [1, 0], [16, 16])],
)
def test_format_view_swizzled_shared_layout(vec, per_phase, max_phase, order, shape, ttl_cli):

    def to_ttg_attr(layout):
        return (f"#ttg.swizzled_shared<{{vec = {layout.vec}, "
                f"perPhase = {layout.per_phase}, maxPhase = {layout.max_phase}, "
                f"order = {fmt(layout.order)}}}>")

    layout = ttgl.SwizzledSharedLayout(vec, per_phase, max_phase, order)
    assert layout.format_tensor_view(shape) == ttl_cli(to_ttg_attr(layout), shape)


@pytest.mark.parametrize("swizzle_byte_width,element_bitwidth,rank,transposed,shape", [(128, 16, 2, True, [64, 16])])
def test_format_view_nvmma_shared_layout(swizzle_byte_width, element_bitwidth, rank, transposed, shape, ttl_cli):

    def to_ttg_attr(layout):
        return (f"#ttg.nvmma_shared<{{swizzlingByteWidth = {layout.swizzle_byte_width}, "
                f"transposed = {str(layout.transposed).lower()}, "
                f"elementBitWidth = {layout.element_bitwidth}}}>")

    layout = ttgl.NVMMASharedLayout(swizzle_byte_width, element_bitwidth, rank, transposed)
    assert layout.format_tensor_view(shape) == ttl_cli(to_ttg_attr(layout), shape)


@pytest.mark.parametrize(
    "reg_bases,lane_bases,warp_bases,block_bases,shape",
    [
        ([[0, 1], [0, 2], [0, 4], [0, 8]],  # register
         [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],  # lane
         [[32, 0], [64, 0]],  # warp
         [],  # block
         [128, 16]),
    ],
)
def test_format_view_distributed_linear_layout(reg_bases, lane_bases, warp_bases, block_bases, shape, ttl_cli):

    def to_ttg_attr(layout):
        return (f"#ttg.linear<{{register = {fmt_bases(layout.reg_bases)}, "
                f"lane = {fmt_bases(layout.lane_bases)}, "
                f"warp = {fmt_bases(layout.warp_bases)}, "
                f"block = {fmt_bases(layout.block_bases)}}}>")

    layout = ttgl.DistributedLinearLayout(reg_bases, lane_bases, warp_bases, block_bases, shape)
    assert layout.format_tensor_view(shape) == ttl_cli(to_ttg_attr(layout), shape)


@pytest.mark.parametrize(
    "offset_bases,block_bases,alignment,shape",
    [
        ([[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8]],  # offset
         [],  # block
         16, [16, 16]),
    ],
)
def test_format_view_shared_linear_layout(offset_bases, block_bases, alignment, shape, ttl_cli):

    def to_ttg_attr(layout):
        result = f"#ttg.shared_linear<{{offset = {fmt_bases(layout.offset_bases)}"
        if layout.block_bases:
            result += f", block = {fmt_bases(layout.block_bases)}"
        result += f"}}, alignment = {layout.alignment}>"
        return result

    layout = ttgl.SharedLinearLayout(offset_bases, block_bases, alignment)
    assert layout.format_tensor_view(shape) == ttl_cli(to_ttg_attr(layout), shape)


def test_format_view_padded_shared_layout():
    layout = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [16, 64], [1, 0])
    with pytest.raises(ValueError, match="PaddedSharedLayout cannot be visualized"):
        layout.format_tensor_view([16, 64])


def test_format_view_auto_layout():
    layout = ttgl.AutoLayout()
    with pytest.raises(ValueError, match="AutoLayout cannot be visualized"):
        layout.format_tensor_view([16, 64])


def test_format_view_coalesced_layout():
    layout = ttgl.CoalescedLayout()
    with pytest.raises(ValueError, match="CoalescedLayout cannot be visualized"):
        layout.format_tensor_view([16, 64])


def test_format_view_kernel():

    @gluon.jit
    def kernel(ptr, BLOCK: ttgl.constexpr, layout: ttgl.constexpr):
        off = ttgl.arange(0, BLOCK, layout=layout)
        tensor = ttgl.load(ptr + off)
        ttgl.static_print("tensor view:\n", tensor.type.layout.format_tensor_view(tensor.shape))

    layout = ttgl.BlockedLayout([2], [THREADS_PER_WARP], [4], [0])
    x = torch.randn(512, device="cuda")
    kernel[(1, )](x, 512, layout)
