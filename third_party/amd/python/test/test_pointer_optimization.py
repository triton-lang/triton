import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip

if not is_hip():
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    "input_size, input_offset, block_size, num_blocks, increment, initial_offset, expect_optimization",
    [(256 * 32, 0, 256, 32, 256, 0, True),  # step forward
     (256 * 33, 0, 256, 32, -256, 256 * 32, True),  # step backward
     (256 * 32, 0, 256, 32, -256, 256 * 31, False),  # step backward #2, analysis is too conservative
     (2**29 - 1, 0, 256, 2, 1, 2**29 - 1 - 256 - 1, True),  # go near the upper limit of befferized tensor size
     (2048, 1024, 256, 2, -1024, 0, False),  # on first iteration offset is positive, on second becomes negative
     ])
def test_buffer_op_ptr_optimization(input_size, input_offset, block_size, num_blocks, increment, initial_offset,
                                    expect_optimization, device):
    """ Test verifies that increments are going to base pointer instead of offset tensor when possible and this transformation is correct:

    expect TTGIR looks like this:

        scf.for (%X) {
        %x = amdg.buffer_load %X[%offset]
        ...

        %X_1 = tt.addptr %X, %step
        scf.yield %X_1
        }

    instead of

        scf.for (%offset) {
        %x = amdg.buffer_load %X[%offset]
        ...

        %offset_1 = arith.addi %offset, %step
        scf.yield %offset_1
        }
    """

    @triton.jit
    def kernel(X, Y, BLOCK_SIZE: tl.constexpr, NUM_BLOCKS: tl.constexpr, INCREMENT: tl.constexpr,
               INITIAL_OFFSET: tl.constexpr):
        offset = tl.arange(0, BLOCK_SIZE) + INITIAL_OFFSET
        y = tl.zeros([BLOCK_SIZE], tl.float32)
        for i in range(NUM_BLOCKS):
            Xs = X + offset
            x = tl.load(Xs)
            y += x
            offset += INCREMENT
        Ys = Y + tl.arange(0, BLOCK_SIZE)
        tl.store(Ys, y)

    def check_ir(ttgir, expect_optimization):
        optimized = False
        buffer_op_present = False
        for line in ttgir.split("\n"):
            if "buffer_load" in line:
                buffer_op_present = True
            if "addptr" in line:
                optimized = True
        assert optimized == expect_optimization
        assert buffer_op_present

    X = torch.randn((input_size, ), device=device, dtype=torch.float32)
    Y = torch.randn((block_size, ), device=device, dtype=torch.float32)

    ref_output = torch.zeros_like(Y)
    for i in range(num_blocks):
        for j in range(block_size):
            ref_output[j] += X[(input_offset + initial_offset + increment * i + j) % input_size]

    grid = (1, )

    pgm = kernel[grid](X[input_offset:], Y, block_size, num_blocks, increment, initial_offset, num_warps=1)

    check_ir(pgm.asm["ttgir"], expect_optimization)

    if expect_optimization:
        assert torch.allclose(Y, ref_output, rtol=1e-3, atol=1e-2)
