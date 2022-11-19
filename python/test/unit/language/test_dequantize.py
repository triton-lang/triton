# flake8: noqa: F821,F841

import random

import torch

import triton
import triton.language as tl


@triton.jit
def dequantize_kernel_int8(output_ptr, input_ptr, size, BLOCK_SIZE: tl.constexpr):
    w_offsets = tl.arange(0, BLOCK_SIZE // 4)
    mask = w_offsets < (size // 4)
    input_ptrs = input_ptr + 1 + w_offsets
    input = tl.load(input_ptrs, mask=mask, other=0)
    scale_shift = tl.load(input_ptr)
    scale = (scale_shift & 65535).to(tl.int16).to(tl.float16, bitcast=True)
    shift = (scale_shift >> 16).to(tl.int16).to(tl.float16, bitcast=True)
    output = tl.dequantize(input, scale, shift, 8)
    offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = tl.multiple_of(output_ptr + offsets, 4)
    tl.store(output_ptrs, output, mask=offsets < size)


@triton.jit
def dequantize_kernel_scale_shift_int8(
    output_ptr, input_ptr, scale_ptr, shift_ptr, size, BLOCK_SIZE: tl.constexpr
):
    w_offsets = tl.arange(0, BLOCK_SIZE // 4)
    mask = w_offsets < (size // 4)
    input_ptrs = tl.multiple_of(input_ptr + w_offsets, 1)
    input = tl.load(input_ptrs, mask=mask, other=0)
    scale = tl.load(scale_ptr)
    shift = tl.load(shift_ptr)
    output = tl.dequantize(input, scale, shift, 8)
    offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = tl.multiple_of(output_ptr + offsets, 4)
    tl.store(output_ptrs, output, mask=offsets < size)


@triton.jit
def dequantize_kernel_int4(output_ptr, input_ptr, size, BLOCK_SIZE: tl.constexpr):
    w_offsets = tl.arange(0, BLOCK_SIZE // 8)
    mask = w_offsets < (size // 8)
    input_ptrs = input_ptr + 1 + w_offsets
    input = tl.load(input_ptrs, mask=mask, other=0)
    scale_shift = tl.load(input_ptr)
    scale = (scale_shift & 65535).to(tl.int16).to(tl.float16, bitcast=True)
    shift = (scale_shift >> 16).to(tl.int16).to(tl.float16, bitcast=True)
    output = tl.dequantize(input, scale, shift, 4)
    offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = tl.multiple_of(output_ptr + offsets, 8)
    tl.store(output_ptrs, output, mask=offsets < size)


@triton.jit
def dequantize_kernel_scale_shift_int4(
    output_ptr, input_ptr, scale_ptr, shift_ptr, size, BLOCK_SIZE: tl.constexpr
):
    w_offsets = tl.arange(0, BLOCK_SIZE // 8)
    mask = w_offsets < (size // 8)
    input_ptrs = tl.multiple_of(input_ptr + w_offsets, 1)
    input = tl.load(input_ptrs, mask=mask, other=0)
    scale = tl.load(scale_ptr)
    shift = tl.load(shift_ptr)
    output = tl.dequantize(input, scale, shift, 4)
    offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = tl.multiple_of(output_ptr + offsets, 8)
    tl.store(output_ptrs, output, mask=offsets < size)


@triton.jit
def dequantize_kernel_int2(output_ptr, input_ptr, size, BLOCK_SIZE: tl.constexpr):
    w_offsets = tl.arange(0, BLOCK_SIZE // 8)
    mask = w_offsets < (size // 8)
    input_ptrs = tl.multiple_of(input_ptr + 2 + w_offsets, 1)
    input = tl.load(input_ptrs, mask=mask, other=0)
    scale = tl.load(input_ptr).to(tl.float16, bitcast=True)
    shift = tl.load(input_ptr + 1).to(tl.float16, bitcast=True)
    output = tl.dequantize(input, scale, shift, 2)
    offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = tl.multiple_of(output_ptr + offsets, 8)
    tl.store(output_ptrs, output, mask=offsets < size)


@triton.jit
def dequantize_kernel_scale_shift_int2(
    output_ptr, input_ptr, scale_ptr, shift_ptr, size, BLOCK_SIZE: tl.constexpr
):
    w_offsets = tl.arange(0, BLOCK_SIZE // 8)
    mask = w_offsets < (size // 8)
    input_ptrs = tl.multiple_of(input_ptr + w_offsets, 1)
    input = tl.load(input_ptrs, mask=mask, other=0)
    scale = tl.load(scale_ptr)
    shift = tl.load(shift_ptr)
    output = tl.dequantize(input, scale, shift, 2)
    offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = tl.multiple_of(output_ptr + offsets, 8)
    tl.store(output_ptrs, output, mask=offsets < size)


def test_dequantize_int8() -> None:
    for i in range(10):
        if i < 5:
            size = random.randrange(16, 128, 4)
        else:
            size = random.randrange(132, 1024, 4)
        device = torch.device(torch.cuda.current_device())

        scale_val = random.uniform(0.1, 4.0)
        shift_val = random.uniform(-10.0, 10.0)
        scale = torch.tensor(scale_val, dtype=torch.float16, device=device)
        shift = torch.tensor(shift_val, dtype=torch.float16, device=device)
        scale_shift = torch.tensor(
            [scale_val, shift_val],
            dtype=torch.float16,
            device=device,
        ).view(torch.int32)

        input_int8 = torch.randint(
            0, 256, (size,), dtype=torch.uint8, device=device
        )
        input_int32 = input_int8.view(torch.int32)

        input = torch.cat((scale_shift, input_int32))
        expected = (input_int8 * scale + shift).to(torch.float16)

        output = torch.empty([size], dtype=torch.float16, device=device)
        block_size = max(triton.next_power_of_2(size), 128)
        grid = (1,)
        dequantize_kernel_int8[grid](
            output, input, size, BLOCK_SIZE=block_size, num_warps=1
        )
        rtol, atol = 1e-02, 1e-02
        assert torch.allclose(output, expected, rtol, atol)

        output = torch.empty([size], dtype=torch.float16, device=device)
        dequantize_kernel_scale_shift_int8[grid](
            output,
            input_int32,
            scale,
            shift,
            size,
            BLOCK_SIZE=block_size,
            num_warps=1,
        )
        assert torch.allclose(output, expected, rtol, atol)


def test_dequantize_int4() -> None:
    for i in range(10):
        if i < 5:
            size = random.randrange(16, 256, 8)
        else:
            size = random.randrange(264, 1024, 8)
        device = torch.device(torch.cuda.current_device())

        scale_val = random.uniform(0.1, 4.0)
        shift_val = random.uniform(-10.0, 10.0)
        scale = torch.tensor(scale_val, dtype=torch.float16, device=device)
        shift = torch.tensor(shift_val, dtype=torch.float16, device=device)
        scale_shift = torch.tensor(
            [scale_val, shift_val],
            dtype=torch.float16,
            device=device,
        ).view(torch.int32)

        input_int8 = torch.randint(
            0, 256, (size // 2,), dtype=torch.uint8, device=device
        )
        input_int32 = input_int8.view(torch.int32)

        input_int8_h1 = input_int8 >> 4
        input_int8_h0 = input_int8 & 15

        input_int4_val = torch.stack(
            (input_int8_h0, input_int8_h1), dim=1
        ).flatten()

        input = torch.cat((scale_shift, input_int32))
        expected = (input_int4_val * scale + shift).to(torch.float16)

        output = torch.empty([size], dtype=torch.float16, device=device)
        block_size = max(triton.next_power_of_2(size), 256)
        grid = (1,)
        dequantize_kernel_int4[grid](
            output, input, size, BLOCK_SIZE=block_size, num_warps=1
        )
        rtol, atol = 1e-02, 1e-02
        assert torch.allclose(output, expected, rtol, atol)

        output = torch.empty([size], dtype=torch.float16, device=device)
        dequantize_kernel_scale_shift_int4[grid](
            output,
            input_int32,
            scale,
            shift,
            size,
            BLOCK_SIZE=block_size,
            num_warps=1,
        )
        assert torch.allclose(output, expected, rtol, atol)


def test_dequantize_int2() -> None:
    for i in range(10):
        if i < 5:
            size = random.randrange(16, 256, 8)
        else:
            size = random.randrange(264, 1024, 8)
        device = torch.device(torch.cuda.current_device())

        scale_val = random.uniform(0.1, 4.0)
        shift_val = random.uniform(-10.0, 10.0)
        scale = torch.tensor(scale_val, dtype=torch.float16, device=device)
        shift = torch.tensor(shift_val, dtype=torch.float16, device=device)
        scale_shift = torch.tensor(
            [scale_val, shift_val],
            dtype=torch.float16,
            device=device,
        ).view(torch.int16)

        input_int8 = torch.randint(
            0, 256, (size // 4,), dtype=torch.uint8, device=device
        )
        input_int16 = input_int8.view(torch.int16)

        input_int8_q3 = input_int8 >> 6
        input_int8_q2 = (input_int8 >> 4) & 3
        input_int8_q1 = (input_int8 >> 2) & 3
        input_int8_q0 = input_int8 & 3

        input_int2_val = torch.stack(
            (input_int8_q0, input_int8_q1, input_int8_q2, input_int8_q3), dim=1
        ).flatten()

        input = torch.cat((scale_shift, input_int16))
        expected = (input_int2_val * scale + shift).to(torch.float16)

        output = torch.empty([size], dtype=torch.float16, device=device)
        block_size = max(triton.next_power_of_2(size), 256)
        grid = (1,)

        dequantize_kernel_int2[grid](
            output, input, size, BLOCK_SIZE=block_size, num_warps=1
        )
        rtol, atol = 1e-02, 1e-02
        assert torch.allclose(output, expected, rtol, atol)

        output = torch.empty([size], dtype=torch.float16, device=device)
        dequantize_kernel_scale_shift_int2[grid](
            output,
            input_int16,
            scale,
            shift,
            size,
            BLOCK_SIZE=block_size,
            num_warps=1,
        )
        assert torch.allclose(output, expected, rtol, atol)
