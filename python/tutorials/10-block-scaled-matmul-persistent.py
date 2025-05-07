import argparse

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_block_scaling():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    kernel_name = kernel.name + ws_str

    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel_persistent(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    output_type: tl.constexpr,
    ELEM_PER_BYTE_A: tl.constexpr,  #
    ELEM_PER_BYTE_B: tl.constexpr,  #
    VEC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    rep_m: tl.constexpr,
    rep_n: tl.constexpr,
    rep_k: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    GROUP_SIZE_M: tl.constexpr = 8
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    tile_id_c = start_pid - NUM_SMS

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    for tile_id in tl.range(start_pid, num_tiles, tl.num_programs(0), flatten=not WARP_SPECIALIZE, disallow_acc_multi_buffer=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        offs_k_a = 0
        offs_k_b = 0

        offs_scale_m = pid_m * rep_m
        offs_scale_n = pid_n * rep_n
        offs_scale_k = 0

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in tl.range(0, k_tiles, num_stages=NUM_STAGES, disallow_acc_multi_buffer=True):
            a = a_desc.load([offs_am, offs_k_a])
            b = b_desc.load([offs_bn, offs_k_b])
            scale_a = a_scale_desc.load([offs_scale_m, offs_scale_k, 0, 0])
            scale_b = b_scale_desc.load([offs_scale_n, offs_scale_k, 0, 0])

            scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
            scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

            if MIXED_PREC:
                accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
            elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
                accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
            else:
                accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

            offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
            offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
            offs_scale_k += rep_k

        if not WARP_SPECIALIZE:
            tile_id_c += NUM_SMS
            pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
            offs_am_c = pid_m * BLOCK_M
            offs_bn_c = pid_n * BLOCK_N
        else:
            tile_id_c = tile_id
            offs_am_c = offs_am
            offs_bn_c = offs_bn

        c_desc.store([offs_am_c, offs_bn_c], accumulator.to(output_dtype))


def block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, dtype_dst, M, N, K, rep_m, rep_n, rep_k, configs, ws=False):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    c_desc = TensorDescriptor.from_tensor(
        output, ([configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"]]))

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (
        min(
            NUM_SMS,
            triton.cdiv(M, configs["BLOCK_SIZE_M"]) * triton.cdiv(N, configs["BLOCK_SIZE_N"]),
        ),
    )

    out = block_scaled_matmul_kernel_persistent[grid](
        a_desc,
        a_scale,
        b_desc,
        b_scale,
        c_desc,
        M,
        N,
        K,
        dtype_dst,
        configs["ELEM_PER_BYTE_A"],
        configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        rep_m,
        rep_n,
        rep_k,
        configs["num_stages"],
        NUM_SMS,
        WARP_SPECIALIZE=ws,
        enable_warp_specialization=ws,
    )
#     print(out.asm["ttgir"])
    return output


def initialize_block_scaled(M, N, K, block_scale_type="nvfp4"):
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    device = "cuda"

    a_ref = MXFP4Tensor(size=(M, K), device=device).random()
    b_ref = MXFP4Tensor(size=(N, K), device=device).random()

    if block_scale_type in ["mxfp8", "mixed"]:
        a_ref = a_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
    else:
        # Pack two fp4 elements per byte along K
        a = a_ref.to_packed_tensor(dim=1)

    if block_scale_type == "mxfp8":
        b_ref = b_ref.to(torch.float32)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=1)

    b_ref = b_ref.to(torch.float32).T

    a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]

    epsilon = 1e-8
    a_scale = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale = torch.rand(b_scale_shape, device=device) + epsilon
    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale
    elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

    a_scale_desc = TensorDescriptor.from_tensor(a_scale, [rep_m, rep_k] + list(a_scale_shape)[-2:])
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, [rep_n, rep_k] + list(b_scale_shape)[-2:])

    a_scale_ref = a_scale_ref.to(torch.float32)
    b_scale_ref = b_scale_ref.to(torch.float32)

    def unpack_scale(packed):
        packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
        num_chunk_m, num_chunk_k, _, _, _ = packed.shape
        return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

    a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_output = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
        "num_stages": 3,
    }
    return a_desc, a_scale_desc, b_desc, b_scale_desc, ref_output, rep_m, rep_n, rep_k, configs


def validate_block_scaled(M, N, K, block_scale_type="nvfp4", ws=False):
    a_desc, a_scale_desc, b_desc_desc, b_scale_desc, reference, rep_m, rep_n, rep_k, configs = initialize_block_scaled(
        M, N, K, block_scale_type
    )
    output = block_scaled_matmul(
        a_desc,
        a_scale_desc,
        b_desc_desc,
        b_scale_desc,
        torch.float16,
        M,
        N,
        K,
        rep_m,
        rep_n,
        rep_k,
        configs,
        ws=ws,
    )
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)
    print(f"✅ (pass {block_scale_type})")


def bench_block_scaled(K, block_scale_type="nvfp4", reps=10, ws=False):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}")

    a_desc, a_scale_desc, b_desc_desc, b_scale_desc, _, rep_m, rep_n, rep_k, configs = initialize_block_scaled(
        M, N, K, block_scale_type
    )
    _ = block_scaled_matmul(
        a_desc,
        a_scale_desc,
        b_desc_desc,
        b_scale_desc,
        torch.float16,
        M,
        N,
        K,
        rep_m,
        rep_n,
        rep_k,
        configs,
        ws=ws,
    )

    proton.activate(0)
    for _ in range(reps):
        _ = block_scaled_matmul(
            a_desc,
            a_scale_desc,
            b_desc_desc,
            b_scale_desc,
            torch.float16,
            M,
            N,
            K,
            rep_m,
            rep_n,
            rep_k,
            configs,
            ws=ws,
        )
    proton.deactivate(0)
    print("Done benchmarking")


def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer

    metric_names = ["time/ms"]
    metric_names = ["tflop/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", action="store_true")
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--format", type=str, choices=["mxfp4", "nvfp4", "mxfp8", "mixed"], default="nvfp4")
    args = parser.parse_args()

    if not supports_block_scaling():
        print("⛔ This example requires GPU support for block scaled matmul")
    else:
        torch.manual_seed(42)

        validate_block_scaled(8192, 8192, 8192, block_scale_type=args.format, ws=args.ws)

        if args.bench:
            if args.ws:
                file_name = f"block_scaled_matmul_ws_{args.format}"
            else:
                file_name = f"block_scaled_matmul_swp_{args.format}"

            proton.start(file_name, hook="triton")
            proton.deactivate()
            for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
                bench_block_scaled(K, reps=10000, block_scale_type=args.format, ws=args.ws)
            proton.finalize()
            show_profile(file_name)
