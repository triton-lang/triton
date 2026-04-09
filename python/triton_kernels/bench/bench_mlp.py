from pathlib import Path
from copy import deepcopy
import os
import triton.profiler as proton
import torch
import argparse
import triton_kernels.roofline as roofline
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.matmul import matmul, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.matmul_details.opt_flags import make_opt_flags, scoped_opt_flags_constraints
from triton_kernels.target_info import get_cdna_version
from triton_kernels.tensor_details import layout
from triton_kernels.tensor_details.layout import BlackwellMX4ValueShuffledLayout
from triton_kernels.reduce import reduce
from triton_kernels.topk import topk
from triton_kernels.tensor import make_ragged_tensor_metadata, remap_ragged_tensor_metadata  # ragged tensor
from triton_kernels.tensor import is_tma_compliant, Tensor, torch_dtype_to_dtype
from triton_kernels.distributed import convert_dp_to_ep, convert_ep_to_dp, make_expt_dict_uniform, make_expt_assignment, SymmetricMemoryPool
from triton_kernels.distributed_details.mesh import Mesh
# quantization
from triton_kernels.tensor import convert_layout, wrap_torch_tensor, FP4
from triton_kernels.numerics import InFlexData
from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE, downcast_to_mxfp


def _shuffle_mx4_weights(tensor, block_k, block_n):
    """
    Convert MX4 weights from BlackwellMXValueLayout to BlackwellMX4ValueShuffledLayout.

    Works directly with the column-major storage data, bypassing the canonical format
    round-trip which uses a different byte-level packing convention.
    """
    from triton_kernels.tensor import Storage, Tensor as TKTensor
    storage = tensor.storage
    # The stored data is column-major [E, K_packed_padded, N] with stride(-2)==1
    data = storage.data
    E = tensor.shape[0]
    K_logical = tensor.shape[-2]
    N = tensor.shape[-1]
    K_packed = K_logical // 2  # 2 FP4 values per byte
    # Trim any padding from the BlackwellMXValueLayout
    data = data[:, :K_packed, :N].contiguous()
    # Now apply the shuffled layout's tiling
    shuffled_layout = BlackwellMX4ValueShuffledLayout(block_k=block_k, block_n=block_n)
    transformation = shuffled_layout.make_transformation([E, K_logical, N], True)
    shuffled_data = transformation.swizzle_data(data)
    return TKTensor(Storage(shuffled_data, shuffled_layout), shape=list(tensor.shape), dtype=tensor.dtype)


def _infer_opt_flags(x, w, ragged_metadata, pc):
    """
    Infer opt_flags by calling make_opt_flags with the same parameters matmul would use.
    This ensures the block shapes match what the kernel will actually select.
    """
    if not isinstance(w, Tensor):
        raise TypeError("w must be a Tensor for block shape inference")
    K = w.shape[-2]
    N = w.shape[-1]
    M = x.shape[-2]
    batch_size = 1
    if not isinstance(x, Tensor):
        x = wrap_torch_tensor(x)
    # Convert out_dtype from torch.dtype to triton dtype (make_opt_flags expects .bitwidth)
    out_dtype = pc.out_dtype or x.dtype
    out_dtype = torch_dtype_to_dtype(out_dtype)
    x_transpose = x.stride(-1) != 1
    b_scale = pc.b_mx_scale
    can_use_tma = (x.numel() > 0 and is_tma_compliant(x) and w.numel() > 0 and is_tma_compliant(w)
                   and (b_scale is None or is_tma_compliant(b_scale)))
    # Respects any constraints set by the caller via scoped_opt_flags_constraints
    opt_flags = make_opt_flags(out_dtype, x.dtype, w.dtype, pc, batch_size, M, N, K, ragged_metadata, can_use_tma,
                               False,  # can_use_split_k=False for MoE
                               None,  # epilogue_effective_itemsize
                               x_transpose, False,  # has_y_acc_in
                               None,  # block_k
                               )
    return opt_flags


def was_launched_with_torchrun():
    required = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    return all(k in os.environ for k in required)


def parse_dtype(dtype):
    ret = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn, "mx4": FP4}[dtype]
    if ret == torch.float8_e4m3fn and get_cdna_version() == 3:
        ret = torch.float8_e4m3fnuz
    return ret


def quantize_weight(w, dtype, **opt):
    if dtype == torch.bfloat16:
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), None
    elif dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype)
        wq = wq.transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), None
    else:
        assert dtype == FP4, f"{dtype=}"
        w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
        if opt:
            w = convert_layout(wrap_torch_tensor(w, dtype=FP4), opt["value_layout"])
            w_scale = convert_layout(wrap_torch_tensor(w_scale), opt["scale_layout"])
        return w, InFlexData(), w_scale


def run_mlp(x_dp_local_bf16, x_dp_local_fp8,  # activations
            wg_global, bg_global, pcg,  # gate parameters / precision config
            w1_ep_local, b1_ep_local, pc1, act1,  # first matmul parameters / precision config / fused activation
            w2_ep_local, b2_ep_local, pc2,  # second matmul parameters / precision config
            n_expts_act, expt_assignment,  # expert assignment
            rank,  # distributed context
            symm_mem_pool,  # symmetric memory pool
            fc1_constraints=None, fc2_constraints=None,  # per-kernel opt_flags constraints
            ):
    # gate matrix multiplication
    l_dp_local = matmul(x_dp_local_bf16, wg_global, bg_global, precision_config=pcg)
    # active global logits (sparse)
    l_global_active = topk(l_dp_local, n_expts_act, apply_softmax=True, all_gather=True, symm_mem_pool=symm_mem_pool)
    # expert histogram, dispatch/combine indx
    active_indx = l_global_active.indx
    expt_sizes = l_global_active.mask_metadata.col_sum
    dispatch_indx = l_global_active.mask_metadata.row_sorted_indx
    combine_indx = l_global_active.mask_metadata.col_sorted_indx
    # ragged tensor metadata
    x_global_metadata = make_ragged_tensor_metadata(expt_sizes, dispatch_indx.shape[0])
    # convert x from dp-local to expert-sorted, ep-local
    y_ep_local = convert_dp_to_ep(x_dp_local_fp8, expt_assignment, active_indx, dispatch_indx, symm_mem_pool)
    y_ep_local_metadata = remap_ragged_tensor_metadata(x_global_metadata, expt_assignment.expt_map[rank, :])
    # first matmul + swiglu
    with scoped_opt_flags_constraints(fc1_constraints or {}):
        y_ep_local = matmul(y_ep_local, w1_ep_local, b1_ep_local, a_ragged_metadata=y_ep_local_metadata,
                            precision_config=pc1, fused_activation=act1)
    # second matmul
    with scoped_opt_flags_constraints(fc2_constraints or {}):
        y_ep_local = matmul(y_ep_local, w2_ep_local, b2_ep_local, a_ragged_metadata=y_ep_local_metadata,
                            precision_config=pc2)
    # convert x from expert-sorted, ep-local to token-sorted, dp-local
    y_dp_local = convert_ep_to_dp(y_ep_local, expt_assignment, active_indx, combine_indx, symm_mem_pool)
    # weighted average of the output token from experts
    y_dp_local = y_dp_local.view(-1, n_expts_act, y_dp_local.shape[-1])
    z_dp_local, _ = reduce(y_dp_local, dim=1)
    return z_dp_local


def bench_mlp(batch_per_expt, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, EP, shuffle_mx4=False,
              num_stages_fc1=None, num_stages_fc2=None, epilogue_subtile_fc1=None):
    assert n_expts_tot % EP == 0
    rank = torch.distributed.get_rank()
    n_ranks = torch.distributed.get_world_size()
    dev = torch.cuda.current_device()
    assert dev == rank, f"{torch.cuda.get_current_device()=}, {rank=}"
    assert n_expts_tot % EP == 0, f"{n_expts_tot=}, {EP=}, n_expts_tot must be divisible by EP"
    batch = batch_per_expt * n_expts_tot // n_expts_act
    assert EP == n_ranks, f"{EP=}, {n_ranks=}"

    #-- init memory pool --
    symm_mem_pool = SymmetricMemoryPool(Mesh(torch.distributed.group.WORLD))
    symm_mem_pool.initialize_matmul(
        n_tokens_global=batch_per_expt * n_expts_tot // n_expts_act,
        d_input=dim1,
        d_model=dim2,
        n_expts_act=n_expts_act,
        n_expts_tot=n_expts_tot,
        dtype=x_dtype,
        device=torch.cuda.current_device(),
    )

    # -- init prameters --
    # weights
    wg_global = torch.randn((dim1, n_expts_tot), device=dev)
    torch.distributed.broadcast(wg_global, src=0)
    w1_ep_local = torch.randn((n_expts_tot // EP, dim1, dim2), device=dev)
    w2_ep_local = torch.randn((n_expts_tot // EP, dim2 // 2, dim1), device=dev)
    # biases
    bg_global = torch.randn((n_expts_tot, ), device=dev)
    torch.distributed.broadcast(bg_global, src=0)
    b1_ep_local = torch.randn((n_expts_tot // EP, dim2), device=dev)
    b2_ep_local = torch.randn((n_expts_tot // EP, dim1), device=dev)
    torch.distributed.barrier()
    # quantize
    opt1 = dict()
    opt2 = dict()
    if w_dtype == FP4:
        num_warps = 4 if batch <= 512 else 8
        value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
        scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=num_warps)
        opt1 = {
            "value_layout": value_layout,
            "scale_layout": scale_layout,
        }
        opt2 = deepcopy(opt1)
    wg_global, wg_flex, wg_scale = quantize_weight(wg_global, torch.bfloat16)
    w1_ep_local, w1_flex, w1_scale = quantize_weight(w1_ep_local, w_dtype, **opt1)
    w2_ep_local, w2_flex, w2_scale = quantize_weight(w2_ep_local, w_dtype, **opt2)
    pcg = PrecisionConfig(
        flex_ctx=FlexCtx(rhs_data=wg_flex),
        b_mx_scale=wg_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
    )
    pc1 = PrecisionConfig(
        flex_ctx=FlexCtx(rhs_data=w1_flex),
        b_mx_scale=w1_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
    )
    pc2 = PrecisionConfig(
        flex_ctx=FlexCtx(rhs_data=w2_flex),
        b_mx_scale=w2_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
    )

    # -- init activation --
    x_dp_local_fp8 = torch.randn((batch // n_ranks, dim1), device=dev).to(x_dtype)
    x_dp_local_bf16 = x_dp_local_fp8.to(torch.bfloat16)

    # -- matmul fusion options --
    act1 = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2), (1.0, 1.0))

    # -- run benchmark --
    expt_dict = make_expt_dict_uniform(EP, n_expts_tot)
    expt_assignment = make_expt_assignment(EP, n_expts_tot, expt_dict, torch.device(dev))

    # For MX4 shuffling: run one dry-run iteration to collect routing data, then infer block shapes
    if shuffle_mx4 and w_dtype == FP4:
        # Disable block swap: with shuffled weights, tile loads are already contiguous,
        # so the swap's cacheline optimization is unnecessary. More importantly, disabling
        # the swap gives block_k=128 (vs 256), halving per-stage smem footprint, which
        # enables fitting 5 pipeline stages instead of 4 — a bigger win than the swap.
        dry_run_constraints = {"disable_mx4_block_swap": True}
        if epilogue_subtile_fc1 is not None:
            dry_run_constraints["epilogue_subtile"] = epilogue_subtile_fc1
        with scoped_opt_flags_constraints(dry_run_constraints):
            # Dry-run routing to get ragged metadata and dispatched activations
            l_dry = matmul(x_dp_local_bf16, wg_global, bg_global, precision_config=pcg)
            l_active_dry = topk(l_dry, n_expts_act, apply_softmax=True, all_gather=True, symm_mem_pool=symm_mem_pool)
            active_indx_dry = l_active_dry.indx
            expt_sizes_dry = l_active_dry.mask_metadata.col_sum
            dispatch_indx_dry = l_active_dry.mask_metadata.row_sorted_indx
            x_global_meta_dry = make_ragged_tensor_metadata(expt_sizes_dry, dispatch_indx_dry.shape[0])
            y_dry = convert_dp_to_ep(x_dp_local_fp8, expt_assignment, active_indx_dry, dispatch_indx_dry, symm_mem_pool)
            y_meta_dry = remap_ragged_tensor_metadata(x_global_meta_dry, expt_assignment.expt_map[rank, :])

            if y_dry.nelement() > 0:
                # Infer block shapes for W1 (includes the block swap)
                opt_flags_w1 = _infer_opt_flags(y_dry, w1_ep_local, y_meta_dry, pc1)
                w1_block_k, w1_block_n = opt_flags_w1.block_k, opt_flags_w1.block_n
                w1_ep_local = _shuffle_mx4_weights(w1_ep_local, w1_block_k, w1_block_n)

                # Run W1 once to get intermediate for W2 block shape inference
                y_fc1_dry = matmul(y_dry, w1_ep_local, b1_ep_local, a_ragged_metadata=y_meta_dry, precision_config=pc1,
                                   fused_activation=act1)

                # Infer block shapes for W2 (includes the block swap)
                opt_flags_w2 = _infer_opt_flags(y_fc1_dry, w2_ep_local, y_meta_dry, pc2)
                w2_block_k, w2_block_n = opt_flags_w2.block_k, opt_flags_w2.block_n
                w2_ep_local = _shuffle_mx4_weights(w2_ep_local, w2_block_k, w2_block_n)

                print(f"Shuffled layout: FC1 block_k={w1_block_k}, block_n={w1_block_n}, "
                      f"stages={opt_flags_w1.num_stages}, subtile={opt_flags_w1.epilogue_subtile}; "
                      f"FC2 block_k={w2_block_k}, block_n={w2_block_n}, "
                      f"stages={opt_flags_w2.num_stages}, subtile={opt_flags_w2.epilogue_subtile}")
        torch.cuda.synchronize()

    # Build per-kernel constraints
    fc1_constraints = {}
    if shuffle_mx4:
        fc1_constraints["disable_mx4_block_swap"] = True
    if num_stages_fc1 is not None:
        fc1_constraints["num_stages"] = num_stages_fc1
    if epilogue_subtile_fc1 is not None:
        fc1_constraints["epilogue_subtile"] = epilogue_subtile_fc1
    fc2_constraints = {}
    if shuffle_mx4:
        fc2_constraints["disable_mx4_block_swap"] = True
    if num_stages_fc2 is not None:
        fc2_constraints["num_stages"] = num_stages_fc2

    fpath = Path(f"profile_{rank}")
    # warmup
    run_mlp(x_dp_local_bf16, x_dp_local_fp8,  #
            wg_global, bg_global, pcg,  #
            w1_ep_local, b1_ep_local, pc1, act1,  #
            w2_ep_local, b2_ep_local, pc2,  #
            n_expts_act, expt_assignment, rank, symm_mem_pool, fc1_constraints=fc1_constraints,
            fc2_constraints=fc2_constraints)
    torch.cuda.synchronize()
    proton.start(str(fpath), hook="triton")
    for i in range(100):
        run_mlp(x_dp_local_bf16, x_dp_local_fp8,  #
                wg_global, bg_global, pcg,  #
                w1_ep_local, b1_ep_local, pc1, act1,  #
                w2_ep_local, b2_ep_local, pc2,  #
                n_expts_act, expt_assignment, rank, symm_mem_pool, fc1_constraints=fc1_constraints,
                fc2_constraints=fc2_constraints)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    proton.finalize()
    return roofline.parse_profile(fpath.with_suffix(".hatchet"), useful_op_regex=".*matmul.*")


def roofline_mlp(batch_sizes, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, EP, \
                  name="", verbose=True, shuffle_mx4=False, num_stages_fc1=None,
                  num_stages_fc2=None, epilogue_subtile_fc1=None):
    suffix = "-shuffled" if shuffle_mx4 else ""
    suffix += f"-fc1stages{num_stages_fc1}" if num_stages_fc1 is not None else ""
    suffix += f"-fc2stages{num_stages_fc2}" if num_stages_fc2 is not None else ""
    suffix += f"-fc1subtile{epilogue_subtile_fc1}" if epilogue_subtile_fc1 is not None else ""
    out_path = Path(f"logs/{name}/{x_dtype}x-{w_dtype}w-EP{EP}{suffix}/")
    out_path.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(torch.distributed.get_rank())
    csv_path = roofline.compute_roofline(dim1, dim2, n_expts_tot, n_expts_act, parse_dtype(x_dtype),
                                         parse_dtype(w_dtype), EP,  # fixed args
                                         shuffle_mx4=shuffle_mx4,  # weight shuffling option
                                         num_stages_fc1=num_stages_fc1,  # override num_stages for FC1 kernel
                                         num_stages_fc2=num_stages_fc2,  # override num_stages for FC2 kernel
                                         epilogue_subtile_fc1=epilogue_subtile_fc1,  # override epilogue subtile for FC1
                                         bench_fn=bench_mlp,  # function to benchmark
                                         intensity_proxy_name="batch_per_expt",  # intensity proxy name
                                         intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
                                         verbose=verbose,  # options
                                         out_path=out_path.with_suffix(".csv"))  # output path
    png_path = roofline.plot_roofline(series=[csv_path],  # roofline data to plot
                                      flops_dtype=x_dtype,  # dtype to use for FLOPS roof
                                      xlabel="batch_per_expt", title=out_path,  # plot option
                                      out_path=out_path.with_suffix(".png"),  # output path
                                      max_tbps="memset", max_tflops="cublas")  # hardware limits
    return png_path


if __name__ == "__main__":
    # torchrun --nproc-per-node=2 ./bench_mlp.py --ep 2 --name gpt-oss-x2
    if not was_launched_with_torchrun():
        print("usage: torchrun --nproc-per-node=<EP> ./bench_mlp.py")
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", world_size=world_size, device_id=torch.device(local_rank))
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized", action="store_true", default=True)
    args = parser.parse_args()
    # set dtypes
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]
    # set model type
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    ep = torch.distributed.get_world_size()

    # Common args for all MoE scenarios
    moe_args = dict(dim1=5760, dim2=5760, n_expts_tot=128, n_expts_act=4, EP=ep, name="gpt-oss-x2")

    # Run 5 scenarios to isolate 3 optimization categories:
    #   Shuffling: scenario 2 → 3
    #   FC2 5stg:  scenario 3 → 4
    #   FC1 5stg:  scenario 4 → 5

    # 1. FP8 baseline
    roofline_mlp(batch_sizes, x_dtype=dense_dtypes[0], w_dtype=dense_dtypes[1], **moe_args)
    # 2. MX4 baseline
    roofline_mlp(batch_sizes, x_dtype=quantized_dtypes[0], w_dtype=quantized_dtypes[1], **moe_args)
    # 3. MX4 shuffled
    roofline_mlp(batch_sizes, x_dtype=quantized_dtypes[0], w_dtype=quantized_dtypes[1], shuffle_mx4=True, **moe_args)
    # 4. MX4 shuffled + FC2 5stg
    roofline_mlp(batch_sizes, x_dtype=quantized_dtypes[0], w_dtype=quantized_dtypes[1], shuffle_mx4=True,
                 num_stages_fc2=5, **moe_args)
    # 5. MX4 shuffled + FC1 subtile2+5stg + FC2 5stg
    #    block_k=128 (swap disabled) + subtile=2 frees enough smem for 5 stages
    roofline_mlp(batch_sizes, x_dtype=quantized_dtypes[0], w_dtype=quantized_dtypes[1], shuffle_mx4=True,
                 epilogue_subtile_fc1=2, num_stages_fc1=5, num_stages_fc2=5, **moe_args)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
