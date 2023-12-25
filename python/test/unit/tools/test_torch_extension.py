import json
import os
import shutil
import tempfile
from dataclasses import dataclass

import pytest
import torch
from torch.utils.cpp_extension import load

import triton
from triton.compiler import CompiledKernel
from triton.runtime.driver import driver
from triton.tools.extension import TorchExtCodeGen

torch.manual_seed(0)


@pytest.fixture
def cache_dir(tmp_path):
    os.environ["TRITON_CACHE_DIR"] = str(tmp_path.absolute())
    yield tmp_path.absolute()


def clear_torch_extension_dir():
    torch_extension_dir = os.path.expanduser("~/.cache/torch_extensions")
    if os.path.exists(torch_extension_dir):
        shutil.rmtree(torch_extension_dir)


@dataclass
class CUDAContext:
    device: int
    stream: int


@pytest.fixture
def cuda_context():
    device = driver.get_current_device()
    stream = driver.get_current_stream(device)

    return CUDAContext(device=device, stream=stream)


@pytest.fixture
def cuda_context():
    device = driver.get_current_device()
    stream = driver.get_current_stream(device)

    return CUDAContext(device=device, stream=stream)


def run_compiled_kernel(kernel, args, grid, stream):
    grid_0, grid_1, grid_2 = grid
    kernel.run(
        grid_0,
        grid_1,
        grid_2,
        kernel.num_warps,
        kernel.num_ctas,
        kernel.cluster_dims[0],
        kernel.cluster_dims[1],
        kernel.cluster_dims[2],
        kernel.shared,
        stream,
        kernel.function,
        CompiledKernel.launch_enter_hook,
        CompiledKernel.launch_exit_hook,
        kernel,
        *driver.assemble_tensormap_to_arg(kernel.metadata["tensormaps_info"], args),
    )


def load_extension(cache_dir, kernel_name):
    clear_torch_extension_dir()
    fn_cache_dir = list(cache_dir.glob(f"**/{kernel_name}.json"))[0].parent

    metadata_path = f"{fn_cache_dir}/{kernel_name}.json"
    metadata_group = f"{fn_cache_dir}/__grp__{kernel_name}.json"

    metadata = json.load(open(metadata_path))
    metadata_group = json.load(open(metadata_group))["child_paths"]
    cubin_path = metadata_group[f"{kernel_name}.cubin"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        extension_path = os.path.join(tmp_dir, f"{kernel_name}.cu")

        codegen = TorchExtCodeGen(
            kernel_name=kernel_name, metadata=metadata, metadata_group=metadata_group
        )
        codegen.generate(extension_path)
        module = load(
            name=f"{kernel_name}",
            sources=[extension_path],
            extra_cflags=["-O2"],
            extra_ldflags=["-lcuda"],
            verbose=True,
        )
    return module, cubin_path


def run_jit(kernel, grid, *args, **kwargs) -> CompiledKernel:
    return kernel.run(*args, grid=grid, warmup=True, save_extra_meta=True, **kwargs)


def skip_torch_extension_tests():
    return not os.environ.get("RUN_TORCH_EXTENSION_TESTS", False)


@pytest.mark.skipif(
    skip_torch_extension_tests(),
    reason="Torch extension tests skipped, set RUN_TORCH_EXTENSION_TESTS=1 to run",
)
def test_add_kernel(cache_dir, cuda_context):
    from kernels import add_kernel

    size = 98432
    dtype = torch.half
    x = torch.rand(size, device="cuda", dtype=dtype)
    y = torch.rand(size, device="cuda", dtype=dtype)
    triton_output = torch.empty_like(x)
    n_elements = triton_output.numel()
    BLOCK_SIZE = 1024

    # Run JIT kernel to generate metadata and cubin
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    jit_kernel = run_jit(
        add_kernel, grid, x, y, triton_output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    args = (x, y, triton_output, n_elements)
    ref_output = x + y
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)
    run_compiled_kernel(jit_kernel, args, grid, cuda_context.stream)
    assert torch.allclose(triton_output, ref_output)

    kernel_module, cubin_path = load_extension(cache_dir, "add_kernel")

    kernel_interface = [
        getattr(kernel_module, attr) for attr in dir(kernel_module) if "Kernel" in attr
    ]
    assert len(kernel_interface) == 1
    kernel_interface = kernel_interface[0]
    assert getattr(kernel_interface, "BLOCK_SIZE") == BLOCK_SIZE

    # Compile AOT kernel
    kernel = kernel_interface(cubin_path)
    extension_output = torch.empty_like(x)
    args = (x, y, extension_output, n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)
    kernel[grid](*args)

    output_torch = x + y
    assert torch.allclose(triton_output, extension_output)


@pytest.mark.skipif(
    skip_torch_extension_tests(),
    reason="Torch extension tests skipped, set RUN_TORCH_EXTENSION_TESTS=1 to run",
)
@pytest.mark.parametrize("shape", [(1823, 781), (2000, 2048), (2000, 4096)])
def test_fused_softmax(cache_dir, shape, cuda_context):
    from kernels import softmax_kernel

    x = torch.randn(*shape, device="cuda")
    y_torch = torch.softmax(x, axis=1)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y_triton = torch.empty_like(x)
    grid = (n_rows, 1, 1)
    jit_kernel = run_jit(
        softmax_kernel,
        grid,
        y_triton,
        x,
        x.stride(0),
        y_triton.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    args = (
        y_triton,
        x,
        x.stride(0),
        y_triton.stride(0),
        n_cols,
    )
    run_compiled_kernel(jit_kernel, args, grid, cuda_context.stream)
    assert torch.allclose(y_triton, torch.softmax(x, axis=1))

    kernel_module, cubin_path = load_extension(cache_dir, f"softmax_kernel")
    kernel_interface = [
        getattr(kernel_module, attr) for attr in dir(kernel_module) if "Kernel" in attr
    ]
    assert len(kernel_interface) == 1
    kernel_interface = kernel_interface[0]

    assert getattr(kernel_interface, "BLOCK_SIZE") == BLOCK_SIZE
    assert getattr(kernel_interface, "NUM_WARPS") == num_warps

    torch_kernel = kernel_interface(cubin_path)
    output = torch.empty_like(x)
    args = (output, x, x.stride(0), output.stride(0), n_cols)
    torch_kernel[grid](*args)
    assert torch.allclose(output, y_torch)


@pytest.mark.skipif(
    skip_torch_extension_tests(),
    reason="Torch extension tests skipped, set RUN_TORCH_EXTENSION_TESTS=1 to run",
)
@pytest.mark.parametrize(
    "activation",
    ["", "leaky_relu"],
)
@pytest.mark.parametrize(
    "config",
    [
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ],
    ids=lambda c: str(c),
)
def test_matmul(cache_dir, config, activation, cuda_context):
    from kernels import matmul_kernel

    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16)

    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    M, K = a.shape
    K, N = b.shape

    triton_output = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = (
        triton.cdiv(M, config.kwargs["BLOCK_SIZE_M"])
        * triton.cdiv(N, config.kwargs["BLOCK_SIZE_N"]),
        1,
        1,
    )
    jit_kernel = run_jit(
        matmul_kernel,
        grid,
        a,
        b,
        triton_output,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        triton_output.stride(0),
        triton_output.stride(1),
        **config.kwargs,
        ACTIVATION=activation,
    )

    args = (
        a,
        b,
        triton_output,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        triton_output.stride(0),
        triton_output.stride(1),
    )

    run_compiled_kernel(jit_kernel, args, grid, cuda_context.stream)
    # ref_output = torch.matmul(a, b)
    # if activation == "leaky_relu":
    #     ref_output = torch.nn.functional.leaky_relu(ref_output)

    # print(f"Max abs diff: {torch.max(torch.abs(ref_output - triton_output))}")
    # assert torch.allclose(triton_output, ref_output, atol=1e-2, rtol=0)
    kernel_extension, cubin_path = load_extension(cache_dir, "matmul_kernel")

    kernel_interface = [
        getattr(kernel_extension, attr)
        for attr in dir(kernel_extension)
        if "Kernel" in attr
    ]
    assert len(kernel_interface) == 1
    kernel_interface = kernel_interface[0]

    for key, value in config.kwargs.items():
        assert getattr(kernel_interface, key) == value

    torch_kernel = kernel_interface(cubin_path)
    extension_output = torch.empty_like(triton_output)
    args = (
        a,
        b,
        extension_output,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        extension_output.stride(0),
        extension_output.stride(1),
    )

    torch_kernel[grid](*args)
    print(
        f"Max abs diff triton vs extension: {torch.max(torch.abs(extension_output - triton_output))}"
    )
    assert torch.allclose(extension_output, triton_output, atol=1e-2, rtol=0)


@pytest.mark.skipif(
    skip_torch_extension_tests(),
    reason="Torch extension tests skipped, set RUN_TORCH_EXTENSION_TESTS=1 to run",
)
@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_flash_attention(Z, H, N_CTX, D_HEAD, causal, dtype, cache_dir, cuda_context):
    from kernels import _attn_fwd

    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5

    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    o = torch.empty_like(q)
    BLOCK_M = 128
    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3
    num_warps = 4
    stage = 3 if causal else 1

    if torch.cuda.get_device_capability()[0] == 9:
        num_warps = 8
        num_stages = 7 if Lk >= 64 else 3
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    constexprs = dict(
        N_CTX=q.shape[2],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        STAGE=stage,
    )
    launch_params = dict(
        num_warps=num_warps,
        num_stages=num_stages,
    )
    jit_kernel = run_jit(
        _attn_fwd,
        grid,
        q,
        k,
        v,
        sm_scale,
        M,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
        **constexprs,
        **launch_params,
    )
    args = (
        q,
        k,
        v,
        sm_scale,
        M,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
    )
    run_compiled_kernel(jit_kernel, args, grid=grid, stream=cuda_context.stream)
    kernel_extension, cubin_path = load_extension(cache_dir, "_attn_fwd")

    kernel_interface = [
        getattr(kernel_extension, attr)
        for attr in dir(kernel_extension)
        if "Kernel" in attr
    ]
    assert len(kernel_interface) == 1
    kernel_interface = kernel_interface[0]

    for key, value in constexprs.items():
        assert getattr(kernel_interface, key) == value

    for key, value in launch_params.items():
        assert getattr(kernel_interface, key.upper()) == value

    torch_kernel = kernel_interface(cubin_path)
    extension_output = torch.empty_like(o)
    args = (
        q,
        k,
        v,
        sm_scale,
        M,
        extension_output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        extension_output.stride(0),
        extension_output.stride(1),
        extension_output.stride(2),
        extension_output.stride(3),
        q.shape[0],
        q.shape[1],
    )

    torch_kernel[grid](*args)
    triton_output = o
    assert torch.allclose(extension_output, triton_output)
