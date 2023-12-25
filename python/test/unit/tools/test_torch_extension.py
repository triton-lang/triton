import json
import os
import shutil
import tempfile

import pytest
import torch
from torch.utils.cpp_extension import load

import triton
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


def load_extension(cache_dir, kernel_name):
    clear_torch_extension_dir()
    fn_cache_dir = list(cache_dir.glob(f"**/{kernel_name}.json"))[0].parent

    metadata_path = f"{fn_cache_dir}/{kernel_name}.json"
    metadata_group = f"{fn_cache_dir}/__grp__{kernel_name}.json"

    metadata = json.load(open(metadata_path))
    metadata_group = json.load(open(metadata_group))["child_paths"]
    cubin_path = metadata_group[f"{kernel_name}.cubin"]
    # extension_path = cache_dir / f"{kernel_name}.cu"

    with tempfile.TemporaryDirectory() as tmp_dir:
        codegen = TorchExtCodeGen(
            kernel_name=kernel_name, metadata=metadata, metadata_group=metadata_group
        )
        extension_path = os.path.join(tmp_dir, f"{kernel_name}.cu")
        codegen.generate(extension_path)
        module = load(
            name=f"{kernel_name}",
            sources=[extension_path],
            extra_cflags=["-O2"],
            extra_ldflags=["-lcuda"],
            verbose=True,
        )
    return module, cubin_path


def run_jit(kernel, grid, *args, **kwargs):
    kernel.run(*args, grid=grid, warmup=True, save_extra_meta=True, **kwargs)


def skip_torch_extension_tests():
    return not os.environ.get("RUN_TORCH_EXTENSION_TEST", False)


@pytest.mark.skipif(
    skip_torch_extension_tests(),
    reason="Torch extension tests skipped, set RUN_TORCH_EXTENSION_TESTS=1 to run",
)
def test_add_kernel(cache_dir):
    from kernels import add_kernel

    size = 98432
    dtype = torch.half
    x = torch.rand(size, device="cuda", dtype=dtype)
    y = torch.rand(size, device="cuda", dtype=dtype)
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024

    # Run JIT kernel to generate metadata and cubin
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    run_jit(add_kernel, grid, x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    kernel_module, cubin_path = load_extension(cache_dir, "add_kernel")

    kernel_interface = [
        getattr(kernel_module, attr) for attr in dir(kernel_module) if "Kernel" in attr
    ]
    assert len(kernel_interface) == 1
    kernel_interface = kernel_interface[0]
    assert getattr(kernel_interface, "BLOCK_SIZE") == BLOCK_SIZE

    # Compile AOT kernel
    kernel = kernel_interface(cubin_path)
    output = torch.empty_like(x)
    args = (x, y, output, n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)
    kernel[grid](*args)

    output_torch = x + y
    assert torch.allclose(output, output_torch)


@pytest.mark.parametrize("shape", [(1823, 781), (2000, 2048), (2000, 4096)])
def test_fused_softmax(cache_dir, shape):
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
    run_jit(
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
