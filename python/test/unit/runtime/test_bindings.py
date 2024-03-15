import triton
import triton.language as tl
from triton.compiler.backends.cuda import CUDABackend
from triton.runtime.driver import driver

import torch


@triton.jit
def add_helper(x, y):
    return x + y


@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    n_elements,
    out_ptr,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = add_helper(x, y)
    tl.store(out_ptr + offsets, output, mask=mask)


def test_module_walk():
    """
    Test the MLIR bindings exposed for the out-ot-tree walk.
    """

    def walk_fn(op):
        name = op.get_name()
        for i in range(op.get_num_results()):
            op.get_result(i).id()
        for i in range(op.get_num_operands()):
            op.get_operand(i).id()
        for i in range(op.get_num_regions()):
            op.get_region(i).id()
        block = op.get_block()
        if block is not None:
            block.id()
            for i in range(block.get_num_arguments()):
                block.get_argument(i)
        if name == "tt.func":
            op.get_str_attr("sym_name")
        if name == "tt.call":
            op.get_flat_symbol_ref_attr("callee")

    kernel = add_kernel
    args = [
        torch.empty((32, 32), device="cuda"),  # in_ptr0
        torch.empty((32, 32), device="cuda"),  # in_ptr1
        1024,  # n_elements
        torch.empty((32, 32), device="cuda"),  # out_ptr
        16,  # BLOCK_SIZE
    ]
    src = triton.compiler.compiler.ASTSource(
        fn=kernel,
        signature={i: kernel._type_of(kernel._key_of(arg))
                   for i, arg in enumerate(args)
                   if i not in kernel.constexprs},
        constants={i: arg
                   for i, arg in enumerate(args)
                   if not isinstance(arg, torch.Tensor)},
        attrs=kernel._get_config(*args, ),
    )

    triton._C.libtriton.ir = triton._C.libtriton.triton.ir
    context = triton._C.libtriton.ir.context()

    target = driver.get_current_target()
    backend = CUDABackend(target)
    options = backend.parse_options(dict())

    ttir_module = src.make_ir(options)
    ttir_module.walk(walk_fn)
