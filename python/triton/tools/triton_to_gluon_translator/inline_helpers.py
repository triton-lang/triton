_torch_dtype_to_triton_def = R"""
def _torch_dtype_to_triton(dtype):
    import torch

    if dtype == torch.float8_e5m2:
        return gl.float8e5
    if dtype == torch.float8_e4m3fn:
        return gl.float8e4nv
    return getattr(gl, str(dtype).split(".")[1])
"""

defs: dict[str, str] = {
    "convert_host_descriptor":
    _torch_dtype_to_triton_def + R"""
def convert_host_descriptor(desc):
    from triton.tools.tensor_descriptor import TensorDescriptor

    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    dtype = desc.base.dtype
    tensor = desc.base
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, _torch_dtype_to_triton(dtype))
    return gluon.nvidia.hopper.TensorDescriptor(
        tensor, desc.shape, desc.strides, block_shape, layout
    )
""",
    "convert_host_descriptor_amd":
    _torch_dtype_to_triton_def + R"""
def convert_host_descriptor(desc):
    from triton.tools.tensor_descriptor import TensorDescriptor

    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    dtype = desc.base.dtype
    layout = gl.PaddedSharedLayout.with_identity_for(
        [[block_shape[-1], 4]], list(block_shape), [1, 0]
    )
    return gluon.amd.gfx1250.TensorDescriptor(
        desc.base, list(desc.shape), list(desc.strides), block_shape, layout
    )
""",
}
