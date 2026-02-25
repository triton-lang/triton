defs: dict[str, str] = {
    "convert_host_descriptor": R"""
def convert_host_descriptor(desc):
    def torch_dtype_to_triton(dtype):
        import torch

        if dtype == torch.float8_e5m2:
            return gl.float8e5
        if dtype == torch.float8_e4m3fn:
            return gl.float8e4nv
        return getattr(gl, str(dtype).split(".")[1])

    from triton.tools.tensor_descriptor import TensorDescriptor

    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    dtype = desc.base.dtype
    tensor = desc.base
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, torch_dtype_to_triton(dtype))
    return gluon.nvidia.hopper.TensorDescriptor(
        tensor, desc.shape, desc.strides, block_shape, layout
    )
"""
}
