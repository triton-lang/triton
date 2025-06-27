import torch
from .swizzle_details.hopper_value import swizzle_mxfp4_value_hopper, unswizzle_mxfp4_value_hopper_torch
from .swizzle_details.hopper_scale import swizzle_mxfp4_scale_hopper, unswizzle_mxfp4_scale_hopper_torch
from .swizzle_details.blackwell_scale import swizzle_mx_scale_bw, unswizzle_mx_scale_bw_torch
from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows
from enum import Enum


class SwizzlingType(Enum):
    HOPPER_VALUE = 0
    HOPPER_SCALE = 1
    BLACKWELL_SCALE = 2


class Tensor:

    def _compute_stride(self, shape, strides, swizzle_mode):
        # Check expected properties of the weights.
        if swizzle_mode == SwizzlingType.BLACKWELL_SCALE:
            mxE, mxK, mxN = shape

            # Compute strides of the 5D swizzled tensor.
            swizzled_shape = (mxE, mxN // 128, mxK // 4, 32, 4, 4)
            s5 = 1
            s4 = swizzled_shape[5] * s5  # 4 * 1 = 4
            s3 = swizzled_shape[4] * s4  # 32 * 4 = 128
            s2 = swizzled_shape[3] * s3  # 4 * 128 = 512
            s1 = swizzled_shape[2] * s2  # (mxK//4) * 512
            s0 = swizzled_shape[1] * s1  # (mxN//128) * ((mxK//4)*512)
            return s0, s2, s1
        return strides

    def __init__(self, handle, shape=None, shape_max=None, swizzle_mode=None):
        if shape is None:
            shape = handle.shape
        if shape_max is None:
            shape_max = [None] * len(shape)
        self.handle = handle
        self.ndim = handle.ndim
        self.dtype = handle.dtype
        self.device = handle.device
        # shape may contain a mix of `int` and `torch.Tensor`
        is_int = lambda s: isinstance(s, int)
        is_item = lambda s: hasattr(s, "numel") and s.numel() == 1
        self.shape = shape
        assert all(map(lambda s: is_int(s) or is_item(s), self.shape))
        # shape_max is guarantee to be all `int`
        self.shape_max = shape_max
        for i, (s, smax) in enumerate(zip(self.shape, self.shape_max)):
            if smax is not None and not is_int(smax):
                raise ValueError(f"shape_max[{i}] must be `int` or `None`; got {type(smax)}")
            if smax is None:
                self.shape_max[i] = s
        assert all(map(is_int, self.shape_max))
        # TODO: clean all this up
        self.strides = self._compute_stride(shape, handle.stride(), swizzle_mode)
        self.is_fp4 = self.dtype == torch.uint8
        self.element_bitwidth = 4 if self.is_fp4 else self.dtype.itemsize * 8
        self.ndim = handle.ndim
        self.swizzle_mode = swizzle_mode

    def stride(self, i=None):
        return self.strides if i is None else self.strides[i]

    def data_ptr(self):
        return self.handle.data_ptr()

    # TODO: clean up
    def numel(self):
        return self.handle.numel()

    def element_size(self):
        return self.handle.element_size()

    def permute(self, *permutation):
        assert self.swizzle_mode is None
        h = self.handle.permute(*permutation)
        return Tensor(h, swizzle_mode=self.swizzle_mode)

    def view(self, *args):
        assert self.swizzle_mode is None
        h = self.handle.view(*args)
        return Tensor(h, swizzle_mode=self.swizzle_mode)


class Bitmatrix(Tensor):
    """
    Represents a boolean matrix in a packed format where each element occupies
    a single bit of memory.

    _scratchpad is either None or an all-zero array of size >= shape[-1]; we pass it along
    with the actual bitmatrix to avoid having to launch a separate memset
    kernel when we call Bitmatrix::sum().
    """

    _scratchpad: torch.Tensor

    def __init__(self, handle, shape, shape_max, scratchpad=None):
        assert handle.shape[-1] * 32 == shape[-1]
        assert handle.ndim == 2
        super().__init__(handle, shape=shape, shape_max=shape_max)
        assert self.dtype == torch.uint32
        self._scratchpad = scratchpad

    def sum(self, partials_block_size):
        _, n_cols = self.shape
        dev = self.device
        if self._scratchpad is None:
            self._scratchpad = clear_sums(n_cols, dev)
        out_ret = self._scratchpad[:n_cols]
        self._scratchpad = None  # throw error if we try to sum again
        return sum_bitmatrix_rows(self, out_ret, partials_block_size)


def perm_to_contig(ndim: int, axis: int, swizzle_axis: int | None = None) -> tuple[int, ...]:
    """
    Permute the shape so that axis is the last dimension and swizzle_axis is the second to last dimension.
    """
    # FIXME(Lezcano): This API is not very good as it's too generic.
    # Chances are we just care about the cases
    # - axis=-2 and swizzle_axis=-1
    # - axis=-1 and swizzle_axis=-2
    # - axis=anything and swizzle_axis=None
    # We could probably just implement
    # perm_to_contig(ndim, transpose: bool)
    # where we transpose the last two dimensions if transpose is True and otherwise we leave them as is.
    axis = axis if axis >= 0 else axis + ndim
    if swizzle_axis is not None:
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim

    assert axis != swizzle_axis
    shape = list(range(ndim))
    shape[axis], shape[-1] = shape[-1], shape[axis]
    if swizzle_axis is not None:
        if swizzle_axis == len(shape) - 1:
            swizzle_axis = axis
        shape[swizzle_axis], shape[-2] = shape[-2], shape[swizzle_axis]
    return tuple(shape)


def perm_from_contig(ndim: int, axis: int, swizzle_axis: int | None = None) -> tuple[int, ...]:
    # Invert the permutation via argsort
    perm = perm_to_contig(ndim, axis, swizzle_axis)
    inv = [0] * ndim
    for i, v in enumerate(perm):
        inv[v] = i
    return tuple(inv)


def perm_tensor_to_contig(x: torch.Tensor, axis: int, swizzle_axis: int | None = None) -> torch.Tensor:
    """
    Permute the tensor x moving axis to the last dimension and swizzle_axis to the second to last dimension.
    """
    return x.permute(perm_to_contig(x.ndim, axis, swizzle_axis))


def perm_tensor_from_contig(x: torch.Tensor, axis: int, swizzle_axis: int | None = None) -> torch.Tensor:
    """
    Permute the tensor x moving the last dimension to axis and the second to last dimension to swizzle_axis.
    """
    return x.permute(perm_from_contig(x.ndim, axis, swizzle_axis))


def swizzle(tensor, swizzle_mode):
    # Permute the tensor so that axis is the last dimension and swizzle_axis is the second to last dimension.
    ret_shape = list(tensor.shape)
    if tensor.dtype == torch.uint8:
        ret_shape[1] *= 2
    # ret_shape = None
    swizzle_axis = 2
    axis = tensor.stride().index(1)
    perm = list(range(tensor.ndim))
    perm[tensor.ndim - 1], perm[axis] = axis, tensor.ndim - 1
    if swizzle_axis is not None:
        perm[tensor.ndim - 2], perm[swizzle_axis] = swizzle_axis, tensor.ndim - 2
    tensor = torch.permute(tensor, perm).contiguous()
    if swizzle_mode == SwizzlingType.HOPPER_VALUE:
        tensor = swizzle_mxfp4_value_hopper(tensor, op_idx=0, mma_version=3)
    elif swizzle_mode == SwizzlingType.BLACKWELL_SCALE:
        tensor = swizzle_mx_scale_bw(tensor, allow_pad=True)
    elif swizzle_mode == SwizzlingType.HOPPER_SCALE:
        tensor = swizzle_mxfp4_scale_hopper(tensor, num_warps=8)
    tensor = perm_tensor_from_contig(tensor, axis, swizzle_axis)
    return Tensor(tensor, shape=ret_shape, swizzle_mode=swizzle_mode)


def unswizzle(tensor, swizzle_mode):
    if isinstance(tensor, Tensor):
        tensor = tensor.handle
    swizzle_axis = 2
    axis = tensor.stride().index(1)
    tensor = perm_tensor_to_contig(tensor, axis, swizzle_axis)
    if swizzle_mode == SwizzlingType.HOPPER_VALUE:
        tensor = unswizzle_mxfp4_value_hopper_torch(tensor, op_idx=0, mma_version=3)
    elif swizzle_mode == SwizzlingType.BLACKWELL_SCALE:
        tensor = unswizzle_mx_scale_bw_torch(tensor)
    elif swizzle_mode == SwizzlingType.HOPPER_SCALE:
        tensor = unswizzle_mxfp4_scale_hopper_torch(tensor, num_warps=8)
    # permute
    ndim = tensor.ndim
    perm = list(range(ndim))
    perm[ndim - 1], perm[axis] = axis, ndim - 1
    if swizzle_axis is not None:
        perm[ndim - 2], perm[swizzle_axis] = swizzle_axis, ndim - 2
    return torch.permute(tensor, perm).contiguous()
