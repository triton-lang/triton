import torch
from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows


class Tensor:

    def __init__(self, handle, shape, shape_max=None):
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

    def stride(self, *args):
        return self.handle.stride(*args)

    def data_ptr(self):
        return self.handle.data_ptr()


class Bitmatrix(Tensor):
    """
    Represents a boolean matrix in a packed format where each element occupies
    a single bit of memory.

    _scratchpad is either None or an all-zero array of size >= shape[-1]; we pass it along
    with the actual bitmatrix to avoid having to launch a separate memset
    kernel when we call Bitmatrix::sum().
    """

    _scratchpad: torch.Tensor

    def __init__(self, handle, shape, scratchpad=None):
        assert handle.ndim == 2
        shape_max = [handle.shape[0], handle.shape[1] * 32]
        super().__init__(handle, shape, shape_max)
        assert self.dtype == torch.uint32
        self._scratchpad = scratchpad

    def sum(self, partials_block_size):
        _, n_cols = self.shape
        dev = self.device
        if self._scratchpad is None:
            self._scratchpad = clear_sums(n_cols, dev)
        out_ret = self._scratchpad[:n_cols]
        self._scratchpad = None  # throw error if we try to sum again
        return sum_bitmatrix_rows(self, out_ret, partials_block_size, self.shape[0])


class SwizzledTensor:

    def _compute_shape(self, handle, swizzle_mode):
        from .numerics_details.mxfp import SwizzlingType
        shape = list(handle.shape)
        if handle.dtype == torch.uint8:
            # Assume 2 fp4s packed into a byte
            shape[1] *= 2
        if swizzle_mode == SwizzlingType.HOPPER:
            shape[1] //= 4
            shape[2] *= 4
        return tuple(shape)

    def _compute_stride(self, handle, swizzle_mode):
        from .numerics_details.mxfp import SwizzlingType
        # Check expected properties of the weights.
        if swizzle_mode == SwizzlingType.BLACKWELL:
            mxE, mxK, mxN = handle.shape

            # Compute strides of the 5D swizzled tensor.
            swizzled_shape = (mxE, mxN // 128, mxK // 4, 32, 4, 4)
            s5 = 1
            s4 = swizzled_shape[5] * s5  # 4 * 1 = 4
            s3 = swizzled_shape[4] * s4  # 32 * 4 = 128
            s2 = swizzled_shape[3] * s3  # 4 * 128 = 512
            s1 = swizzled_shape[2] * s2  # (mxK//4) * 512
            s0 = swizzled_shape[1] * s1  # (mxN//128) * ((mxK//4)*512)
            return s0, s2, s1
        return handle.stride()

    def __init__(self, handle, swizzle_mode=None):
        self.handle = handle
        self.dtype = handle.dtype
        self.is_fp4 = self.dtype == torch.uint8
        self.element_bitwidth = 4 if self.is_fp4 else self.dtype.itemsize * 8
        self.ndim = handle.ndim
        self.shape = self._compute_shape(handle, swizzle_mode)
        self.strides = self._compute_stride(handle, swizzle_mode)
        self.swizzle_mode = swizzle_mode

    def element_size(self):
        return self.handle.element_size()

    def stride(self, i=None):
        return self.strides if i is None else self.strides[i]

    def data_ptr(self):
        return self.handle.data_ptr()
