import torch
from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows


class Tensor:

    def __init__(self, handle, shape_raw, shape_pad=None):
        self.handle = handle
        self.ndim = handle.ndim
        self.dtype = handle.dtype
        self.device = handle.device
        self.shape_pad = handle.shape if shape_pad is None else shape_pad
        self.shape_raw = shape_raw

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

    def __init__(self, handle, shape_raw, scratchpad=None):
        assert handle.ndim == 2
        shape_pad = [handle.shape[0], handle.shape[1] * 32]
        super().__init__(handle, shape_raw, shape_pad)
        assert self.dtype == torch.uint32
        self._scratchpad = scratchpad

    def sum(self, partials_block_size):
        _, n_cols = self.shape_raw
        dev = self.device
        if self._scratchpad is None:
            self._scratchpad = clear_sums(n_cols, dev)
        out_ret = self._scratchpad[:n_cols]
        self._scratchpad = None  # throw error if we try to sum again
        return sum_bitmatrix_rows(self, out_ret, partials_block_size, self.shape_raw[0])


def _raise_if_not(cond, msg):
    if not cond: raise ValueError(msg)


class MxTensor:

    def _check_invariants(self):
        from .numerics_details.mxfp import SwizzlingType
        valid_dtypes = {torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn}
        _raise_if_not(self.dtype in valid_dtypes, f"Tensor dtype must be in {valid_dtypes}; got {self.dtype}")
        _raise_if_not(len(self.data_shape) == 3, f"Data must be 3D; got {self.data_shape}")
        _raise_if_not(len(self.scale_shape) == 3, f"Scale must be 3D; got {self.scale_shape}")
        # TODO: this should be done in a new swizzling abstraction
        # _raise_if_not(self.data_shape[::2] == self.scale_shape[::2], f"Shape mismatch between data and shape;"
        #                                                              f"{self.data_shape} != {self.scale_shape}")
        # _raise_if_not(self.data_shape[1] % self.block_size == 0, \
        #              f"reduction dimension should be divisible by {self.block_size}")
        if self.swizzle_mode == SwizzlingType.HOPPER:
            _raise_if_not(self.data_shape[1] % 64 == 0 and self.data_shape[2] % 64 == 0, \
                f"Hopper scale swizzling acts on a 64x64 tile (4x4 mma tiles). Got {self.data_shape=}")

    def _compute_scale_stride(self, scale, swizzle_mode):
        from .numerics_details.mxfp import SwizzlingType
        # Check expected properties of the weights.
        if swizzle_mode == SwizzlingType.BLACKWELL:
            mxE, mxK, mxN = self.scale.shape

            # Compute strides of the 5D swizzled tensor.
            swizzled_shape = (mxE, mxN // 128, mxK // 4, 32, 4, 4)
            s5 = 1
            s4 = swizzled_shape[5] * s5  # 4 * 1 = 4
            s3 = swizzled_shape[4] * s4  # 32 * 4 = 128
            s2 = swizzled_shape[3] * s3  # 4 * 128 = 512
            s1 = swizzled_shape[2] * s2  # (mxK//4) * 512
            s0 = swizzled_shape[1] * s1  # (mxN//128) * ((mxK//4)*512)
            return s0, s1, s2
        return scale.stride()

    def _compute_data_shape(self, handle, swizzle_mode):
        from .numerics_details.mxfp import SwizzlingType
        shape = list(handle.shape)
        if handle.dtype == torch.uint8:
            # Assume 2 fp4s packed into a byte
            shape[1] *= 2
        if swizzle_mode == SwizzlingType.HOPPER:
            shape[1] //= 4
            shape[2] *= 4
        return tuple(shape)

    def __init__(self, data, scale, axis=1, swizzle_mode=None):
        assert axis == 1
        self.data = data
        self.block_size = 32
        self.scale = scale
        self.ndim = data.ndim
        self.dtype = data.dtype
        self.device = data.device
        self.swizzle_mode = swizzle_mode
        self.scale_element_size = scale.element_size()
        self.data_shape = self._compute_data_shape(data, swizzle_mode)
        self.data_strides = data.stride()
        self.scale_shape = scale.shape
        self.scale_strides = self._compute_scale_stride(scale, swizzle_mode)
        self.shape = self.data_shape
        self._check_invariants()

    def stride(self, *args):
        return self.data.stride(*args)

    def element_size(self):
        return self.data.element_size()

    def data_ptr(self):
        return self.data.data_ptr()
