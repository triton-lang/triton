"""
Helper classes for working with low precision floating point types that
align with the opencompute (OCP) microscaling (MX) specification.
  * MXFP4Tensor: 4-bit E2M1 floating point data
  * MXFP6Tensor: 6-bit E2M3/E3M2 floating point data
  * MXScaleTensor: 8-bit E8M0 floating point data
Reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

import torch


class MXFP4Tensor:

    def __init__(self, data=None, size=None, device=None):
        """
        Tensor class for working with four bit E2M1 floating point data as defined by the
        opencompute microscaling specification.


        Parameters:
        - data: A torch tensor of float32 numbers to convert to fp4e2m1 microscaling format.
        - size: The size of the tensor to create.
        - device: The device on which to create the tensor.
        """
        self.device = device
        if data is not None:
            assert isinstance(data, torch.Tensor), "Parameter data must be a torch tensor"
            self.device = data.device
            self.data = self._from_float(data)
        elif size is not None:
            self.size = size if isinstance(size, tuple) else (size, )
        else:
            raise ValueError("Either parameter data or size must be provided")

    def random(self):
        S = torch.randint(0, 2, size=self.size, dtype=torch.uint8, device=self.device)
        E = torch.randint(0, 4, size=self.size, dtype=torch.uint8, device=self.device)
        M = torch.randint(0, 2, size=self.size, dtype=torch.uint8, device=self.device)

        self.data = ((S << 3) | (E << 1) | M).type(torch.uint8)
        return self

    def to(self, dtype):
        """
        Convert fp4e2m1 data to float32.

        Returns:
        - A torch tensor of type dtype representing the fp4e2m1 data.
        """
        assert dtype == torch.float32, "Currently only float32 is supported for fp4e2m1 to float conversion"

        data = self.data
        S = ((data >> 3) & 0x1).type(dtype)
        E = ((data >> 1) & 0x3).type(dtype)
        M = (data & 0x1).type(dtype)

        # The MXF4 E2M1 spec defines 0bS000 as zero
        value = torch.zeros_like(S)
        is_zero = (E == 0) & (M == 0)
        non_zero_mask = ~is_zero
        if non_zero_mask.any():
            S_nz = S[non_zero_mask]
            E_nz = E[non_zero_mask]
            M_nz = M[non_zero_mask]

            sign = torch.pow(-1, S_nz)
            # Normal and subnormal handling for the exponent and mantissa
            exponent = torch.where(E_nz == 0, E_nz, E_nz - 1)
            mantissa = torch.where(E_nz == 0, M_nz * 0.5, 1.0 + M_nz * 0.5)
            value_nz = sign * torch.pow(2, exponent) * mantissa

            value[non_zero_mask] = value_nz

        # For zeros, the values must remain zero with the correct sign
        value[is_zero & (S == 1)] *= -1
        return value.type(torch.float32)

    def _from_float(self, values):
        """
        Convert float32 numbers to mxf4 e2m1 format.
        * No encodings are reserved for Inf or NaN in mxf4.
        * Conversion from float supports roundTiesToEven rounding mode.
        * If a value exceeds the mxf4 representable range after rounding,
          clamps to the maximum mxf4 magnitude, preserving the sign.
        * If a value has magnitude less than the minimum subnormal magnitude
          in mxf4 after rounding, converts to zero.

        Parameters:
        - values: A torch tensor of float32 numbers to convert to fp4 format.
        """
        S = torch.signbit(values).type(torch.uint8)
        abs_values = torch.abs(values)

        is_zero = (abs_values == 0)
        is_invalid = torch.isnan(values) | torch.isinf(values)

        # Enumerate all possible E2M1 exponent and mantissa values. We will
        # use these to compare the distance between float32 and all possible
        # E2M1 floats to find the nearest E2M1 representable value
        E_bits = torch.tensor([0, 1, 2, 3], dtype=torch.uint8, device=self.device)
        M_bits = torch.tensor([0, 1], dtype=torch.uint8, device=self.device)

        candidate_values = []
        candidate_E = []
        candidate_M = []

        for E in E_bits:
            if E == 0:
                # Subnormals
                exponent = 0
                for M in M_bits:
                    significand = M * 0.5
                    value = significand * (2**exponent)
                    candidate_values.append(value)
                    candidate_E.append(E)
                    candidate_M.append(M)
            else:
                # Normals
                exponent = E.item() - 1
                for M in M_bits:
                    significand = 1.0 + M * 0.5
                    value = significand * (2**exponent)
                    candidate_values.append(value)
                    candidate_E.append(E)
                    candidate_M.append(M)

        candidates = torch.tensor(candidate_values, dtype=torch.float32, device=self.device)
        candidate_E = torch.tensor(candidate_E, dtype=torch.uint8, device=self.device)
        candidate_M = torch.tensor(candidate_M, dtype=torch.uint8, device=self.device)

        abs_values_flat = abs_values.view(-1)
        N = abs_values_flat.shape[0]
        abs_values_expanded = abs_values_flat.unsqueeze(1)

        # Clamp invalid values to the max e2m1 representable value
        max_candidate_value = candidates.max().item()
        abs_values_flat[is_invalid.view(-1)] = max_candidate_value

        # Compute distance between all abs_values and candidate e2m1 values
        errors = torch.abs(abs_values_expanded - candidates.unsqueeze(0))

        # To implement roundTiesToEven, we need to break ties by preferring
        # even mantissas (M == 0). We do so by adding an epsilon bias to shift
        # the closest candidate with an even mantissa closer to the float value
        min_errors, _ = torch.min(errors, dim=1, keepdim=True)
        is_tie = (errors == min_errors)
        # More than one candidate has the min error for some float value
        if is_tie.sum() > 1:
            M_bits_expanded = candidate_M.unsqueeze(0).expand(N, -1)
            tie_breaker = (M_bits_expanded == 0).type(torch.int32)

            errors = errors - (tie_breaker * 1e-6)

        best_indices = torch.argmin(errors, dim=1)

        E_selected = candidate_E[best_indices]
        M_selected = candidate_M[best_indices]
        E = E_selected.view(abs_values.shape)
        M = M_selected.view(abs_values.shape)

        E[is_zero] = 0
        M[is_zero] = 0

        return ((S << 3) | (E << 1) | M).type(torch.uint8)

    def to_packed_tensor(self, dim):
        """
        Packs two e2m1 elements into a single uint8 along the specified dimension.

        Parameters:
        - dim: The dimension along which to pack the elements.

        Returns:
        - A torch tensor of dtype uint8 with two e2m1 elements packed into one uint8.
        """
        data = self.data
        assert 0 <= dim < data.ndim, \
            "The dimension to pack along is not within the range of tensor dimensions"

        size_along_dim = data.size(dim)
        new_size_along_dim = (size_along_dim + 1) // 2

        # If the size is odd, we pad the data along dim with zeros at the end
        if size_along_dim % 2 != 0:
            pad_sizes = [0] * (2 * data.ndim)
            pad_index = (data.ndim - dim - 1) * 2 + 1
            pad_sizes[pad_index] = 1
            data = torch.nn.functional.pad(data, pad_sizes, mode='constant', value=0)

        new_shape = list(data.shape)
        new_shape[dim] = new_size_along_dim
        new_shape.insert(dim + 1, 2)  # packed dimension of length 2
        data = data.reshape(*new_shape)

        low = data.select(dim + 1, 0)
        high = data.select(dim + 1, 1)
        packed = (high << 4) | low

        return packed

    def unpack_packed_tensor(self, packed_tensor, dim, original_shape):
        """
        Unpacks a tensor where two fp4 elements are packed into a single uint8.

        Parameters:
        - packed_tensor: The packed tensor
        - dim: The dimension along which the tensor was packed.
        - original_shape: The shape of the original tensor before packing.

        Returns:
        - A tensor with the original data unpacked into uint8 elements containing one
          fp4e2m1 element in the least significant bits.
        """
        high = (packed_tensor >> 4) & 0xF
        low = packed_tensor & 0xF

        stacked = torch.stack((low, high), dim=dim + 1)

        # Flatten along dim and dim+1 and then merge
        shape = list(stacked.shape)
        new_shape = shape[:dim] + [shape[dim] * 2] + shape[dim + 2:]
        data = stacked.reshape(*new_shape)

        # Remove any padding
        if original_shape[dim] % 2 != 0:
            indices = [slice(None)] * data.ndim
            indices[dim] = slice(0, original_shape[dim])
            data = data[tuple(indices)]

        return data.type(torch.uint8)


class MXFP6Tensor:

    def __init__(self, data=None, size=None, device=None, e=2):
        """
        Tensor class for working with six bit E2M3/E3M2 floating point data as defined by the
        opencompute microscaling specification.


        Parameters:
        - data: A torch tensor of float32 numbers to convert to fp6e2m3/fp6e3m2 microscaling format.
        - size: The size of the tensor to create.
        - device: The device on which to create the tensor.
        """
        self.device = device
        self.s_shift = 5
        self.e_size = e
        self.m_size = 5 - e
        if data is not None:
            assert isinstance(data, torch.Tensor), "Parameter data must be a torch tensor"
            self.device = data.device
            self.data = self._from_float(data)
        elif size is not None:
            self.size = size if isinstance(size, tuple) else (size, )
        else:
            raise ValueError("Either parameter data or size must be provided")

    def ravel(self):
        return self.data.ravel()

    def random(self):
        upper_bound = 7.5 if self.e_size == 2 else 28.0
        data = upper_bound * torch.rand(self.size, dtype=torch.float32, device=self.device)
        self.data = self._from_float(data)
        return self

    def to(self, dtype):
        """
        Convert fp6e2m3/fp6e3m2 data to float32.

        Returns:
        - A torch tensor of type dtype representing the fp6e2m3/fp6e3m2 data.
        """
        assert dtype == torch.float32, "Currently only float32 is supported for fp6e2m3/fp6e3m2 to float conversion"

        data = self.data
        E_shift = self.m_size
        E_mask = 0x3 if self.e_size == 2 else 0x7
        M_mask = 0x3 if self.m_size == 2 else 0x7

        S = ((data >> self.s_shift) & 0x1).type(dtype)
        E = ((data >> E_shift) & E_mask).type(dtype)
        M = (data & M_mask).type(dtype)

        # The MXF6 E2M3 or E3M2 spec defines 0bS00000 as zero
        value = torch.zeros_like(S)
        is_zero = (E == 0) & (M == 0)
        non_zero_mask = ~is_zero
        if non_zero_mask.any():
            S_nz = S[non_zero_mask]
            E_nz = E[non_zero_mask]
            M_nz = M[non_zero_mask]

            sign = torch.pow(-1, S_nz)
            # Normal and subnormal handling for the exponent and mantissa
            E_bias = 1 if self.e_size == 2 else 3
            M_factor = 0.25 if self.m_size == 2 else 0.125
            exponent = torch.where(E_nz == 0, E_nz, E_nz - E_bias)
            mantissa = torch.where(E_nz == 0, M_nz * M_factor, 1.0 + M_nz * M_factor)
            value_nz = sign * torch.pow(2, exponent) * mantissa

            value[non_zero_mask] = value_nz

        # For zeros, the values must remain zero with the correct sign
        value[is_zero & (S == 1)] *= -1
        return value.type(torch.float32)

    def _from_float(self, values):
        """
        Convert float32 numbers to mxf6 e2m3/e3m2 format.
        * No encodings are reserved for Inf or NaN in mxf6.
        * Conversion from float supports roundTiesToEven rounding mode.
        * If a value exceeds the mxf6 representable range after rounding,
          clamps to the maximum mxf6 magnitude, preserving the sign.
        * If a value has magnitude less than the minimum subnormal magnitude
          in mxf6 after rounding, converts to zero.

        Parameters:
        - values: A torch tensor of float32 numbers to convert to fp6e2m3 format.
        """
        S = torch.signbit(values).type(torch.uint8)
        abs_values = torch.abs(values)

        is_zero = (abs_values == 0)
        is_invalid = torch.isnan(values) | torch.isinf(values)

        # Enumerate all possible E2M3/E3M2 exponent and mantissa values. We will
        # use these to compare the distance between float32 and all possible
        # E2M3/E3M2 floats to find the nearest E2M3/E3M2 representable value
        E_shift = self.m_size
        E_range = 4 if self.e_size == 2 else 8
        E_bits = torch.arange(0, E_range, dtype=torch.uint8, device=self.device)
        M_range = 4 if self.m_size == 2 else 8
        M_factor = 0.25 if self.m_size == 2 else 0.125
        M_bits = torch.arange(0, M_range, dtype=torch.uint8, device=self.device)
        E_bias = 1 if self.e_size == 2 else 3

        candidate_values = []
        candidate_E = []
        candidate_M = []

        for E in E_bits:
            if E == 0:
                # Subnormals
                exponent = 0
                for M in M_bits:
                    significand = M * M_factor
                    value = significand * (2**exponent)
                    candidate_values.append(value)
                    candidate_E.append(E)
                    candidate_M.append(M)
            else:
                # Normals
                exponent = E.item() - E_bias
                for M in M_bits:
                    significand = 1.0 + M * M_factor
                    value = significand * (2**exponent)
                    candidate_values.append(value)
                    candidate_E.append(E)
                    candidate_M.append(M)

        candidates = torch.tensor(candidate_values, dtype=torch.float32, device=self.device)
        candidate_E = torch.tensor(candidate_E, dtype=torch.uint8, device=self.device)
        candidate_M = torch.tensor(candidate_M, dtype=torch.uint8, device=self.device)

        abs_values_flat = abs_values.view(-1)
        N = abs_values_flat.shape[0]
        abs_values_expanded = abs_values_flat.unsqueeze(1)

        # Clamp invalid values to the max e2m3/e3m2 representable value
        max_candidate_value = candidates.max().item()

        abs_values_flat[is_invalid.view(-1)] = max_candidate_value

        # Compute distance between all abs_values and candidate e2m1 values
        errors = torch.abs(abs_values_expanded - candidates.unsqueeze(0))

        # To implement roundTiesToEven, we need to break ties by preferring
        # even mantissas (M == 0). We do so by adding an epsilon bias to shift
        # the closest candidate with an even mantissa closer to the float value
        min_errors, _ = torch.min(errors, dim=1, keepdim=True)
        is_tie = (errors == min_errors)
        # More than one candidate has the min error for some float value
        if torch.any(is_tie):
            M_bits_expanded = candidate_M.unsqueeze(0).expand(N, -1)
            tie_breaker = (M_bits_expanded == 0).type(torch.int32)

            errors = errors - (tie_breaker * 1e-6)

        best_indices = torch.argmin(errors, dim=1)

        E_selected = candidate_E[best_indices]
        M_selected = candidate_M[best_indices]
        E = E_selected.view(abs_values.shape)
        M = M_selected.view(abs_values.shape)

        E[is_zero] = 0
        M[is_zero] = 0

        return ((S << self.s_shift) | (E << E_shift) | M).type(torch.uint8)

    def to_packed_tensor(self, dim):
        """
        Packs 32 e2m3 elements into 24 uint8's along the specified dimension.
        and padding the last 8 uint8's with 0's

        Parameters:
        - dim: The dimension along which to pack the elements.

        Returns:
        - A torch tensor of dtype uint8 with 32 e2m3 elements packed into one uint8.
        """
        data = self.data
        orig_shape = data.shape
        assert 0 <= dim < data.ndim, \
            "The dimension to pack along is not within the range of tensor dimensions"
        size_along_dim = data.size(dim)
        assert size_along_dim % 4 == 0, "Dim must be multiple of 4"
        new_size_along_dim = size_along_dim // 4
        new_shape = list(data.shape)
        new_shape[dim] = new_size_along_dim
        # reshape to the dim dimension to dim/4, 4
        new_shape.insert(dim + 1, 4)
        data = data.reshape(*new_shape)

        elem0 = data.select(dim + 1, 0)
        elem1 = data.select(dim + 1, 1)
        elem2 = data.select(dim + 1, 2)
        elem3 = data.select(dim + 1, 3)

        # pack the 4 fp6e2m3 into 3 uint8's
        out0 = (elem1 << 6) | elem0
        out1 = (elem2 << 4) | ((elem1 >> 2) & 0xF)
        out2 = elem3 << 2 | ((elem2 >> 4) & 0x3)
        out3 = torch.zeros_like(elem0)

        packed = torch.stack((out0, out1, out2, out3), dim=dim + 1)
        packed = packed.reshape(orig_shape)

        return packed

    def unpack_packed_tensor(self, packed_tensor, dim):
        """
        Unpacks a tensor where 4 fp6e2m3 elements are packed into 3 uint8.

        Parameters:
        - packed_tensor: The packed tensor
        - dim: The dimension along which the tensor was packed.
        - original_shape: The shape of the original tensor before packing.

        Returns:
        - A tensor with the original data unpacked into uint8 elements containing one
          fp6e2m3 element in the least significant bits.
        """
        original_shape = packed_tensor.shape
        size_along_dim = packed_tensor.size(dim)
        assert size_along_dim % 4 == 0, "Dim must be multiple of 4"
        new_size_along_dim = size_along_dim // 4
        new_shape = list(packed_tensor.shape)
        new_shape[dim] = new_size_along_dim

        # reshape to the dim dimension to dim/4, 4
        new_shape.insert(dim + 1, 4)
        packed_tensor = packed_tensor.reshape(*new_shape)

        elem0 = packed_tensor.select(dim + 1, 0)
        elem1 = packed_tensor.select(dim + 1, 1)
        elem2 = packed_tensor.select(dim + 1, 2)
        packed_tensor.select(dim + 1, 3)

        out0 = elem0 & 0x3F
        out1 = ((elem1 << 2) & 0x3C) | ((elem0 >> 6) & 0x3)
        out2 = ((elem2 << 4) & 0x30) | ((elem1 >> 4) & 0xF)
        out3 = (elem2 >> 2) & 0x3F

        unpacked = torch.stack((out0, out1, out2, out3), dim=dim + 1)
        unpacked = unpacked.reshape(original_shape)

        return unpacked.type(torch.uint8)


class MXScaleTensor:

    def __init__(self, data=None, size=None, device=None):
        """
        Tensor class for working with microscaling E8M0 block scale factors.

        Parameters:
        - data: A torch tensor of float32 numbers to convert to fp8e8m0 microscaling format.
        - size: The size of the tensor to create.
        - device: The device on which to create the tensor.
        """
        self.device = device
        if data is not None:
            assert isinstance(data, torch.Tensor), "Parameter data must be a torch tensor"
            self.device = data.device
            self.data = self._from_float(data)
        elif size is not None:
            self.size = size if isinstance(size, tuple) else (size, )
        else:
            raise ValueError("Either parameter data or size must be provided")

    def random(self, low=None, high=None):
        """
        Generate random E8M0 data within a specified range.
        * Excludes the NaN encoding (255).
        """
        bias = 127

        min_exponent = 0 if low is None else max(0, int(torch.log2(torch.tensor(low))) + bias)
        max_exponent = 254 if high is None else min(254, max(0, int(torch.log2(torch.tensor(high))) + bias))
        assert min_exponent <= max_exponent, "Low must be less than or equal to high"

        E = torch.randint(min_exponent, max_exponent + 1, size=self.size, dtype=torch.uint8, device=self.device)
        self.data = E
        return self

    def to(self, dtype):
        assert dtype == torch.float32, "Currently only float32 is supported for f8e8m0 to float conversion"
        data = self.data.type(dtype)
        is_nan = (data == 255)
        e_biased = data.clone()
        e_biased[is_nan] = 0
        e = e_biased - 127
        value = torch.pow(2.0, e)
        value[is_nan] = torch.nan
        return value.type(dtype)

    def _from_float(self, values):
        """
        Convert float32 numbers to E8M0 format.
        * Values <= 0, NaNs, and Infs are converted to the NaN encoding (255).
        * Positive values are converted by computing the floor of log2(value) to get the exponent.

        Parameters:
        - values: A torch tensor of float32 numbers to convert to E8M0 format.
        """
        result = torch.empty_like(values, dtype=torch.uint8, device=self.device)

        is_invalid = torch.isnan(values) | torch.isinf(values) | (values <= 0)
        result[is_invalid] = 255

        valid_values = values[~is_invalid]
        e = torch.floor(torch.log2(valid_values))
        e_biased = e + 127
        e_biased_int = e_biased.type(torch.int32)
        e_biased_clamped = torch.clamp(e_biased_int, 0, 254)
        result[~is_invalid] = e_biased_clamped.type(torch.uint8)

        return result
