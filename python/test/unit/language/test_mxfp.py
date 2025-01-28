import pytest
import torch


class MXBaseTest:

    @pytest.fixture
    def device(self):
        return "cpu"


class TestMXFP4Tensor(MXBaseTest):

    @pytest.mark.parametrize("K, N", [(64, 128), (128, 256)])
    def test_roundtrip(self, K, N, device):
        tensor = MXFP4Tensor(size=(K, N), device=device).random()
        tensor2 = MXFP4Tensor(tensor.to(torch.float32))
        torch.testing.assert_close(tensor.data, tensor2.data)

    @pytest.mark.parametrize("K, N, dim", [(64, 128, 0), (64, 128, 1)])
    def test_packed_tensor(self, K, N, dim, device):
        tensor = MXFP4Tensor(size=(K, N), device=device).random()
        packed = tensor.to_packed_tensor(dim=dim)
        unpacked = tensor.unpack_packed_tensor(packed, dim=dim, original_shape=(K, N))
        torch.testing.assert_close(tensor.data, unpacked)

    def test_padding(self, device):
        tensor_pad = MXFP4Tensor(torch.tensor([4], device=device))
        pad_packed = tensor_pad.to_packed_tensor(dim=0)
        torch.testing.assert_close(tensor_pad.data,
                                   tensor_pad.unpack_packed_tensor(pad_packed, dim=0, original_shape=(1, )))

    def test_zero_values(self, device):
        test_values = torch.tensor([0.0, -0.0], device=device)
        tensor = MXFP4Tensor(test_values)
        expected_encodings = torch.tensor([0b0000, 0b1000], dtype=torch.uint8, device=device)
        assert torch.equal(tensor.data, expected_encodings), "Zero values should be encoded as 0"
        torch.testing.assert_close(tensor.to(torch.float32), test_values)

    def test_out_of_range_values(self, device):
        test_values = torch.tensor([7.0, -7.0, float('inf'), float('-inf')], device=device)
        tensor = MXFP4Tensor(test_values)
        expected_values = torch.tensor([6.0, -6.0, 6.0, -6.0], device=device)
        torch.testing.assert_close(tensor.to(torch.float32), expected_values)

    def test_subnormal_numbers(self, device):
        test_values = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
        tensor = MXFP4Tensor(test_values)
        expected_values = torch.tensor([0.0, 0.0, 0.5, 0.5], device=device)
        torch.testing.assert_close(tensor.to(torch.float32), expected_values)

    def test_rounding_edge_cases(self, device):
        test_values = torch.tensor([0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=device)
        expected_values = torch.tensor([1.0, 1.0, 2.0, 2.0, 4.0, 4.0], device=device)
        tensor = MXFP4Tensor(test_values)
        torch.testing.assert_close(tensor.to(torch.float32), expected_values)

    def test_negative_values(self, device):
        test_values = torch.tensor([-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=device)
        tensor = MXFP4Tensor(test_values)
        torch.testing.assert_close(tensor.to(torch.float32), test_values)

    def test_negative_out_of_range(self, device):
        tensor = MXFP4Tensor(torch.tensor([-7.0, -8.0, -10.0], device=device))
        expected_values = torch.tensor([-6.0, -6.0, -6.0], device=device)
        torch.testing.assert_close(tensor.to(torch.float32), expected_values)

    @pytest.mark.parametrize("shape, dim", [
        ((1024, ), 0),
        ((128, 256), 0),
        ((128, 256), 1),
        ((64, 64, 64), 2),
    ])
    def test_packing(self, shape, dim, device):
        tensor = MXFP4Tensor(size=shape, device=device).random()
        packed = tensor.to_packed_tensor(dim=dim)
        unpacked = tensor.unpack_packed_tensor(packed, dim=dim, original_shape=shape)
        torch.testing.assert_close(tensor.data, unpacked)

    def test_packing_with_padding(self, device):
        shape = (7, 5)
        dim = 1
        tensor = MXFP4Tensor(size=shape, device=device).random()
        packed = tensor.to_packed_tensor(dim=dim)
        unpacked = tensor.unpack_packed_tensor(packed, dim=dim, original_shape=shape)
        torch.testing.assert_close(tensor.data, unpacked)

    def test_invalid_packing_dimension(self, device):
        tensor = MXFP4Tensor(size=(4, 4), device=device).random()
        with pytest.raises(AssertionError):
            tensor.to_packed_tensor(dim=2)  # Invalid dimension

    def test_empty_tensor(self, device):
        tensor = MXFP4Tensor(torch.tensor([], device=device))
        assert tensor.to(torch.float32).numel() == 0


class TestMXScaleTensor(MXBaseTest):

    def test_positive_values(self, device):
        values = torch.tensor([1.0, 2.0, 4.0, 8.0], device=device)
        data = MXScaleTensor(values)
        torch.testing.assert_close(data.to(torch.float32), values)

    def test_special_values(self, device):
        values = torch.tensor([0.0, -1.0, float('nan'), float('inf'), float('-inf')], device=device)
        tensor = MXScaleTensor(values)
        expected_data = torch.tensor([255, 255, 255, 255, 255], dtype=torch.uint8, device=device)
        assert torch.equal(expected_data, tensor.data), "Special values should be encoded as NaN (255)"

    def test_e8m0_nan_to_float_nan(self, device):
        tensor = MXScaleTensor(size=(1, ), device=device)
        tensor.data = torch.tensor([255], device=device, dtype=torch.uint8)
        assert torch.isnan(tensor.to(torch.float32)), "E8M0 NaN encoding should convert to float32 NaN"

    def test_random_generation(self, device):
        data = MXScaleTensor(size=(1000, ), device=device).random()
        data = data.data
        assert ((data >= 0) & (data <= 254)).all(), "Generated data should be between 0 and 254"
        assert (data != 255).all(), "Generated data should not include NaN encoding (255)"

    @pytest.mark.parametrize("K, N", [(64, 128), (128, 256)])
    def test_roundtrip(self, K, N, device):
        tensor = MXScaleTensor(size=(K, N), device=device).random()
        tensor2 = MXScaleTensor(tensor.to(torch.float32))
        torch.testing.assert_close(tensor.data, tensor2.data)


class MXFP4Tensor:

    def __init__(self, data=None, size=None, device=None):
        """
        Tensor class for working with four bit E2M1 floating point data as defined by the
        opencompute microscaling specification. c.f.
        https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

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
        S = torch.randint(0, 2, size=self.size, dtype=torch.uint8, device=self.device)  # 1 bit
        E = torch.randint(0, 4, size=self.size, dtype=torch.uint8, device=self.device)  # 2 bits
        M = torch.randint(0, 2, size=self.size, dtype=torch.uint8, device=self.device)  # 1 bit

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
