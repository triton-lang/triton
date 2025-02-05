import pytest
import torch
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


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
