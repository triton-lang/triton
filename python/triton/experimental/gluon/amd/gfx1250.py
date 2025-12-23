from dataclasses import dataclass
from typing import List, Any
from triton._utils import validate_block_shape
from triton.experimental.gluon.language._layouts import PaddedSharedLayout, SwizzledSharedLayout

__all__ = ["TensorDescriptor"]


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    layout: PaddedSharedLayout | SwizzledSharedLayout
    padding: str = "zero"

    def __post_init__(self):
        ndim = len(self.shape)
        assert 1 <= ndim <= 5, f"Expected 1-5 dimensions but got {ndim} dimensions"
        assert len(self.strides) == ndim, f"Expected {ndim} strides but got {len(self.strides)}"
        assert len(self.block_shape) == ndim, \
            f"Expected block_shape to have {ndim} dimensions but got {len(self.strides)}"
        validate_block_shape(self.block_shape)
        assert self.strides[-1] == 1, "Last dimension must be contiguous"
        assert isinstance(self.layout, (PaddedSharedLayout, SwizzledSharedLayout)), \
            "Expected layout to be a PaddedSharedLayout or SwizzledSharedLayout"
        if isinstance(self.layout, SwizzledSharedLayout):
            assert self.layout.max_phase == 1, "Expected max_phase to be 1 for SwizzledSharedLayout"
        assert self.padding == "zero", "Only 'zero' padding is supported"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], layout: PaddedSharedLayout | SwizzledSharedLayout):
        """ Create a TensorDescriptor object from a tensor.

        Args:
            tensor (torch.Tensor): The input tensor.
            block_shape (List[int]): The block shape of the tensor.
            layout (PaddedSharedLayout | SwizzledSharedLayout): The layout of the tensor in shared memory.

        Returns:
            tensor_descriptor: the created TensorDescriptor object

        """
        return TensorDescriptor(tensor, tensor.shape, tensor.stride(), block_shape, layout)
