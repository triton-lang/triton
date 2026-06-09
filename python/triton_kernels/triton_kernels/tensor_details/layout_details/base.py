from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutTransformation(ABC):

    shape: list[int]
    is_fp4: bool

    @property
    def storage_shape(self) -> list[int]:
        """Physical storage shape produced by this transformation."""
        raise NotImplementedError

    def _validate_storage_shape(self, data):
        assert list(data.shape) == self.storage_shape
        return data

    @abstractmethod
    def swizzle_data(self, data):
        pass

    @abstractmethod
    def unswizzle_data(self, data):
        pass


@dataclass(frozen=True)
class Layout(ABC):

    def can_preserve_storage_as(self, other: "Layout", rank: int) -> bool:
        """Whether existing storage is already valid for `other`."""
        return self == other

    def storage_shape(self, shape: list[int], is_fp4: bool) -> list[int]:
        """Return the physical storage shape for a logical tensor shape."""
        return self.make_transformation(shape, is_fp4).storage_shape

    @abstractmethod
    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        pass

    @abstractmethod
    def swizzle_block_shape(self, block_shape):
        pass
