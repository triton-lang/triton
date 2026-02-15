from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutTransformation(ABC):

    shape: list[int]
    is_fp4: bool

    @abstractmethod
    def swizzle_data(self, data):
        pass

    @abstractmethod
    def unswizzle_data(self, data):
        pass


@dataclass(frozen=True)
class Layout(ABC):

    @abstractmethod
    def make_transformation(self, shape: list[int]) -> LayoutTransformation:
        pass

    @abstractmethod
    def swizzle_block_shape(self, block_shape):
        pass
