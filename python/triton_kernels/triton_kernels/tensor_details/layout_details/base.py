from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class LayoutTransformation(ABC):
    shape: list[int]
    is_fp4: bool

    @abstractmethod
    def swizzle_data(self, data: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def unswizzle_data(self, data: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class Layout(ABC):

    @abstractmethod
    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        ...

    @abstractmethod
    def swizzle_block_shape(self, block_shape: Sequence[int]) -> Sequence[int]:
        ...
