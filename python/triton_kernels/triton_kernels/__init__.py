from dataclasses import dataclass


@dataclass
class Bitmatrix:
    data: "torch.Tensor"  # noqa: F821
    shape: tuple[int]
