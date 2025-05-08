from dataclasses import dataclass


@dataclass
class Bitmatrix:
    data: "torch.Tensor"
    shape: tuple[int]
