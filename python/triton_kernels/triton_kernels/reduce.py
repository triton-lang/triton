import torch
from dataclasses import dataclass
from typing import Optional
from .reduce_details import reduce_grouped as _reduce_grouped
from .reduce_details import reduce_standard as _reduce_standard


@dataclass
class ReductionSpecs:
    dim: int
    group_indx: Optional[torch.Tensor] = None
    expensive_checks: bool = False

    # input-agnostic checks
    def _check_group_indx(self):
        if self.group_indx is not None:
            vals = self.group_indx[self.group_indx >= 0]
            if vals.numel() > 0 and torch.unique(vals).numel() != vals.numel():
                raise ValueError("reduce: duplicate indices detected across groups (aliasing)")

    def __post_init__(self):
        if self.group_indx is not None:
            if self.group_indx.ndim != 2:
                raise ValueError("group_indx must be 2D")
            if self.expensive_checks:
                self._check_group_indx()

    @property
    def mode(self):
        if self.group_indx is not None:
            return "grouped"
        return "standard"


def _validate(x: torch.Tensor, specs: ReductionSpecs, inplace: bool):
    if specs.group_indx is not None:
        if specs.dim == x.ndim - 1:
            return NotImplementedError("group_indx only supported for dim = -1")
        if x.shape[specs.dim] != specs.group_indx.numel():
            return ValueError("group_indx must have the same number of elements as the reduction dim")
    if specs.mode == "grouped" and not inplace:
        raise NotImplementedError("grouped reduction must be inplace")


def reduce(x: torch.Tensor, specs: ReductionSpecs, inplace: bool = False):
    _validate(x, specs, inplace)
    if specs.mode == "grouped":
        return _reduce_grouped.reduce_grouped(x, indx=specs.group_indx)
    if specs.mode == "standard":
        return _reduce_standard.reduce_standard(x, dim=specs.dim)
    assert False


def reduce_torch(x: torch.Tensor, specs: ReductionSpecs, inplace: bool = True):
    _validate(x, specs, inplace)
    if specs.mode == "grouped":
        return _reduce_grouped.reduce_grouped_torch(x, specs.group_indx)
    if specs.mode == "standard":
        return _reduce_standard.reduce_standard_torch(x, dim=specs.dim)
    assert False
