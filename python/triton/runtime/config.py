from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..compiler.compiler import ASTSource, IRSource


@dataclass(frozen=True)
class CompileTimes:
    """
    Model holding timing information for an invocation of the compiler.

    All times in microseconds.
    """

    # Duration of make_ir
    prologue: int

    # Ordered mapping from lowering stage to duration spent in that stage.
    # Keyed by stage extension, e.g. ttir, ttgir
    lowering_stages: list[tuple[str, int]]

    # Duration of post-lowering
    epilogue: int

    @property
    def total_lowering(self) -> int:
        return sum((stage[1] for stage in self.lowering_stages))

    @property
    def total(self) -> int:
        return self.prologue + self.total_lowering + self.epilogue


class CompilationListener(Protocol):

    def __call__(self, *, src: Union[ASTSource, IRSource], metadata: dict[str, Any], times: CompileTimes,
                 cache_hit: bool) -> None:
        ...


class TritonConfig:
    compilation_listener: CompilationListener | None = None
