from typing import NamedTuple

import triton.language as tl


class Closure(NamedTuple):
    fn: tl.constexpr
    captured: tuple
