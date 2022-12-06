from typing import List

import triton
import triton.language as tl


def _i_printf(
    prefix: str,
    args: List[tl.tensor],
    builder: tl.ir.builder,
) -> tl.tensor:
    new_args = []
    for arg in args:
        new_args.append(arg.handle)
    return tl.tensor(builder.create_printf(prefix, new_args), tl.void)


@triton.builtin
def printf(prefix, *args, _builder=None):
    import string

    new_prefix = prefix
    if isinstance(prefix, tl.constexpr):
        new_prefix = prefix.value
    assert isinstance(new_prefix, str), f"{new_prefix} is not string"
    b_ascii = True
    for ch in new_prefix:
        if ch not in string.printable:
            b_ascii = False
            break
    assert b_ascii, f"{new_prefix} is not an ascii string"
    new_args = []
    for arg in args:
        new_args.append(tl._to_tensor(arg, _builder))
    return _i_printf(new_prefix, new_args, _builder)
