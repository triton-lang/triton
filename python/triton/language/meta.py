from typing import List

import triton.language as tl


def _i_globaltimer(builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_globaltimer, tl.int64)


@tl.builtin
def globaltimer(_builder: tl.ir.builder = None) -> tl.tensor:
    return _i_globaltimer(_builder)


def _i_clock(builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_clock(), tl.int64)


@tl.builtin
def clock(_builder: tl.ir.builder = None) -> tl.tensor:
    return _i_clock(_builder)


def _i_debug_barrier(builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_barrier(""), tl.void)


@tl.builtin
def debug_barrier(_builder: tl.ir.builder = None) -> tl.tensor:
    return _i_debug_barrier(_builder)


def _i_program_id(axis: int, builder: tl.ir.builder) -> tl.tensor:
    axis = tl._constexpr_to_value(axis)
    return tl.tensor(builder.create_get_program_id(axis), tl.int32)


@tl.builtin
def program_id(axis, _builder: tl.ir.builder = None):
    """
    Returns the id of the current program instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    # if axis == -1:
    #     pid0 = program_id(0, _builder)
    #     pid1 = program_id(1, _builder)
    #     pid2 = program_id(2, _builder)
    #     npg0 = num_programs(0, _builder)
    #     npg1 = num_programs(0, _builder)
    #     return pid0 + pid1*npg0 + pid2*npg0*npg1
    axis = tl._constexpr_to_value(axis)
    return _i_program_id(axis, _builder)


def _i_num_programs(axis: int, builder: tl.ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_get_num_programs(axis), tl.int32)


@tl.builtin
def num_programs(axis, _builder: tl.ir.builder = None):
    """
    Returns the number of program instances launched along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    axis = tl._constexpr_to_value(axis)
    return _i_num_programs(axis, _builder)


def _i_multiple_of(x: tl.tensor, values: List[int]) -> tl.tensor:
    if len(x.shape) != len(values):
        raise ValueError(
            "Shape of input to multiple_of does not match the length of values"
        )
    x.handle.set_attr(
        "tt.divisibility", tl.ir.make_attr(values, x.handle.get_context())
    )
    return x


@tl.builtin
def multiple_of(input, values, _builder: tl.ir.builder = None):
    """
    Let the compiler knows that the values in :code:`input` are all multiples of :code:`value`.
    """
    if isinstance(values, tl.constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, tl.constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(
                f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]"
            )
    values = [x.value for x in values]
    return _i_multiple_of(input, values)


def _i_max_contiguous(x: tl.tensor, values: List[int]) -> tl.tensor:
    if len(x.shape) != len(values):
        raise ValueError(
            "Shape of input to max_contiguous does not match the length of values"
        )
    x.handle.set_attr("tt.contiguity", tl.ir.make_attr(values, x.handle.get_context()))
    return x


@tl.builtin
def max_contiguous(input, values, _builder: tl.ir.builder = None):
    """
    Let the compiler knows that the `value` first values in :code:`input` are contiguous.
    """
    if isinstance(values, tl.constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, tl.constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(
                f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]"
            )
    values = [x.value for x in values]
    return _i_max_contiguous(input, values)
