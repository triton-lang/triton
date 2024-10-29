from typing import List

TRITON_MAX_TENSOR_NUMEL = 1048576


def is_power_of_two(x):
    return (x & (x - 1)) == 0


def validate_block_shape(shape: List[int]):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]")
        if not is_power_of_two(d):
            raise ValueError(f"Shape element {i} must be a power of 2")
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel
