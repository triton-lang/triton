import pytest

import triton
from triton._utils import is_power_of_two, validate_block_shape


@pytest.mark.parametrize("y", [1, 32])
def test_cdiv_symbolic_numerator(y):

    class Expr:

        def __init__(self, evaluate):
            self.evaluate = evaluate

        def __add__(self, other):
            return Expr(lambda x: self.evaluate(x) + other)

        def __floordiv__(self, other):
            return Expr(lambda x: self.evaluate(x) // other)

    result = triton.cdiv(Expr(lambda x: x), y)
    assert isinstance(result, Expr)
    for x in [0, 1, 31, 32, 33, 64, 65]:
        assert result.evaluate(x) == -(-x // y)


@pytest.mark.parametrize("x", [-33, -1, 0, 1, 31, 32, 33, 2**100, 2**100 + 1])
@pytest.mark.parametrize("y", [1, 2, 32, 2**64])
def test_cdiv_integers(x, y):
    assert triton.cdiv(x, y) == -(-x // y)


def test_cdiv_zero_divisor():
    with pytest.raises(ZeroDivisionError):
        triton.cdiv(1, 0)


def test_is_power_of_two():
    assert is_power_of_two(1)
    assert is_power_of_two(2)
    assert is_power_of_two(8)
    assert is_power_of_two(1024)
    # 0 is not a power of two; x & (x - 1) == 0 alone wrongly accepts it.
    assert not is_power_of_two(0)
    assert not is_power_of_two(3)
    assert not is_power_of_two(6)
    assert not is_power_of_two(-4)


def test_validate_block_shape_rejects_zero():
    # validate_block_shape promises every element is a power of 2, but a 0
    # element used to slip through because is_power_of_two(0) returned True.
    with pytest.raises(ValueError, match="must be a power of 2"):
        validate_block_shape([0])
    with pytest.raises(ValueError, match="must be a power of 2"):
        validate_block_shape([8, 0])


def test_validate_block_shape_accepts_powers_of_two():
    assert validate_block_shape([8, 16]) == 128
