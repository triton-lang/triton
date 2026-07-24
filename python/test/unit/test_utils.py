import pytest

from triton._utils import canonicalize_dtype, is_power_of_two, validate_block_shape


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


def test_canonicalize_packed_torch_dtypes_are_rejected():
    torch = pytest.importorskip("torch")
    # MXFP4 data and E8M0 scales have no Triton element type. Canonicalization
    # must refuse them with guidance rather than silently reinterpreting as uint8,
    # which would let later arithmetic run as integer math.
    for name in ("float4_e2m1fn_x2", "float8_e8m0fnu"):
        dtype = getattr(torch, name, None)
        if dtype is None:
            pytest.skip(f"torch has no {name}")
        with pytest.raises(TypeError, match="uint8"):
            canonicalize_dtype(dtype)
