from dataclasses import dataclass
from typing import TypeAlias


# data types
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntegerType:
    bitwidth: int
    is_signed: bool


@dataclass(frozen=True)
class FloatType:
    bitwidth_exponent: int
    bitwidth_mantissa: int
    is_signed: bool
    unsigned_zero: bool = False

    @property
    def bitwidth(self):
        return int(self.is_signed) + self.bitwidth_exponent + self.bitwidth_mantissa


BIT = IntegerType(1, is_signed=False)
UINT8 = IntegerType(8, is_signed=False)
FP4 = FloatType(bitwidth_exponent=2, bitwidth_mantissa=1, is_signed=True)
FP8_E4M3FN = FloatType(bitwidth_exponent=4, bitwidth_mantissa=3, is_signed=True)
FP8_E4M3FNUZ = FloatType(bitwidth_exponent=4, bitwidth_mantissa=3, is_signed=True, unsigned_zero=True)
FP8_E5M2 = FloatType(bitwidth_exponent=5, bitwidth_mantissa=2, is_signed=True)
FP8_E5M2FNUZ = FloatType(bitwidth_exponent=5, bitwidth_mantissa=2, is_signed=True, unsigned_zero=True)
BF16 = FloatType(bitwidth_exponent=8, bitwidth_mantissa=7, is_signed=True)
FP16 = FloatType(bitwidth_exponent=5, bitwidth_mantissa=10, is_signed=True)
FP32 = FloatType(bitwidth_exponent=8, bitwidth_mantissa=23, is_signed=True)
FP64 = FloatType(bitwidth_exponent=11, bitwidth_mantissa=52, is_signed=True)
INT16 = IntegerType(16, is_signed=True)
INT32 = IntegerType(32, is_signed=True)
INT64 = IntegerType(64, is_signed=True)

DataType: TypeAlias = IntegerType | FloatType


def promote_dtype(lhs_dtype: FloatType, rhs_dtype: FloatType) -> FloatType:
    """Returns a lossless common float type, preferring FP16 for distinct FP8 formats."""
    if lhs_dtype == rhs_dtype:
        return lhs_dtype
    exponent = max(lhs_dtype.bitwidth_exponent, rhs_dtype.bitwidth_exponent)
    mantissa = max(lhs_dtype.bitwidth_mantissa, rhs_dtype.bitwidth_mantissa)
    for dtype in (FP16, BF16, FP32, FP64):
        if dtype.bitwidth_exponent >= exponent and dtype.bitwidth_mantissa >= mantissa:
            return dtype
    raise ValueError(f"Cannot losslessly promote {lhs_dtype} and {rhs_dtype}")
