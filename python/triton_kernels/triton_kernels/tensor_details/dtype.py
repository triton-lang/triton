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
BF16 = FloatType(bitwidth_exponent=8, bitwidth_mantissa=7, is_signed=True)
FP16 = FloatType(bitwidth_exponent=5, bitwidth_mantissa=10, is_signed=True)
FP32 = FloatType(bitwidth_exponent=8, bitwidth_mantissa=23, is_signed=True)
FP64 = FloatType(bitwidth_exponent=11, bitwidth_mantissa=52, is_signed=True)

DataType: TypeAlias = IntegerType | FloatType
