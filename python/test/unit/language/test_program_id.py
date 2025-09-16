from types import SimpleNamespace

import pytest

pytest.importorskip("triton._C.libtriton")
import triton.language as tl


class StubBuilder:
    def __init__(self, width=32):
        self.options = SimpleNamespace(program_id_width=width)
        self.cast_args = None
        self.axis = None

    def create_get_program_id(self, axis):
        self.axis = axis
        return f"pid_{axis}"

    def create_int_cast(self, value, ty, is_signed):
        self.cast_args = (value, ty, is_signed)
        return f"cast_{value}"

    def get_int64_ty(self):
        return "i64"


def test_program_id_respects_backend_width():
    builder = StubBuilder(width=64)
    pid = tl.program_id(0, _builder=builder)
    assert pid.dtype == tl.int64
    assert builder.cast_args == ("pid_0", "i64", False)


def test_program_id_explicit_override():
    builder = StubBuilder(width=32)
    pid = tl.program_id(0, bitwidth=64, _builder=builder)
    assert pid.dtype == tl.int64
    assert builder.cast_args == ("pid_0", "i64", False)


def test_program_id_invalid_width():
    builder = StubBuilder(width=32)
    with pytest.raises(ValueError):
        tl.program_id(0, bitwidth=48, _builder=builder)


def test_program_id_default_remains_int32():
    builder = StubBuilder(width=32)
    pid = tl.program_id(1, _builder=builder)
    assert pid.dtype == tl.int32
    assert builder.cast_args is None
    assert builder.axis == 1
