import triton
import triton.language as tl
from triton._filecheck import filecheck_test

# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


@triton.jit
def anchor(v):
    pass


@tl.core._aggregate
class Pair:
    first: tl.tensor
    second: tl.tensor

    def __init__(self, first, second):
        self.first = first
        self.second = second

    @triton.jit
    def get_first(self):
        return self.first

    def get_second(self, _builder=None):
        return self.second

    @triton.jit
    def unpack(self):
        return self.get_first(), self.get_second()


@filecheck_test
@triton.jit
def test_assign_attribute():
    # CHECK-LABEL: assign_attribute
    # CHECK: %c11_i32 = arith.constant 11 : i32
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    scalar = 11
    pair = Pair(tl.arange(0, 4), scalar)
    # CHECK: %c42_i32 = arith.constant 42 : i32
    # CHECK-NEXT: call @"anchor{{.*}}"([[RANGE]], %c42_i32)
    pair.second = 42
    anchor(pair)


@filecheck_test
@triton.jit
def test_jit_method():
    # CHECK-LABEL: test_jit_method
    # CHECK: %c11_i32 = arith.constant 11 : i32
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    scalar = 11
    # CHECK: [[V:%.*]]:2 = tt.call @"unpack{{.*}}"([[RANGE]], %c11_i32)
    pair = Pair(tl.arange(0, 4), scalar)
    a, b = pair.unpack()
    # CHECK: call @anchor{{.*}}([[V]]#0)
    anchor(a)
    # CHECK: call @anchor{{.*}}([[V]]#1)
    anchor(b)


@tl.core._aggregate
class TypeWithBuiltinInitializer:
    value: tl.tensor

    def __init__(self, _builder=None):
        self.value = tl.arange(0, 4, _builder=_builder)

    def modify(self, value, _builder=None):
        self.value = value


@filecheck_test
@triton.jit
def test_aggregate_initializers():
    # CHECK-LABEL: test_aggregate_initializers
    value = TypeWithBuiltinInitializer()
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    # CHECK: call @"anchor{{.*}}"([[RANGE]])
    anchor(value)
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 8 : i32, start = 4 : i32}
    # CHECK: call @"anchor{{.*}}"([[RANGE]])
    value.modify(tl.arange(4, 8))
    anchor(value)


@triton.jit
def forward(arg):
    return arg


@triton.jit
def list_of_functions_constexpr(arg, fns: tl.constexpr):
    for i in tl.static_range(len(fns)):
        fns[i](arg)


@filecheck_test
@triton.jit
def test_list_of_functions():
    # CHECK-LABEL: test_list_of_functions
    # CHECK: call @"list_of_functions_constexpr{{.*}}cJITFunction(test_frontend:anchor){{.*}}cJITFunction(test_frontend:forward)"

    # CHECK-LABEL: tt.func private @"list_of_functions_constexpr
    # CHECK-NEXT: call @anchor
    # CHECK-NEXT: call @forward
    list_of_functions_constexpr(tl.arange(0, 4), [anchor, forward])


@triton.jit
def accumulate(a, b):
    return a + b


# Check that we can call a function returning a value from a loop.
@filecheck_test
@triton.jit
def test_call_in_loop():
    # CHECK-LABEL: test_call_in_loop
    acc = 0
    # CHECK: scf.for
    # CHECK:   call @accumulate
    for i in range(10):
        acc = accumulate(acc, i)
