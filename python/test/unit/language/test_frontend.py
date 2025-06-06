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

    def get_second(self, _semantic=None):
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
    # CHECK-NEXT: call @{{.*}}anchor{{.*}}([[RANGE]], %c42_i32)
    pair.second = 42
    anchor(pair)


@filecheck_test
@triton.jit
def test_augassign_attribute():
    # CHECK-LABEL: test_augassign_attribute
    # CHECK: %c11_i32 = arith.constant 11 : i32
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    scalar = 11
    pair = Pair(tl.arange(0, 4), scalar)
    # CHECK: %c42_i32 = arith.constant 42 : i32
    # CHECK: [[VALUE:%.*]] = arith.addi %c11_i32, %c42_i32
    pair.second += 42
    # CHECK-NEXT: call @{{.*}}anchor{{.*}}([[RANGE]], [[VALUE]])
    anchor(pair)


@filecheck_test
@triton.jit
def test_jit_method():
    # CHECK-LABEL: test_jit_method
    # CHECK: %c11_i32 = arith.constant 11 : i32
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    scalar = 11
    # CHECK: [[V:%.*]]:2 = tt.call @{{.*}}unpack{{.*}}([[RANGE]], %c11_i32)
    pair = Pair(tl.arange(0, 4), scalar)
    a, b = pair.unpack()
    # CHECK: call @{{.*}}anchor{{.*}}([[V]]#0)
    anchor(a)
    # CHECK: call @{{.*}}anchor{{.*}}([[V]]#1)
    anchor(b)


@tl.core._aggregate
class TypeWithBuiltinInitializer:
    value: tl.tensor

    def __init__(self, _semantic=None):
        self.value = tl.arange(0, 4, _semantic=_semantic)

    def modify(self, value, _semantic=None):
        self.value = value


@filecheck_test
@triton.jit
def test_aggregate_initializers():
    # CHECK-LABEL: test_aggregate_initializers
    value = TypeWithBuiltinInitializer()
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    # CHECK: call @{{.*}}anchor{{.*}}([[RANGE]])
    anchor(value)
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 8 : i32, start = 4 : i32}
    # CHECK: call @{{.*}}anchor{{.*}}([[RANGE]])
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
    # CHECK: call @{{.*}}list_of_functions_constexpr{{.*}}cJITFunction(test_frontend:anchor){{.*}}cJITFunction(test_frontend:forward)

    # CHECK: tt.func private @{{.*}}list_of_functions_constexpr
    # CHECK-NEXT: call @{{.*}}anchor
    # CHECK-NEXT: call @{{.*}}forward
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
    # CHECK:   call @{{.*}}accumulate
    for i in range(10):
        acc = accumulate(acc, i)


@tl.core._aggregate
class FunctionParent:

    @triton.jit
    def function_with_name():
        pass


@triton.jit
def function_with_name():
    pass


@filecheck_test
@triton.jit
def test_function_name_mangling():
    # CHECK-LABEL: test_function_name_mangling
    # CHECK: call @test_frontend.function_with_name
    # CHECK: call @test_frontend.FunctionParent.function_with_name
    function_with_name()
    FunctionParent.function_with_name()


@tl.core._aggregate
class AggregateWithConstexpr:
    a: tl.tensor
    b: tl.constexpr

    def __init__(self, a, b):
        self.a = a
        self.b = b

    @staticmethod
    def create(a):
        return AggregateWithConstexpr(a, tl.constexpr(42))

    @triton.jit
    def modify(self, a):
        self.a = a
        return self


@triton.jit
def add_rhs_constexpr(agg):
    _ = agg.a + agg.b


@filecheck_test
@triton.jit
def test_aggregate_with_constexpr():
    # CHECK-LABEL: test_aggregate_with_constexpr
    # CHECK: tt.call @"test_frontend.add_rhs_constexpr__test_frontend.AggregateWithConstexpr<i32S4S, constexpr[42]>
    agg = AggregateWithConstexpr.create(tl.arange(0, 4))
    add_rhs_constexpr(agg)

    # CHECK: tt.func private @"test_frontend.add_rhs_constexpr__test_frontend.AggregateWithConstexpr<i32S4S, constexpr[42]>
    # CHECK: %cst = arith.constant dense<42> : tensor<4xi32>
    # CHECK: arith.addi %arg0, %cst : tensor<4xi32>


@tl.constexpr_function
def constexpr_function(x):
    return x + 1


@filecheck_test
@triton.jit
def test_constexpr_function_from_jit():
    # CHECK-LABEL: test_constexpr_function
    x: tl.constexpr = constexpr_function(7)
    # CHECK: make_range {end = 8 : i32, start = 0 : i32}
    tl.arange(0, x)


def test_constexpr_function_from_python():
    assert constexpr_function(7) == 8


@triton.jit
def swap(pair):
    return pair.second, pair.first


@filecheck_test
@triton.jit
def test_assign_tuple_attrs():
    # CHECK-LABEL: test_assign_tuple_attrs
    p = Pair(tl.arange(0, 4), tl.arange(4, 8))
    # CHECK: [[P:%.*]]:2 = tt.call @{{.*}}swap
    p.first, p.second = swap(p)
    # CHECK: call @{{.*}}anchor{{.*}}([[P]]#0)
    # CHECK: call @{{.*}}anchor{{.*}}([[P]]#1)
    anchor(p.first)
    anchor(p.second)


@filecheck_test
@triton.jit
def test_reassign_aggregate_with_constexpr():
    # CHECK-LABEL: test_reassign_aggregate_with_constexpr
    agg = AggregateWithConstexpr.create(tl.arange(0, 4))
    var = 1
    # CHECK: [[AGG:%.*]] = scf.if {{.*}} -> (tensor<4xi32>)
    # CHECK:   [[VALUE:%.*]] = tt.call {{.*}}modify
    # CHECK:   yield [[VALUE]]
    # CHECK: else
    # CHECK:   [[VALUE:%.*]] = tt.call {{.*}}modify
    # CHECK:   yield [[VALUE]]
    if var == 0:
        agg = agg.modify(tl.arange(4, 8))
    else:
        agg = agg.modify(tl.arange(8, 12))
    # CHECK: call @{{.*}}anchor{{.*}}([[AGG]])
    anchor(agg)
