import functools
import triton
import triton.language as tl
from triton._filecheck import filecheck_test, run_filecheck_test, run_parser
from triton.compiler.errors import CompilationError
import pytest
from typing import NamedTuple

# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def doesnt_compile(kernel):

    @functools.wraps(kernel)
    def test_fn():
        with pytest.raises(triton.CompilationError):
            run_parser(kernel)

    return test_fn


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

    def __getitem__(self, ind: tl.constexpr, _semantic=None):
        if ind == 0:
            return self.first
        assert ind == 1
        return self.second

    def __setitem__(self, ind: tl.constexpr, value, _semantic=None):
        if ind == 0:
            self.first = value
        assert ind == 1
        self.second = value


@doesnt_compile
@triton.jit
def test_assign_attribute():
    scalar = 11
    pair = Pair(tl.arange(0, 4), scalar)
    pair.second = 42


@doesnt_compile
@triton.jit
def test_augassign_attribute():
    scalar = 11
    pair = Pair(tl.arange(0, 4), scalar)
    pair.second += 42


@filecheck_test
@triton.jit
def test_retrieve_item():
    # CHECK-LABEL: test_retrieve_item
    # CHECK: %c11_i32 = arith.constant 11 : i32
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    scalar = 11
    pair = Pair(tl.arange(0, 4), scalar)
    # CHECK-NEXT: call @{{.*}}anchor{{.*}}(%c11_i32)
    anchor(pair[1])


@doesnt_compile
@triton.jit
def test_assign_item():
    scalar = 11
    pair = Pair(tl.arange(0, 4), scalar)
    pair[1] = 42


@doesnt_compile
@triton.jit
def test_augassign_item():
    scalar = 11
    pair = Pair(tl.arange(0, 4), scalar)
    pair[1] += 42


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
class TypeWithJitGetItem:
    value: tl.tensor

    def __init__(self, value):
        self.value = value

    @triton.jit
    def __getitem__(self, ind):
        return self.value


@filecheck_test
@triton.jit
def test_jit_getitem():
    # CHECK-LABEL: test_jit_getitem
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    v = TypeWithJitGetItem(tl.arange(0, 4))
    # CHECK: [[V:%.*]] = tt.call [[METHOD:@.*__getitem__.*]]([[RANGE]])
    a = v[0]
    # CHECK: call @{{.*}}anchor{{.*}}([[V]])
    anchor(a)
    # CHECK: tt.func private [[METHOD]]([[ARG0:%.*]]:
    # CHECK: tt.return [[ARG0]]


@tl.core._aggregate
class TypeWithBuiltinInitializer:
    value: tl.tensor

    def __init__(self, _semantic=None):
        self.value = tl.arange(0, 4, _semantic=_semantic)


@filecheck_test
@triton.jit
def test_aggregate_initializers():
    # CHECK-LABEL: test_aggregate_initializers
    value = TypeWithBuiltinInitializer()
    # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
    # CHECK: call @{{.*}}anchor{{.*}}([[RANGE]])
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
    # CHECK: tt.call @"test_frontend.add_rhs_constexpr__test_frontend.AggregateWithConstexpr<i32S4S, c42>
    agg = AggregateWithConstexpr.create(tl.arange(0, 4))
    add_rhs_constexpr(agg)

    # CHECK: tt.func private @"test_frontend.add_rhs_constexpr__test_frontend.AggregateWithConstexpr<i32S4S, c42>
    # CHECK: %cst = arith.constant dense<42> : tensor<4xi32>
    # CHECK: arith.addi %arg0, %cst : tensor<4xi32>


@tl.core._aggregate
class AggregateWithTuple:
    a: tl.tuple

    @triton.constexpr_function
    def __init__(self, a):
        self.a = tl.tuple((a, ))

    @staticmethod
    @triton.jit
    def create(a):
        return AggregateWithTuple(a)


@triton.jit
def pass_tuple_aggregate(agg):
    pass


@filecheck_test
@triton.jit
def test_aggregate_with_tuple():
    # CHECK-LABEL: test_aggregate_with_tuple
    # CHECK: tt.call @"test_frontend.pass_tuple_aggregate__test_frontend.AggregateWithTuple<Ti32S4ST>"
    agg = AggregateWithTuple.create(tl.arange(0, 4))
    pass_tuple_aggregate(agg)
    # CHECK: tt.func private @"test_frontend.pass_tuple_aggregate__test_frontend.AggregateWithTuple<Ti32S4ST>"


@triton.constexpr_function
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


@doesnt_compile
@triton.jit
def test_assign_tuple_attrs_kernel():
    p = Pair(tl.arange(0, 4), tl.arange(4, 8))
    p.first, p.second = swap(p)


@doesnt_compile
@triton.jit
def test_reassign_aggregate_with_constexpr():
    agg = AggregateWithConstexpr.create(tl.arange(0, 4))
    agg = agg.modify(tl.arange(4, 8))


@triton.constexpr_function
def make_shape(m, n):
    return (m, n)


@triton.constexpr_function
def add_shape_dims(m, n):
    return m + n


@filecheck_test
@triton.jit
def test_constexpr_getitem():
    # CHECK-LABEL: test_constexpr_getitem
    # CHECK: make_range {end = 12 : i32, start = 4 : i32}
    shape: tl.constexpr = make_shape(4, 8)
    sum: tl.constexpr = add_shape_dims(shape[0], shape[1])
    tl.arange(4, sum)


@triton.constexpr_function
def Box(T):

    @tl.core._aggregate
    class BoxImpl:
        value: T

        @triton.jit
        def create(value):
            return BoxImpl(value)

        def __init__(self, value):
            self.value = value

    return BoxImpl


def test_late_bound_class_reference():
    TensorBox = Box(tl.tensor)

    @triton.jit
    def kernel():
        # CHECK: [[RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32}
        # CHECK: call @{{.*}}anchor{{.*}}([[RANGE]])
        value = TensorBox(tl.arange(0, 4))
        anchor(value)

    run_filecheck_test(kernel)


@triton.jit
def recursive_reduce(x):
    if x.shape[0] == 1:
        return x
    else:
        x0, x1 = x.reshape((x.shape[0] // 2, 2)).split()
        return recursive_reduce(x0) + recursive_reduce(x1)


@filecheck_test
@triton.jit
def test_specialized_recursion():
    # CHECK-LABEL: test_specialized_recursion
    # CHECK: call {{.*}}recursive_reduce__i32S16S
    x = tl.arange(0, 16)
    recursive_reduce(x)

    # CHECK: func {{.*}}recursive_reduce__i32S16S
    # CHECK-COUNT-2: call {{.*}}recursive_reduce__i32S8S

    # CHECK: func {{.*}}recursive_reduce__i32S8S
    # CHECK-COUNT-2: call {{.*}}recursive_reduce__i32S4S

    # CHECK: func {{.*}}recursive_reduce__i32S4S
    # CHECK-COUNT-2: call {{.*}}recursive_reduce__i32S2S


@triton.jit
def trivial_return():
    return


@filecheck_test
@triton.jit
def test_call_in_while():
    # CHECK-LABEL: test_call_in_while
    i = 0
    while i < 10:
        if i == 5:
            trivial_return()
        else:
            trivial_return()


def test_return_in_while():

    @triton.jit
    def kernel():
        i = 0
        while i < 10:
            if i == 5:
                return
            i += 1

    with pytest.raises(CompilationError) as e:
        run_parser(kernel)

    assert "Cannot have `return` statements inside `while` or `for` statements in triton" in str(e.value)


class TensorPtr(NamedTuple):
    test: tl.constexpr


class TestTuple(NamedTuple):
    __test__ = False
    test: TensorPtr


@triton.jit
def foo(test: TestTuple):
    x: tl.constexpr = tl.constexpr(1)
    for i in tl.range(x):
        # Tests that it compiles and is usable.
        tl.static_assert(test.test.test == 1)


def test_tuple_constexpr():
    test = TestTuple(test=TensorPtr(tl.constexpr(1)))
    run_parser(foo, args=(test, ))


@tl.core._aggregate
class AggregateWithConstexprFunction:
    val: tl.constexpr
    val_squared: tl.constexpr

    def __init__(self, val):
        self.val = tl.constexpr(val)
        self.val_squared = tl.constexpr(self.square_val())

    @triton.constexpr_function
    def square_val(self):
        return self.val * self.val


@filecheck_test
@triton.jit
def test_aggregate_constexpr_function():
    agg = AggregateWithConstexprFunction(4)
    # CHECK: call @{{.*}}anchor{{.*}}c4
    anchor(agg.val)

    # CHECK: call @{{.*}}anchor{{.*}}c16
    anchor(agg.val_squared)

    # CHECK: call @{{.*}}anchor{{.*}}c16
    anchor(agg.square_val())


@tl.core.builtin
def make_list(*args, _semantic=None):
    return list(args)


@triton.constexpr_function
def function_taking_list(arg):
    return arg[1]


@filecheck_test
@triton.jit
def test_constexpr_function_taking_list():
    a: tl.constexpr = function_taking_list(make_list(4, 8, 16))
    # CHECK: call @{{.*}}anchor{{.*}}c8
    anchor(a)


@filecheck_test
@triton.jit
def test_constexpr_min_max():
    a: tl.constexpr = min(1, 2)
    # CHECK: call @{{.*}}anchor{{.*}}c1
    anchor(a)

    b: tl.constexpr = min(1, 2, -3)
    # CHECK: call @{{.*}}anchor{{.*}}c-3
    anchor(b)

    c: tl.constexpr = max(3, 4)
    # CHECK: call @{{.*}}anchor{{.*}}c4
    anchor(c)

    d: tl.constexpr = max(3, 4, 5)
    # CHECK: call @{{.*}}anchor{{.*}}c5
    anchor(d)


def test_constexpr_min_error():

    @triton.jit
    def min_kernel(a: tl.constexpr, b: tl.constexpr):
        min(a, b)

    with pytest.raises(CompilationError):
        run_parser(min_kernel, args=(1.0, float("nan")))

    with pytest.raises(CompilationError):
        run_parser(min_kernel, args=(1.0, -0.0))


def test_constexpr_max_error():

    @triton.jit
    def max_kernel(a: tl.constexpr, b: tl.constexpr):
        max(a, b)

    with pytest.raises(CompilationError):
        run_parser(max_kernel, args=(1.0, float("nan")))

    with pytest.raises(CompilationError):
        run_parser(max_kernel, args=(1.0, -0.0))


@filecheck_test
@triton.jit
def test_for_loop_iv_modification():
    # CHECK: scf.for %[[I:.*]] = {{.*}} to {{.*}} step {{.*}} : i32 {
    for i in range(4):
        # CHECK: anchor{{.*}}%[[I]]
        anchor(i)
        # CHECK: %[[I2:.*]] = arith.addi %[[I]], %{{.*}} : i32
        i += 1
        # CHECK: anchor{{.*}}%[[I2]]
        anchor(i)


@pytest.mark.interpreter
def test_constexpr_return():

    @triton.jit
    def get_constexpr_value():
        return tl.constexpr(42)

    @triton.jit
    def test():
        x: tl.constexpr = get_constexpr_value()
        tl.static_assert(x == 42)

    run_parser(test)


@pytest.mark.interpreter
def test_return_promotion():

    @triton.jit
    def signbit(x):
        if x < 0:
            return 1
        else:
            return 0

    @triton.jit
    def tuple_return(x):
        if x < 0:
            return 1, x
        else:
            return 0, x

    @triton.jit
    def kernel():
        # constexpr if -> constexpr returned
        a: tl.constexpr = signbit(-1)
        tl.static_assert(a == 1)

        # dynamic if -> promote to tensor
        tmp = -1
        tl.static_assert(signbit(tmp).type == tl.int32)

        # constexpr if -> single return
        b: tl.constexpr = tuple_return(-1)
        tl.static_assert(b[0] == 1 and b[1] == -1)

        c = tuple_return(tmp)
        tl.static_assert(c.type == tl.tuple_type([tl.int32, tl.int32]))

    run_parser(kernel)
