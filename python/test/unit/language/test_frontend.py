import collections
import functools
import triton
import triton.language as tl
from triton.experimental import gluon
from triton._filecheck import filecheck_test, run_filecheck_test, run_parser
from triton.compiler.code_generator import CodeGenerator
from triton.runtime.jit import MockTensor
from triton.compiler.errors import CompilationError
import pytest
from typing import NamedTuple

# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


@pytest.mark.parametrize("jit", [triton.jit, gluon.jit])
def test_jit_variadic_keyword_arguments(jit):

    def kernel(**kwargs):
        pass

    with pytest.raises(TypeError, match=r"JIT functions do not support \*\*kwargs"):
        jit(kernel)


def doesnt_compile(kernel):

    @functools.wraps(kernel)
    def test_fn():
        with pytest.raises(triton.CompilationError):
            run_parser(kernel)

    return test_fn


@triton.jit
def anchor(v):
    pass


@pytest.mark.parametrize("dtype",
                         [tl.float16, tl.bfloat16, tl.float32, tl.float64, tl.float8e4nv, tl.float8e5, tl.float8e4b15],
                         ids=str)
def test_scalar_constant_preserves_signed_zero(dtype):

    @triton.jit
    def kernel(dtype: tl.constexpr):
        # CHECK: arith.constant -0.000000e+00
        anchor(tl.full((), -0.0, dtype))
        # CHECK: arith.constant 0.000000e+00
        anchor(tl.full((), 0.0, dtype))

    run_filecheck_test(kernel, args=(dtype, ))


@pytest.mark.parametrize("dtype",
                         [tl.int1, tl.int8, tl.int16, tl.int32, tl.int64, tl.uint8, tl.uint16, tl.uint32, tl.uint64],
                         ids=str)
@pytest.mark.parametrize("value", [0.0, -0.0])
def test_scalar_constant_float_zero_to_integer(dtype, value):

    @triton.jit
    def kernel(dtype: tl.constexpr, value: tl.constexpr):
        # CHECK: arith.constant {{0|false}}
        anchor(tl.full((), value, dtype))

    run_filecheck_test(kernel, args=(dtype, value))


@pytest.mark.parametrize("op", [tl.minimum, tl.maximum])
@pytest.mark.parametrize("dtype, value, expected_dtype", [
    (tl.float16, 0.0, tl.float16),
    (tl.float16, 0, tl.float16),
    (tl.float32, 0.0, tl.float32),
    (tl.float64, 0.0, tl.float64),
    (tl.bfloat16, 0.0, tl.float32),
    (tl.int8, False, tl.int8),
    (tl.int16, 0, tl.int16),
    (tl.uint16, 0, tl.uint16),
    (tl.int16, 0.0, tl.float32),
])
def test_minimum_maximum_scalar_promotion(op, dtype, value, expected_dtype):

    @triton.jit
    def kernel(op: tl.constexpr, dtype: tl.constexpr, value: tl.constexpr, expected_dtype: tl.constexpr):
        x = tl.full((8, ), 1, dtype)
        lhs = op(x, value)
        rhs = op(value, x)
        tl.static_assert(lhs.dtype == expected_dtype)
        tl.static_assert(rhs.dtype == expected_dtype)

    run_parser(kernel, args=(op, dtype, value, expected_dtype))


@pytest.mark.parametrize("op", [tl.minimum, tl.maximum])
@pytest.mark.parametrize("other_dtype", [tl.float32, tl.bfloat16])
def test_minimum_maximum_tensor_promotion(op, other_dtype):

    @triton.jit
    def kernel(op: tl.constexpr, other_dtype: tl.constexpr):
        x = tl.full((8, ), 1, tl.float16)
        y = tl.full((), 0.0, other_dtype)
        lhs = op(x, y)
        rhs = op(y, x)
        tl.static_assert(lhs.dtype == tl.float32)
        tl.static_assert(rhs.dtype == tl.float32)

    run_parser(kernel, args=(op, other_dtype))


@triton.aggregate
class Pair:
    first: tl.tensor
    second: tl.tensor

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


@triton.aggregate
class TypeWithJitGetItem:
    value: tl.tensor

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


@triton.aggregate
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


def test_aggregate_auto_init_assigns_members():

    @triton.aggregate
    class State:
        x: tl.constexpr
        y: tl.constexpr

    state = State(3, y=7)
    assert isinstance(state.x, tl.constexpr)
    assert isinstance(state.y, tl.constexpr)
    assert state.x.value == 3
    assert state.y.value == 7


def test_aggregate_auto_init_with_tuples():

    class Shape(NamedTuple):
        x: tl.constexpr
        y: tl.constexpr

    @triton.aggregate
    class State:
        shape: tl.tuple
        strides: tl.tuple

    state = State(Shape(3, tl.constexpr(7)), (7, tl.constexpr(1)))
    assert isinstance(state.shape, tl.tuple)
    assert isinstance(state.shape.x, tl.constexpr)
    assert isinstance(state.shape.y, tl.constexpr)
    assert state.shape[0].value == 3
    assert state.shape[1].value == 7

    assert isinstance(state.strides, tl.tuple)
    assert isinstance(state.strides[0], tl.constexpr)
    assert isinstance(state.strides[1], tl.constexpr)
    assert state.strides[0].value == 7
    assert state.strides[1].value == 1


def test_aggregate_auto_init_respects_user_defined_init():

    @triton.aggregate
    class State:
        x: tl.constexpr

        def __init__(self, x):
            self.x = tl.constexpr(x + 1)

    state = State(10)
    assert state.x.value == 11


@triton.jit
def forward(arg):
    return arg


@triton.jit
def list_of_functions_constexpr(arg, fns: tl.constexpr):
    for i in tl.static_range(len(fns)):
        fns[i](arg)


@triton.jit
def consume_varargs(*dims):
    return dims[0]


@filecheck_test
@triton.jit
def test_list_of_functions():
    # CHECK-LABEL: test_list_of_functions
    # CHECK: call @{{.*}}list_of_functions_constexpr{{.*}}cJITFunction(test_frontend:anchor){{.*}}cJITFunction(test_frontend:forward)

    # CHECK: tt.func private @{{.*}}list_of_functions_constexpr
    # CHECK-NEXT: call @{{.*}}anchor
    # CHECK-NEXT: call @{{.*}}forward
    list_of_functions_constexpr(tl.arange(0, 4), [anchor, forward])


@filecheck_test
@triton.jit
def test_starred_varargs():
    # CHECK-LABEL: test_starred_varargs
    # CHECK: call @{{.*}}consume_varargs
    dims: tl.constexpr = (1, 0)
    consume_varargs(*dims)


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


@triton.aggregate
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


@triton.aggregate
class AggregateWithConstexpr:
    a: tl.tensor
    b: tl.constexpr

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


@triton.aggregate
class AggregateWithTuple:
    a: tl.tuple

    @staticmethod
    @triton.jit
    def create(a):
        return AggregateWithTuple((a, ))


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

    @triton.constexpr_function
    def with_kwargs(**kwargs):
        return kwargs["value"] + 1

    assert with_kwargs(value=7) == 8


@filecheck_test
@triton.jit
def test_named_expr():
    # CHECK-LABEL: test_named_expr
    x = (y := 0)
    # CHECK: %c0_i32 = arith.constant 0 : i32
    # CHECK-NEXT: call @{{.*}}anchor{{.*}}(%c0_i32)
    anchor(x)
    # CHECK-NEXT: call @{{.*}}anchor{{.*}}(%c0_i32)
    anchor(y)


def test_tuple_assignment_respects_prior_constexpr_annotation():

    @triton.jit
    def kernel():
        y: tl.constexpr
        x, y = 0, 0
        tl.static_assert(x.dtype == tl.int32)
        tl.static_assert(y.type == tl.constexpr_type(0))

    run_parser(kernel)


def test_tuple_assignment_constexpr_tuple_matches_annassign():

    @triton.jit
    def kernel():
        a: tl.constexpr
        a, b = (0, 1), 2

        assigned_a: tl.constexpr = (0, 1)
        assigned_b = 2

        tl.static_assert(a == assigned_a)
        tl.static_assert(a.type == ((0, 1)).type)
        tl.static_assert(assigned_a.type == ((0, 1)).type)
        tl.static_assert(b.dtype == tl.int32)
        tl.static_assert(assigned_b.dtype == tl.int32)

    run_parser(kernel)


def test_tuple_assignment_constexpr_tuple_normalizes_recursively():

    @triton.jit
    def kernel():
        a: tl.constexpr
        a, b = ((0, 1), (2, 3)), 4

        assigned_a: tl.constexpr = ((0, 1), (2, 3))

        tl.static_assert(a == assigned_a)
        tl.static_assert(a.type == (((0, 1), (2, 3))).type)
        tl.static_assert(assigned_a.type == (((0, 1), (2, 3))).type)
        tl.static_assert(b.dtype == tl.int32)

    run_parser(kernel)


def test_tuple_assignment_rejects_too_many_values():

    @triton.jit
    def kernel():
        a, b = (1, 2, 3)  # noqa: F841

    with pytest.raises(CompilationError, match="too many values to unpack"):
        run_parser(kernel)


def test_tuple_assignment_rejects_too_few_values():

    @triton.jit
    def kernel():
        a, b, c = (1, 2)  # noqa: F841

    with pytest.raises(CompilationError, match=r"not enough values to unpack \(expected 3, got 2\)"):
        run_parser(kernel)


def test_tuple_assignment_rejects_nested_mismatch():

    @triton.jit
    def kernel():
        (a, b), c = ((1, 2, 3), 4)  # noqa: F841

    with pytest.raises(CompilationError, match="too many values to unpack"):
        run_parser(kernel)


def test_tuple_assignment_rejects_starred_target():

    @triton.jit
    def kernel():
        a, *rest = (1, 2, 3)  # noqa: F841

    with pytest.raises(CompilationError, match="starred assignment targets are not supported"):
        run_parser(kernel)


def test_list_comprehension_if_filter():

    @triton.jit
    def kernel():
        # an `if` filter drops the elements whose condition is false
        vals: tl.constexpr = [x for x in (10, 20, 30, 40) if x >= 30]
        tl.static_assert(len(vals) == 2)
        tl.static_assert(vals[0] == 30)
        tl.static_assert(vals[1] == 40)

        # multiple `if` clauses compose as "and"
        multi: tl.constexpr = [x for x in (0, 1, 2, 3, 4, 5) if x > 1 if x % 2 == 0]
        tl.static_assert(len(multi) == 2)
        tl.static_assert(multi[0] == 2)
        tl.static_assert(multi[1] == 4)

        # an unfiltered comprehension is unchanged
        allv: tl.constexpr = [x for x in (10, 20, 30, 40)]
        tl.static_assert(len(allv) == 4)

    run_parser(kernel)


def test_named_expr_respects_prior_constexpr_annotation():

    @triton.jit
    def kernel():
        x: tl.constexpr
        if (x := constexpr_function(10)) != 10:
            tl.static_assert(isinstance(x.type, tl.constexpr_type))
        else:
            tl.static_assert(False)

    run_parser(kernel)


@filecheck_test
@triton.jit
def test_named_expr_without_prior_annotation_decays():
    # CHECK-LABEL: test_named_expr_without_prior_annotation_decays
    # CHECK: [[COND:%.*]] = arith.cmpi ne, %c11_i32, %c10_i32 : i32
    # CHECK: scf.if [[COND]] {
    # CHECK:   tt.call @{{.*}}anchor{{.*}}(%c11_i32) : (i32) -> ()
    # CHECK: } else {
    # CHECK:   [[ADD:%.*]] = arith.addi %c11_i32, %c1_i32_0 : i32
    # CHECK:   tt.call @{{.*}}anchor{{.*}}([[ADD]]) : (i32) -> ()
    # CHECK: }
    if (x := constexpr_function(10)) != 10:
        anchor(x)
    else:
        anchor(x + 1)


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

    @triton.aggregate
    class BoxImpl:
        value: T

        @triton.jit
        def create(value):
            return BoxImpl(value)

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


@filecheck_test
@triton.jit
def test_while_integer_condition():
    # CHECK-LABEL: test_while_integer_condition
    i = tl.program_id(0)
    # CHECK: scf.while
    # CHECK: [[COND:%.*]] = arith.cmpi ne, %{{.*}}, %{{.*}} : i32
    # CHECK: scf.condition([[COND]])
    while i:
        i -= 1
    anchor(i)


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


@triton.jit
def tuple_arg_identity(xs):
    return xs


def test_jit_call_tuple_of_tensors_plus_tuple_of_int():

    @triton.jit
    def kernel():
        x0 = tl.program_id(0)
        x1 = tl.program_id(1)
        tuple_arg_identity((x0, x1)[:-1] + (1, ))

    run_parser(kernel)


@triton.aggregate
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


@filecheck_test
@triton.jit
def test_atomic_scalar_masks():
    # CHECK-LABEL: test_atomic_scalar_masks
    BLOCK: tl.constexpr = 128
    ptr = tl.full((BLOCK, ), 0, tl.int64).to(tl.pointer_type(tl.int32), bitcast=True)
    offs = tl.arange(0, BLOCK)
    ptrs = ptr + offs
    val = tl.full((BLOCK, ), 1, tl.int32)
    mask = offs >= 0
    scalar_mask = True
    constexpr_value: tl.constexpr = 1
    constexpr_mask: tl.constexpr = True

    # CHECK: {{.*}} = tt.atomic_rmw add, acq_rel, gpu
    tl.atomic_add(ptrs, val, mask=mask)
    # CHECK: {{.*}} = tt.atomic_rmw add, acq_rel, gpu
    tl.atomic_add(ptrs, 1, mask=True)
    # CHECK: {{.*}} = tt.atomic_rmw add, acq_rel, gpu
    tl.atomic_add(ptrs, constexpr_value, mask=constexpr_mask)
    # CHECK: {{.*}} = tt.atomic_rmw add, acq_rel, gpu
    tl.atomic_add(ptrs, val, mask=scalar_mask)

    # CHECK: {{.*}} = tt.atomic_rmw exch, acq_rel, gpu
    tl.atomic_xchg(ptrs, 1, mask=True)
    # CHECK: {{.*}} = tt.atomic_rmw max, acq_rel, gpu
    tl.atomic_max(ptrs, 1, mask=True)
    # CHECK: {{.*}} = tt.atomic_rmw min, acq_rel, gpu
    tl.atomic_min(ptrs, 1, mask=True)
    # CHECK: {{.*}} = tt.atomic_rmw and, acq_rel, gpu
    tl.atomic_and(ptrs, 1, mask=True)
    # CHECK: {{.*}} = tt.atomic_rmw or, acq_rel, gpu
    tl.atomic_or(ptrs, 1, mask=True)
    # CHECK: {{.*}} = tt.atomic_rmw xor, acq_rel, gpu
    tl.atomic_xor(ptrs, 1, mask=True)


@filecheck_test
@triton.jit
def test_atomic_poll():
    # CHECK-LABEL: test_atomic_poll
    ptr = tl.to_tensor(0).to(tl.int64).to(tl.pointer_type(tl.int32), bitcast=True)
    # CHECK: %{{.*}} = tt.atomic_poll relaxed, sys, %{{.*}}, %{{.*}} : !tt.ptr<i32>, i32 -> i1
    tl.atomic_poll(ptr, 1, sem="relaxed", scope="sys")


@filecheck_test
@triton.jit
def test_atomic_poll_timeout():
    # CHECK-LABEL: test_atomic_poll_timeout
    ptr = tl.to_tensor(0).to(tl.int64).to(tl.pointer_type(tl.int32), bitcast=True)
    # CHECK: %{{.*}} = tt.atomic_poll acquire, gpu, %{{.*}}, %{{.*}} timeout %{{.*}} : !tt.ptr<i32>, i32 -> i1
    tl.atomic_poll(ptr, 1, timeout_ns=1000)


@doesnt_compile
@triton.jit
def test_atomic_poll_rejects_tensor_pointer():
    ptrs = tl.full((1, ), 0, tl.int64).to(tl.pointer_type(tl.int32), bitcast=True)
    tl.atomic_poll(ptrs, 1)


@doesnt_compile
@triton.jit
def test_atomic_poll_rejects_release_semantics():
    ptr = tl.to_tensor(0).to(tl.int64).to(tl.pointer_type(tl.int32), bitcast=True)
    tl.atomic_poll(ptr, 1, sem="release")


@doesnt_compile
@triton.jit
def test_atomic_poll_rejects_negative_timeout():
    ptr = tl.to_tensor(0).to(tl.int64).to(tl.pointer_type(tl.int32), bitcast=True)
    tl.atomic_poll(ptr, 1, timeout_ns=-1)


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


def test_fp8_div_mod_promotion():
    # `/` and `%` do not exist natively for floats narrower than fp32, so the
    # result of a division or modulo with a floating operand is promoted to
    # fp32 -- one rule covering fp8, fp16 and bfloat16, tensor and scalar
    # operands alike. Other ops keep the existing promotions (same fp8 stays
    # that fp8, mixed fp8 goes to float16).

    @triton.jit
    def kernel():
        x = tl.full((8, ), 0, tl.float16).to(tl.float8e5)
        y = tl.full((8, ), 0, tl.float16).to(tl.float8e5)
        z = tl.full((8, ), 0, tl.float16).to(tl.float8e4nv)
        h = tl.full((8, ), 0, tl.float16)
        b = tl.full((8, ), 0, tl.bfloat16)
        d = tl.full((8, ), 0, tl.float64)
        i = tl.full((8, ), 0, tl.int32)
        tl.static_assert((x / y).dtype == tl.float32)
        tl.static_assert((x / z).dtype == tl.float32)
        tl.static_assert((x % y).dtype == tl.float32)
        tl.static_assert((x * y).dtype == tl.float8e5)
        tl.static_assert((x * z).dtype == tl.float16)
        # fp16 and bfloat16 division/modulo upcast through the same rule
        tl.static_assert((h / h).dtype == tl.float32)
        tl.static_assert((h % h).dtype == tl.float32)
        tl.static_assert((b / b).dtype == tl.float32)
        tl.static_assert((h * h).dtype == tl.float16)
        tl.static_assert((b * b).dtype == tl.bfloat16)
        # integer division and modulo keep integer promotion
        tl.static_assert((i // i).dtype == tl.int32)
        tl.static_assert((i % i).dtype == tl.int32)
        # A scalar operand doesn't participate in promotion, so / and % against
        # a narrow float tensor must upcast to fp32 for the same reason, while other
        # ops keep the tensor's type.
        tl.static_assert((2.0 / x).dtype == tl.float32)
        tl.static_assert((x / 2.0).dtype == tl.float32)
        tl.static_assert((x % 2).dtype == tl.float32)
        tl.static_assert((h / 2.0).dtype == tl.float32)
        tl.static_assert((d / 2.0).dtype == tl.float64)
        tl.static_assert((d % 2.0).dtype == tl.float64)
        tl.static_assert((x * 2.0).dtype == tl.float8e5)
        tl.static_assert((h * 2.0).dtype == tl.float16)
        tl.static_assert((i // 2).dtype == tl.int32)

    run_parser(kernel)


# ===-----------------------------------------------------------------------===#
# Aggregate inheritance, __post_init__, and aggregate_replace tests
# ===-----------------------------------------------------------------------===#


def test_aggregate_field_inheritance():
    """Child aggregate inherits parent fields."""

    @triton.aggregate
    class Base:
        x: tl.constexpr

    @triton.aggregate
    class Child(Base):
        y: tl.constexpr

    child = Child(10, 20)
    assert isinstance(child.x, tl.constexpr)
    assert isinstance(child.y, tl.constexpr)
    assert child.x.value == 10
    assert child.y.value == 20


def test_aggregate_multilevel_inheritance():
    """Multi-level inheritance: grandparent -> parent -> child."""

    @triton.aggregate
    class GrandParent:
        a: tl.constexpr

    @triton.aggregate
    class Parent(GrandParent):
        b: tl.constexpr

    @triton.aggregate
    class Child(Parent):
        c: tl.constexpr

    child = Child(1, 2, 3)
    assert child.a.value == 1
    assert child.b.value == 2
    assert child.c.value == 3


def test_aggregate_inheritance_requires_aggregate_base():

    class Base:
        pass

    with pytest.raises(TypeError, match="Aggregates can only inherit from other aggregates"):

        @triton.aggregate
        class Child(Base):
            x: tl.constexpr


def test_aggregate_field_inheritance_with_methods():
    """Inherited methods work with inherited fields."""

    @triton.aggregate
    class Base:
        x: tl.constexpr

        @triton.constexpr_function
        def get_x(self):
            return self.x

    @triton.aggregate
    class Child(Base):
        y: tl.constexpr

    child = Child(10, 20)
    assert child.get_x().value == 10


def test_aggregate_default_values():
    """Fields with default values can be omitted from constructor."""

    @triton.aggregate
    class WithDefaults:
        x: tl.constexpr
        y: tl.constexpr = tl.constexpr(42)

    # Provide both
    obj1 = WithDefaults(10, 20)
    assert obj1.x.value == 10
    assert obj1.y.value == 20

    # Use default for y
    obj2 = WithDefaults(10)
    assert obj2.x.value == 10
    assert obj2.y.value == 42


def test_aggregate_replace():
    """aggregate_replace creates a copy with modified fields."""

    @triton.aggregate
    class State:
        x: tl.constexpr
        y: tl.constexpr

    original = State(10, 20)
    modified = tl.aggregate_replace(original, x=30)

    # Modified has the new value
    assert modified.x.value == 30
    assert modified.y.value == 20

    # Original is unchanged
    assert original.x.value == 10
    assert original.y.value == 20


def test_aggregate_replace_invalid_field():
    """aggregate_replace raises on unknown field names."""

    @triton.aggregate
    class State:
        x: tl.constexpr

    obj = State(10)
    with pytest.raises(TypeError, match="has no field 'z'"):
        tl.aggregate_replace(obj, z=99)


def test_aggregate_replace_non_aggregate():
    """aggregate_replace raises on non-aggregate instances."""
    with pytest.raises(TypeError, match="expects an aggregate instance"):
        tl.aggregate_replace(42, x=1)


def test_aggregate_inherited_defaults():
    """Child inherits default values from parent fields."""

    @triton.aggregate
    class Base:
        x: tl.constexpr = tl.constexpr(100)

    @triton.aggregate
    class Child(Base):
        y: tl.constexpr

    child = Child(y=7)
    assert child.x.value == 100
    assert child.y.value == 7


def test_aggregate_string_annotations_resolved():
    """String annotations (PEP 649 / forward refs) resolve via typing.get_type_hints.

    On Python 3.13+ class annotations may be stored as strings rather than evaluated
    types. _resolve_aggregate_fields walks the MRO directly, so it must call
    typing.get_type_hints to resolve those strings — otherwise downstream
    isinstance(value, ann) raises 'isinstance() arg 2 must be a type'.
    """

    @triton.aggregate
    class StringAnnoBase:
        x: "tl.constexpr"  # explicit string annotation — must resolve

    @triton.aggregate
    class StringAnnoChild(StringAnnoBase):
        y: "tl.constexpr"  # inherited annotation chain must resolve too

    child = StringAnnoChild(10, 20)
    assert isinstance(child.x, tl.constexpr)
    assert isinstance(child.y, tl.constexpr)
    assert child.x.value == 10
    assert child.y.value == 20


def test_aggregate_default_value_auto_wrapped():
    """A raw-int default (`y: tl.constexpr = 42`) is auto-wrapped to constexpr at init."""

    @triton.aggregate
    class State:
        x: tl.constexpr
        y: tl.constexpr = 42  # raw int default — no tl.constexpr() wrap

    obj = State(10)
    assert isinstance(obj.y, tl.constexpr)
    assert obj.y.value == 42
    # Explicit override still works.
    obj2 = State(10, 99)
    assert obj2.y.value == 99


def test_aggregate_post_construction_immutable():
    """Field assignment after construction is rejected (matches dataclasses(frozen=True))."""

    @triton.aggregate
    class State:
        x: tl.constexpr
        y: tl.constexpr

    obj = State(10, 20)
    with pytest.raises(AttributeError, match="cannot assign to field 'x' on immutable aggregate"):
        obj.x = tl.constexpr(99)
    # Original value unchanged.
    assert obj.x.value == 10

    # aggregate_replace() builds a modified copy without mutating the original.
    new = tl.aggregate_replace(obj, x=tl.constexpr(77))
    assert new.x.value == 77
    assert obj.x.value == 10


# ===-----------------------------------------------------------------------===#
# IR-level checks for inheritance + replace (moved from test_core.py per
# review feedback — frontend is sufficient since aggregates compile to flat
# field structures, no GPU runtime needed to verify language semantics).
# ===-----------------------------------------------------------------------===#


@triton.aggregate
class _AggInhBase:
    data: tl.tensor
    BLOCK: tl.constexpr


@triton.aggregate
class _AggInhChild(_AggInhBase):
    bias: tl.tensor


@filecheck_test
@triton.jit
def test_aggregate_inheritance_ir():
    # CHECK-LABEL: test_aggregate_inheritance_ir
    # CHECK: [[A:%.*]] = tt.make_range {end = 8 : i32, start = 0 : i32}
    # CHECK: [[B:%.*]] = tt.make_range {end = 16 : i32, start = 8 : i32}
    a = tl.arange(0, 8)
    b = tl.arange(8, 16)
    child = _AggInhChild(a, 8, b)
    # Inherited base field flows through unchanged.
    # CHECK: call @{{.*}}anchor{{.*}}([[A]])
    anchor(child.data)
    # Child-only field flows through unchanged.
    # CHECK: call @{{.*}}anchor{{.*}}([[B]])
    anchor(child.bias)


@triton.aggregate
class _AggMethodBase:
    val: tl.tensor
    BLOCK: tl.constexpr

    @triton.jit
    def doubled(self):
        return self.val + self.val


@triton.aggregate
class _AggMethodChild(_AggMethodBase):
    offset: tl.tensor


@filecheck_test
@triton.jit
def test_aggregate_inherited_method_ir():
    # CHECK-LABEL: test_aggregate_inherited_method_ir
    # CHECK: [[V:%.*]] = tt.make_range {end = 8 : i32, start = 0 : i32}
    # CHECK: [[O:%.*]] = tt.make_range {end = 16 : i32, start = 8 : i32}
    v = tl.arange(0, 8)
    o = tl.arange(8, 16)
    child = _AggMethodChild(v, 8, o)
    # The inherited method dispatches with mangling that includes the child type
    # — confirms the method came from the base but operates over child layout.
    # CHECK: [[D:%.*]] = tt.call @{{.*}}_AggMethodBase.doubled{{.*}}_AggMethodChild{{.*}}([[V]], [[O]])
    d = child.doubled()
    # CHECK: call @{{.*}}anchor{{.*}}([[D]])
    anchor(d)
    # CHECK: call @{{.*}}anchor{{.*}}([[O]])
    anchor(child.offset)


@triton.aggregate
class _AggReplaceState:
    vals: tl.tensor
    BLOCK: tl.constexpr


@filecheck_test
@triton.jit
def test_aggregate_replace_ir():
    # CHECK-LABEL: test_aggregate_replace_ir
    # CHECK: [[A:%.*]] = tt.make_range {end = 8 : i32, start = 0 : i32}
    # CHECK: [[B:%.*]] = tt.make_range {end = 16 : i32, start = 8 : i32}
    a = tl.arange(0, 8)
    b = tl.arange(8, 16)
    state = _AggReplaceState(a, 8)
    state2 = tl.aggregate_replace(state, vals=b)
    # Replaced field is the new tensor in the new aggregate.
    # CHECK: call @{{.*}}anchor{{.*}}([[B]])
    anchor(state2.vals)
    # Original aggregate still references original tensor.
    # CHECK: call @{{.*}}anchor{{.*}}([[A]])
    anchor(state.vals)


def test_dot_fp16_accumulator():

    @triton.jit
    def fp16_acc_kernel():
        c = tl.zeros([16, 16], dtype=tl.float16)
        a = tl.full([16, 16], 1, dtype=tl.float16)
        b = tl.full([16, 16], 1, dtype=tl.float16)
        tl.dot(a, b, c)

    run_parser(fp16_acc_kernel)


# ===-----------------------------------------------------------------------===#
# Loop-carried variable lowering
# ===-----------------------------------------------------------------------===#


@filecheck_test
@triton.jit
def test_loop_carry_readonly_alias():
    # CHECK-LABEL: test_loop_carry_readonly_alias
    # CHECK: %[[SEED:.*]] = arith.constant 1 : i32
    seed = 1
    acc = seed
    # CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[SEED]]) -> (i32)
    for i in range(3):
        # CHECK: %[[SUM:.*]] = arith.addi %[[ACC]], %[[SEED]] : i32
        acc = acc + seed
        # CHECK: scf.yield %[[SUM]] : i32
    anchor(acc)


@filecheck_test
@triton.jit
def test_loop_carry_shared_initial_value():
    # CHECK-LABEL: test_loop_carry_shared_initial_value
    # CHECK: %[[SEED:.*]] = arith.constant 1 : i32
    seed = 1
    a = seed
    b = seed
    # CHECK: scf.for {{.*}} iter_args(%[[A:.*]] = %[[SEED]], %[[B:.*]] = %[[SEED]])
    for i in range(3):
        # CHECK: %[[NEXT_A:.*]] = arith.addi %[[A]],
        a = a + 1
        # CHECK: %[[NEXT_B:.*]] = arith.addi %[[B]],
        b = b + 2
        # CHECK: scf.yield %[[NEXT_A]], %[[NEXT_B]] : i32, i32
    anchor(a)
    anchor(b)


@filecheck_test
@triton.jit
def test_loop_carry_while_shared_initial_value():
    # CHECK-LABEL: test_loop_carry_while_shared_initial_value
    # CHECK: %[[SEED:.*]] = arith.constant 1 : i32
    seed = 1
    a = seed
    b = seed
    # CHECK: scf.while
    while a < 4:
        # CHECK: } do {
        # CHECK-NEXT: ^bb0(%[[A:.*]]: i32, %[[B:.*]]: i32):
        # CHECK: %[[NEXT_A:.*]] = arith.addi %[[A]], %[[SEED]] : i32
        a = a + seed
        # CHECK: %[[NEXT_B:.*]] = arith.addi %[[B]],
        b = b + 2
        # CHECK: scf.yield %[[NEXT_A]], %[[NEXT_B]] : i32, i32
    anchor(a)
    anchor(b)


@filecheck_test
@triton.jit
def test_loop_carry_tuple_shared_initial_value():
    # CHECK-LABEL: test_loop_carry_tuple_shared_initial_value
    # CHECK: %[[SEED:.*]] = arith.constant 1 : i32
    seed = 1
    pair = (seed, seed)
    # CHECK: scf.for {{.*}} iter_args(%[[A:.*]] = %[[SEED]], %[[B:.*]] = %[[SEED]])
    for i in range(3):
        # CHECK: %[[NEXT_A:.*]] = arith.addi %[[A]],
        # CHECK: %[[NEXT_B:.*]] = arith.addi %[[B]],
        pair = (pair[0] + 1, pair[1] + 2)
        # CHECK: scf.yield %[[NEXT_A]], %[[NEXT_B]] : i32, i32
    anchor(pair[0])
    anchor(pair[1])


@filecheck_test
@triton.jit
def test_loop_carry_swap():
    # CHECK-LABEL: test_loop_carry_swap
    a = 0
    b = 1
    # CHECK: scf.for {{.*}} iter_args(%[[A:.*]] = {{.*}}, %[[B:.*]] = {{.*}})
    for i in range(3):
        a, b = b, a
        # CHECK: scf.yield %[[B]], %[[A]] : i32, i32
    anchor(a)
    anchor(b)


@filecheck_test
@triton.jit
def test_loop_carry_invariant_identity():
    # CHECK-LABEL: test_loop_carry_invariant_identity
    seed = 1
    alias = seed
    acc = 0
    # CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i32)
    for i in range(3):
        tl.static_assert(alias is seed)
        acc = acc + alias
    anchor(acc)


@filecheck_test
@triton.jit
def test_loop_carry_empty_nested():
    # CHECK-LABEL: test_loop_carry_empty_nested
    # CHECK: %[[SEED:.*]] = arith.constant 1 : i32
    seed = 1
    # CHECK: scf.for
    # CHECK-NOT: iter_args
    for i in range(2):
        # CHECK: scf.while : () -> ()
        while seed < 0:
            # CHECK: } do {
            # CHECK: tt.call @{{.*}}anchor{{.*}}(%[[SEED]])
            anchor(seed)
            # CHECK: scf.yield


@triton.jit
def _loop_carry_nest_depth_3(KIND: tl.constexpr):
    acc = 0
    if KIND == 0:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    acc += 1
    elif KIND == 1:
        i = 0
        while i < 2:
            j = 0
            while j < 2:
                k = 0
                while k < 2:
                    acc += 1
                    k += 1
                j += 1
            i += 1
    elif KIND == 2:
        for i in range(2):
            j = 0
            while j < 2:
                for k in range(2):
                    acc += 1
                j += 1
    else:
        i = 0
        while i < 2:
            for j in range(2):
                k = 0
                while k < 2:
                    acc += 1
                    k += 1
            i += 1
    anchor(acc)


@pytest.mark.parametrize("kind", range(4), ids=["for", "while", "for-while-for", "while-for-while"])
def test_loop_carry_discovery_avoids_exponential_revisits(monkeypatch, kind):
    """A depth-three body is generated four times, rather than eight."""
    per_body = collections.Counter()
    visit = CodeGenerator.visit_compound_statement

    def counted(self, stmts):
        per_body[id(stmts)] += 1
        return visit(self, stmts)

    monkeypatch.setattr(CodeGenerator, "visit_compound_statement", counted)
    run_parser(_loop_carry_nest_depth_3, args=(kind, ))
    assert max(per_body.values()) == 4


def test_const_ptr_is_constant_addrspace():

    @triton.jit
    def kernel(In: tl.const, Out, N, BLOCK: tl.constexpr):
        # CHECK-LABEL: tt.func public @kernel
        # A `tl.const` pointer argument is tagged with Triton's constant address
        # CHECK-SAME: %arg0: !tt.ptr<f32, "constant">
        # A plain pointer stays global
        # CHECK-SAME: %arg1: !tt.ptr<f32> {
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(Out + offs, tl.load(In + offs, mask=mask), mask=mask)

    run_filecheck_test(kernel, args=(MockTensor(tl.float32), MockTensor(tl.float32), 8, 128))
