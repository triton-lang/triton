# RUN: %PYTHON %s | FileCheck %s

from triton._C.triton_mlir_bindings.dialects import triton as tt
from triton._C.triton_mlir_bindings.ir import Context, IntegerType, RankedTensorType, Location


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        tt.register_dialect()
        f()
    return f


# CHECK-LABEL: TEST: testMakeRangeOp
@run
def testMakeRangeOp():
    i32 = IntegerType.get_signless(32)
    tensor_type = RankedTensorType.get([10], i32)
    r = tt.MakeRangeOp(tensor_type, 0, 10)
    # CHECK: %{{.*}} = "tt.make_range"() {end = 10 : i32, start = 0 : i32} : () -> tensor<10xi32>
    print(r)
