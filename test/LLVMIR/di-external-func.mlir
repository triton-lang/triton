// RUN: triton-opt %s -o - --mlir-print-debuginfo --enable-line-info --extract-variable-info | FileCheck %s

// COM: Check that fuseFuncArgVariables skips external declarations (no body),
// COM: while still inserting dbg intrinsics for non-external functions.

// CHECK: llvm.func @vprintf
// CHECK: llvm.func @kernel
// CHECK: llvm.intr.dbg.value
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @kernel(%arg0: !llvm.ptr<1> {tt.pointee_type = f32} loc(#loc1),
                    %arg1: i32 loc(#loc2)) {
    %c = llvm.mlir.constant(0 : i32) : i32 loc(#loc3)
    llvm.return
  }
}
#loc = loc("test.py":1:0)
#loc1 = loc("x_ptr"(#loc))
#loc2 = loc("n_elements"(#loc))
#loc3 = loc("c"(#loc))
