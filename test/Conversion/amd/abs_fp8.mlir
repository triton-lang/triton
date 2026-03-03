// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s

// Verify that abs on FNUZ FP8 types preserves NaN (0x80).
// The lowering must include an icmp+select to keep abs(NaN)=NaN,
// because simply clearing the sign bit turns 0x80 into 0x00 (zero).

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// CHECK-LABEL: llvm.func @abs_f8E4M3FNUZ
// CHECK: llvm.mlir.constant(127 : i8)
// CHECK: llvm.and
// CHECK: llvm.mlir.constant(-128 : i8)
// CHECK: llvm.icmp "eq"
// CHECK: llvm.select
  tt.func public @abs_f8E4M3FNUZ(%arg0: f8E4M3FNUZ) {
    %0 = math.absf %arg0 : f8E4M3FNUZ
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// CHECK-LABEL: llvm.func @abs_f8E5M2FNUZ
// CHECK: llvm.mlir.constant(127 : i8)
// CHECK: llvm.and
// CHECK: llvm.mlir.constant(-128 : i8)
// CHECK: llvm.icmp "eq"
// CHECK: llvm.select
  tt.func public @abs_f8E5M2FNUZ(%arg0: f8E5M2FNUZ) {
    %0 = math.absf %arg0 : f8E5M2FNUZ
    tt.return
  }
}

// -----

// Non-FNUZ FP8 types should NOT have the icmp+select NaN guard.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// CHECK-LABEL: llvm.func @abs_f8E4M3FN
// CHECK:     llvm.mlir.constant(127 : i8)
// CHECK:     llvm.and
// CHECK-NOT: llvm.icmp
// CHECK-NOT: llvm.select
// CHECK:     llvm.return
  tt.func public @abs_f8E4M3FN(%arg0: f8E4M3FN) {
    %0 = math.absf %arg0 : f8E4M3FN
    tt.return
  }
}
