// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s --check-prefix=CHECK-GFX950

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK: llvm.func @min_max
// CHECK-COUNT-2: llvm.fcmp
// CHECK: llvm.or
// CHECK: llvm.intr.minnum
// CHECK-COUNT-2: llvm.fcmp
// CHECK: llvm.or
// CHECK: llvm.intr.maxnum

// CHECK-GFX950: llvm.func @min_max
// CHECK-GFX950-NEXT: llvm.intr.minimum
// CHECK-GFX950-NEXT: llvm.intr.maximum
  tt.func public @min_max(%arg0: f32, %arg1: f32) {
    %0 = arith.minimumf %arg0, %arg1 : f32
    %1 = arith.maximumf %arg0, %arg1 : f32
    tt.return
  }
}
