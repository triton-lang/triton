// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK: llvm.func @min_max
// CHECK: llvm.intr.minimum
// CHECK-NEXT: llvm.intr.maximum
  tt.func public @min_max(%arg0: f32, %arg1: f32) {
    %0 = arith.minimumf %arg0, %arg1 : f32
    %1 = arith.maximumf %arg0, %arg1 : f32
    tt.return
  }
}
