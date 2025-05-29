// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s --check-prefix=GFX942
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s --check-prefix=GFX950

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// GFX942: llvm.func @min_max
// GFX942-COUNT-2: llvm.fcmp
// GFX942: llvm.or
// GFX942: llvm.intr.minnum
// GFX942-COUNT-2: llvm.fcmp
// GFX942: llvm.or
// GFX942: llvm.intr.maxnum

// GFX950: llvm.func @min_max
// GFX950-NEXT: llvm.intr.minimum
// GFX950-NEXT: llvm.intr.maximum
  tt.func public @min_max(%arg0: f32, %arg1: f32) {
    %0 = arith.minimumf %arg0, %arg1 : f32
    %1 = arith.maximumf %arg0, %arg1 : f32
    tt.return
  }
}
