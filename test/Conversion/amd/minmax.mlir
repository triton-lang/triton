// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s --check-prefixes=CHECK,SW-NAN
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s --check-prefixes=CHECK,LLVM-MINMAX
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck %s --check-prefixes=CHECK,SW-NAN
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 | FileCheck %s --check-prefixes=CHECK,LLVM-MINMAX
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 | FileCheck %s --check-prefixes=CHECK,SW-NAN
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s --check-prefixes=CHECK,LLVM-MINMAX
//
// This grouping reflects Triton's supportMaximumMinimum() table, not the
// target's native instruction capabilities.

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

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @clamp_propagate_nan
  // AMD's higher-benefit ClampFOpConversion lowers f16/f32 through v_med3.
  // Use f64 here to exercise the generic supportMaximumMinimum() path.
  tt.func public @clamp_propagate_nan(%x: f64, %lo: f64, %hi: f64) {
    // LLVM-MINMAX-NOT: llvm.fcmp
    // LLVM-MINMAX-NOT: llvm.intr.maxnum
    // LLVM-MINMAX-NOT: llvm.intr.minnum
    // LLVM-MINMAX-NOT: llvm.select
    // LLVM-MINMAX: llvm.intr.maximum
    // LLVM-MINMAX-NEXT: llvm.intr.minimum
    // LLVM-MINMAX-NOT: llvm.fcmp
    // LLVM-MINMAX-NOT: llvm.intr.maxnum
    // LLVM-MINMAX-NOT: llvm.intr.minnum
    // LLVM-MINMAX-NOT: llvm.select
    // SW-NAN-NOT: llvm.intr.maximum
    // SW-NAN-NOT: llvm.intr.minimum
    // SW-NAN: llvm.fcmp "une"
    // SW-NAN: llvm.intr.maxnum
    // SW-NAN-NEXT: llvm.intr.minnum
    // SW-NAN: llvm.select
    // SW-NAN-NOT: llvm.intr.maximum
    // SW-NAN-NOT: llvm.intr.minimum
    %0 = tt.clampf %x, %lo, %hi, propagateNan = all : f64
    tt.return
  }
}
