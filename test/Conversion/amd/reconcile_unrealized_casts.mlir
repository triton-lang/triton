// RUN: triton-opt %s --reconcile-unrealized-casts | FileCheck %s

// AMD's make_llir runs reconcile-unrealized-casts right after
// ConvertTritonGPUToLLVM. A no-op convert_layout on a loop-carried value (e.g.
// between two MFMA layouts that are physically identical at a given shape) can
// survive that partial conversion as a `struct -> tensorA -> tensorB -> struct`
// cast chain that transitively converts a value back to its original type. The
// reconcile pass must eliminate such a chain; otherwise it reaches, and fails,
// MLIR->LLVM translation.

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1], instrShape = [16, 16, 32], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @reconcile_roundtrip_cast_chain
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK: llvm.return %arg0
  llvm.func @reconcile_roundtrip_cast_chain(
      %arg0: !llvm.struct<(f32, f32, f32, f32)>) -> !llvm.struct<(f32, f32, f32, f32)> {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(f32, f32, f32, f32)> to tensor<16x16xf32, #mma>
    %1 = builtin.unrealized_conversion_cast %0 : tensor<16x16xf32, #mma> to tensor<16x16xf32, #mma1>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<16x16xf32, #mma1> to !llvm.struct<(f32, f32, f32, f32)>
    llvm.return %2 : !llvm.struct<(f32, f32, f32, f32)>
  }
}
