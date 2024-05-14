// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

// CHECK-LABEL: dedup_by_constancy_mfma
// CHECK-COUNT-4: llvm.icmp "slt"
// CHECK-NOT: llvm.icmp "slt"
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @dedup_by_constancy_mfma(%arg0: i32 {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %1 = tt.splat %arg0 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %2 = arith.cmpi slt, %0, %1 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi1, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<32x1xi1, #mma>
    %4 = tt.broadcast %3 : tensor<32x1xi1, #mma> -> tensor<32x32xi1, #mma>
    %cst = arith.constant dense<0.100000e+00> : tensor<32x32xf16, #mma>
    %5 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #mma>
    %6 = tt.broadcast %5 : tensor<32x1x!tt.ptr<f16>, #mma> -> tensor<32x32x!tt.ptr<f16>, #mma>
    tt.store %6, %cst, %4 : tensor<32x32x!tt.ptr<f16>, #mma>
    tt.return
  }
}
