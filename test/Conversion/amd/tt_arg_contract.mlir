// The tt.noalias argument attribute (a frontend-surfaced caller contract) lowers
// to the LLVM parameter attribute llvm.noalias on kernel pointer arguments.
// See FuncOpToLLVM / handlePointerContractArgs.

// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @contract
  // CHECK-SAME: %arg0: !llvm.ptr<1> {llvm.noalias
  // CHECK-NOT: %arg1: !llvm.ptr<1> {llvm.noalias
  tt.func public @contract(
      %arg0: !tt.ptr<i16> {tt.noalias = 1 : i32},
      %arg1: !tt.ptr<f16>) {
    tt.return
  }
}
