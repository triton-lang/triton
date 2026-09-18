// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck --check-prefix=SELECT %s
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck --check-prefix=ARITH %s
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck --check-prefix=SELECT %s
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck --check-prefix=ARITH %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // SELECT-LABEL: llvm.func @fp8e4m3fn_to_fp16
  // ARITH-LABEL: llvm.func @fp8e4m3fn_to_fp16
  tt.func @fp8e4m3fn_to_fp16(%arg0: tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // The fixture converts eight elements per thread. Each value
    // uses one branchless magnitude conversion and one select for signed NaN,
    // rather than an eight-entry denormal lookup chain.
    // SELECT-COUNT-8: llvm.select
    // SELECT-NOT: llvm.select
    // ARITH: llvm.fmul
    // ARITH: llvm.fptrunc
    // ARITH: llvm.mlir.constant(32256 : i16)
    // ARITH: llvm.icmp "eq"
    // ARITH: llvm.select
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}
