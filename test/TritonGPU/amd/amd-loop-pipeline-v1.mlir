// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc("/data/users/dberard/triton-env/scripts/matmul.py":6:0)
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = false}>
module attributes {"triton_gpu.target" = "hip:gfx942", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @use_dep_args
  tt.func @use_dep_args(%a_ptrs: tensor<64x32x!tt.ptr<bf16>, #blocked>, %b_ptrs: tensor<32x64x!tt.ptr<bf16>, #blocked1>, %loop_range: i32) -> (tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<bf16>, #blocked>, tensor<32x64x!tt.ptr<bf16>, #blocked1>) {
    %cst = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %cst2 = arith.constant dense<2048> : tensor<32x64xi32, #blocked1>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: tt.load
    // CHECK: [[FOR_OUT:%[a-z0-9_]+]]:{{[0-9]+}} = scf.for
    %for:3 = scf.for %arg6 = %c0_i32 to %loop_range step %c32_i32 iter_args(%arg7 = %cst_0, %arg8 = %a_ptrs, %arg9 = %b_ptrs) -> (tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<bf16>, #blocked>, tensor<32x64x!tt.ptr<bf16>, #blocked1>)  : i32 {
      %63 = tt.load %arg8 : tensor<64x32x!tt.ptr<bf16>, #blocked>
      %64 = tt.load %arg9 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
      %65 = triton_gpu.convert_layout %63 : tensor<64x32xbf16, #blocked> -> tensor<64x32xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %66 = triton_gpu.convert_layout %64 : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %67 = tt.dot %65, %66, %arg7 : tensor<64x32xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xbf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
      %68 = tt.addptr %arg8, %cst : tensor<64x32x!tt.ptr<bf16>, #blocked>, tensor<64x32xi32, #blocked>
      %69 = tt.addptr %arg9, %cst2 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
      scf.yield %67, %68, %69 : tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<bf16>, #blocked>, tensor<32x64x!tt.ptr<bf16>, #blocked1>
    }
    // CHECK: tt.return {{[^,]+}}, [[FOR_OUT]]#3, [[FOR_OUT]]#4
    tt.return %for#0, %for#1, %for#2 : tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<bf16>, #blocked>, tensor<32x64x!tt.ptr<bf16>, #blocked1>
  }
}
