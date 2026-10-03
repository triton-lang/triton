// RUN: not triton-opt %s --nvgpu-warp-specialization=num-stages=2 -o /dev/null 2>&1 | FileCheck %s

// CHECK: warp specialization cannot fall back from unsupported data partition in function with preexisting async_task_id attributes

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @preexisting_unsupported_gather(%arg0: !tt.tensordesc<128x128xf16>, %arg1: !tt.tensordesc<128x128xf16>, %arg2: !tt.tensordesc<128x128xf16>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x128xi32, #blocked>
    %indices_b = arith.constant dense<1> : tensor<128x128xi32, #blocked1>
    %init = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x128xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked>
      %gathered_a = tt.gather %a[%indices] {axis = 1 : i32} : (tensor<128x128xf16, #blocked>, tensor<128x128xi32, #blocked>) -> tensor<128x128xf16, #blocked>
      %a_smem = ttg.local_alloc %gathered_a : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked1>
      %gathered_b = tt.gather %b[%indices_b] {axis = 1 : i32} : (tensor<128x128xf16, #blocked1>, tensor<128x128xi32, #blocked1>) -> tensor<128x128xf16, #blocked1>
      %b_smem = ttg.local_alloc %gathered_b : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem> * !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma>
      scf.yield %dot : tensor<128x128xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x128xf16>, tensor<128x128xf16, #blocked1>
    tt.return
  }
}
