// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-task-partition=num-warp-groups=3 | FileCheck %s

// CHECK-LABEL: @matmul_persistent_tma_ws_cooperative_kernel
// CHECK: %[[#GA:]] = tt.descriptor_load {{.*}} {async_task_id = array<i32: 0>}
// CHECK: %[[#LA:]] = ttg.local_alloc %[[#GA]]
// CHECK: %[[#GB:]] = tt.descriptor_load {{.*}} {async_task_id = array<i32: 0>}
// CHECK: %[[#LB:]] = ttg.local_alloc %[[#GB]]
// CHECK: %[[#C:]] = ttng.warp_group_dot %[[#LA]], %[[#LB]], {{.*}} {async_task_id = array<i32: 1, 2>
// CHECK: tt.descriptor_store {{.*}} {async_task_id = array<i32: 1, 2>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 0}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_persistent_tma_ws_cooperative_kernel(%arg0: !tt.tensordesc<tensor<128x64xf16>>, %arg1: !tt.tensordesc<tensor<64x256xf16>>, %arg2: !tt.tensordesc<tensor<128x256xf16>>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    scf.for %arg6 = %0 to %arg3 step %1  : i32 {
      %2:2 = scf.for %arg7 = %c0_i32 to %arg5 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
        %5 = tt.descriptor_load %arg0[%arg6, %arg9] : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
        %6 = ttg.local_alloc %5 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %7 = tt.descriptor_load %arg1[%arg9, %arg6] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked1>
        %8 = ttg.local_alloc %7 : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %9 = ttng.warp_group_dot %6, %8, %arg8 {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        %10 = arith.addi %arg9, %c64_i32 : i32
        scf.yield %9, %10 : tensor<128x256xf32, #mma>, i32
      }
      %3 = arith.truncf %2#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %4 = ttg.convert_layout %3 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.descriptor_store %arg2[%arg6, %arg6], %4 : !tt.tensordesc<tensor<128x256xf16>>, tensor<128x256xf16, #blocked1>
    }
    tt.return
  }
}
