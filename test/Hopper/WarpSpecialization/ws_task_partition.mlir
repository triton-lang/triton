// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-task-partition=num-warp-groups=3 | FileCheck %s

// CHECK-LABEL: @matmul_persistent_tma_ws_cooperative_kernel
// CHECK: %[[#GA:]] = tt.descriptor_load {{.*}} {async_task_id = dense<0> : vector<1xi32>}
// CHECK: %[[#LA:]] = ttg.local_alloc %[[#GA]]
// CHECK: %[[#GB:]] = tt.descriptor_load {{.*}} {async_task_id = dense<0> : vector<1xi32>}
// CHECK: %[[#LB:]] = ttg.local_alloc %[[#GB]]
// CHECK: %[[#C:]] = ttng.warp_group_dot %[[#LA]], %[[#LB]], {{.*}} {async_task_id = dense<[1, 2]> : vector<2xi32>
// CHECK: tt.descriptor_store {{.*}} {async_task_id = dense<[1, 2]> : vector<2xi32>}


#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_persistent_tma_ws_cooperative_kernel(%arg0: !tt.tensordesc<tensor<128x64xf16>>, %arg1: !tt.tensordesc<tensor<64x256xf16>>, %arg2: !tt.tensordesc<tensor<128x256xf16>>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = arith.addi %arg3, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = arith.addi %arg4, %c255_i32 : i32
    %3 = arith.divsi %2, %c256_i32 : i32
    %4 = arith.muli %1, %3 : i32
    %5 = tt.get_program_id x : i32
    %6 = tt.get_num_programs x : i32
    %7 = arith.muli %3, %c8_i32 : i32
    %8 = arith.addi %arg5, %c63_i32 : i32
    %9 = arith.divsi %8, %c64_i32 : i32
    scf.for %arg6 = %5 to %4 step %6  : i32 {
      %10 = arith.divsi %arg6, %7 : i32
      %11 = arith.muli %10, %c8_i32 : i32
      %12 = arith.subi %1, %11 : i32
      %13 = arith.minsi %12, %c8_i32 : i32
      %14 = arith.remsi %arg6, %7 : i32
      %15 = arith.remsi %14, %13 : i32
      %16 = arith.addi %11, %15 : i32
      %17 = arith.divsi %14, %13 : i32
      %18 = arith.muli %16, %c128_i32 : i32
      %19 = arith.muli %17, %c256_i32 : i32
      %true = arith.constant true
      %false = arith.constant false
      %20:2 = scf.for %arg7 = %c0_i32 to %9 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
        %23 = tt.descriptor_load %arg0[%18, %arg9] : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
        %24 = ttg.local_alloc %23 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
        %25 = tt.descriptor_load %arg1[%arg9, %19] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked1>
        %26 = ttg.local_alloc %25 : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #ttg.shared_memory>
        %27 = ttng.warp_group_dot %24, %26, %arg8 {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x256xf16, #shared, #ttg.shared_memory> -> tensor<128x256xf32, #mma>
        %28 = arith.addi %arg9, %c64_i32 : i32
        scf.yield %27, %28 : tensor<128x256xf32, #mma>, i32
      }
      %21 = arith.truncf %20#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %22 = ttg.convert_layout %21 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.descriptor_store %arg2[%18, %19], %22 : !tt.tensordesc<tensor<128x256xf16>>, tensor<128x256xf16, #blocked1>
    }
    tt.return
  }
}
