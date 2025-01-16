// RUN: triton-opt %s -split-input-file --tritongpu-warp-spec-task-partition=num-consumer-groups=2 | FileCheck %s

// CHECK-LABEL: @matmul_persistent_tma_ws_cooperative_kernel
// CHECK: %[[#GA:]] = tt.experimental_descriptor_load {{.*}} {async_task_id = dense<0> : vector<1xi32>}
// CHECK: %[[#LA:]] = triton_gpu.local_alloc %[[#GA]]
// CHECK: %[[#GB:]] = tt.experimental_descriptor_load {{.*}} {async_task_id = dense<0> : vector<1xi32>}
// CHECK: %[[#LB:]] = triton_gpu.local_alloc %[[#GB]]
// CHECK: %[[#C:]] = triton_nvidia_gpu.warp_group_dot %[[#LA]], %[[#LB]], {{.*}} {async_task_id = dense<[1, 2]> : vector<2xi32>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_persistent_tma_ws_cooperative_kernel(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
        %23 = tt.experimental_descriptor_load %arg0[%18, %arg9] : !tt.ptr<i8, 0> -> tensor<128x64xf16, #blocked>
        %24 = triton_gpu.local_alloc %23 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
        %25 = tt.experimental_descriptor_load %arg1[%arg9, %19] : !tt.ptr<i8, 0> -> tensor<64x256xf16, #blocked1>
        %26 = triton_gpu.local_alloc %25 : (tensor<64x256xf16, #blocked1>) -> !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory>
        %27 = triton_nvidia_gpu.warp_group_dot %24, %26, %arg8 {inputPrecision = 0 : i32} : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x256xf32, #mma>
        %28 = arith.addi %arg9, %c64_i32 : i32
        scf.yield %27, %28 : tensor<128x256xf32, #mma>, i32
      }
      %21 = arith.truncf %20#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %22 = triton_gpu.convert_layout %21 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.experimental_descriptor_store %arg2[%18, %19], %22 : !tt.ptr<i8, 0>, tensor<128x256xf16, #blocked1>
    }
    tt.return
  }
}
