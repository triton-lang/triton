// RUN: not triton-opt %s --nvgpu-test-ws-data-partition=num-warp-groups=3 2>&1 | FileCheck %s

// A reduce on the dimension the accumulator is partitioned along has no slice
// implementation. The pass must reject it with a diagnostic instead of
// asserting mid-partition (#11958).

// CHECK: reduce on the warp-specialized partition dimension is not supported

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_on_partitioned_dim(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %1 = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2>} : i32
    scf.for %arg6 = %0 to %arg3 step %1  : i32 {
      %2 = tt.splat %arg0 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
      %3 = tt.splat %arg1 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
      %4 = scf.for %arg7 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg8 = %cst) -> (tensor<128x256xf32, #mma>)  : i32 {
        %8 = tt.load %2 {async_task_id = array<i32: 0>} : tensor<128x64x!tt.ptr<f16>, #blocked>
        %9 = ttg.local_alloc %8 {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %10 = tt.load %3 {async_task_id = array<i32: 0>} : tensor<64x256x!tt.ptr<f16>, #blocked1>
        %11 = ttg.local_alloc %10 {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %12 = ttng.warp_group_dot %9, %11, %arg8 {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        %13 = "tt.reduce"(%12) <{axis = 0 : i32}> ({
        ^bb0(%a: f32, %b: f32):
          %s = arith.addf %a, %b {async_task_id = array<i32: 1, 2>} : f32
          tt.reduce.return %s {async_task_id = array<i32: 1, 2>} : f32
        }) {async_task_id = array<i32: 1, 2>} : (tensor<128x256xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
        %14 = tt.expand_dims %13 {async_task_id = array<i32: 1, 2>, axis = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xf32, #mma>
        %15 = tt.broadcast %14 {async_task_id = array<i32: 1, 2>} : tensor<1x256xf32, #mma> -> tensor<128x256xf32, #mma>
        %16 = arith.mulf %12, %15 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma>
        scf.yield {async_task_id = array<i32: 0, 1, 2>} %16 : tensor<128x256xf32, #mma>
      } {async_task_id = array<i32: 0, 1, 2>}
      %5 = arith.truncf %4 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %6 = ttg.convert_layout %5 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      %7 = tt.splat %arg2 {async_task_id = array<i32: 1, 2>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
      tt.store %7, %6 {async_task_id = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f16>, #blocked1>
    }
    tt.return
  }
}
