// RUN: triton-opt %s -tritongpu-pipeline | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @softmax_kernel
tt.func public @softmax_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<0xFF800000> : tensor<128xf32, #blocked>
  %0 = tt.get_program_id x : i32
  %1 = tt.get_num_programs x : i32
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
  %3 = tt.splat %arg5 : i32 -> tensor<128xi32, #blocked>
  // CHECK: [[MASK:%.*]] = arith.cmpi slt, {{.*}} tensor<128xi32,
  %4 = arith.cmpi slt, %2, %3 : tensor<128xi32, #blocked>
  // CHECK: scf.for
  scf.for %arg6 = %0 to %arg4 step %1  : i32 {
    %5 = tt.splat %arg1 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %2 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    // CHECK: [[RESULT:%.*]] = triton_gpu.local_load
    // CHECK-NEXT: arith.select [[MASK]], [[RESULT]], %cst
    %7 = tt.load %6, %4, %cst {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked>
    %8 = tt.splat %arg0 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %2 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %9, %7, %4 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x!tt.ptr<f32>, #blocked>
  } {tt.num_stages = 2 : i32}
  tt.return
}

}
