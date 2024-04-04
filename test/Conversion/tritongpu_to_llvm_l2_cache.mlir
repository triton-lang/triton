// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s
// CHECK: %[[policy:[0-9]*]] = {{.*}}createpolicy.fractional.L2::evict_last.b64
// CHECK: ld.global.L1::evict_last.L2::cache_hint.v4.b32 {{.*}} %{{[0-9]+}}, %[[policy]]
// CHECK: ld.global.L1::evict_last.L2::cache_hint.v4.b32
// CHECK: st.global.L1::evict_last.L2::cache_hint.v4.b32

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel_0d1d2d3d(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<512xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<512xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<512x!tt.ptr<f32, 1>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<f32, 1>, #blocked>, tensor<512xi32, #blocked>
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<512xf32, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f32, 1> -> tensor<512x!tt.ptr<f32, 1>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<512x!tt.ptr<f32, 1>, #blocked>, tensor<512xi32, #blocked>
    %12 = tt.load %11, %6 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<512xf32, #blocked>
    %13 = arith.addf %9, %12 : tensor<512xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32, 1> -> tensor<512x!tt.ptr<f32, 1>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<512x!tt.ptr<f32, 1>, #blocked>, tensor<512xi32, #blocked>
    tt.store %15, %13, %6 {cache = 1 : i32, evict = 3 : i32} : tensor<512xf32, #blocked>
    tt.return
  }
}
