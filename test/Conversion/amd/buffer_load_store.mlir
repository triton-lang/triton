// RUN: AMDGCN_USE_BUFFER_OPS=1 triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: buffer_load_store_vec8
    tt.func @buffer_load_store_vec8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // Load 8 elements from A with two vectorized load instruction
    // CHECK-COUNT-5: llvm.select
    // CHECK: %[[mask0:.*]] = llvm.select
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[mask0]]
    // CHECK: %[[mask1:.*]] = llvm.select
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[mask1]]
    // CHECK: %[[mask2:.*]] = llvm.select
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[mask2]]
    // CHECK: %[[mask3:.*]] = llvm.select
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[mask3]]
    %9 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    // CHECK: %[[mask4:.*]] = llvm.select
    // CHECK: rocdl.raw.ptr.buffer.store{{.*}}, {{.*}}, %[[mask4]]
    // CHECK: %[[mask5:.*]] = llvm.select
    // CHECK: rocdl.raw.ptr.buffer.store{{.*}}, {{.*}}, %[[mask5]]
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}
