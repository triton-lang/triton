// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: buffer_load_store_vec4
    tt.func @buffer_load_store_vec4(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        // Load 8 elements from A with two vectorized load instruction
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load {{.*}} : vector<4xf32>
        %9 = amdgpu.buffer_load %arg0[%4] : tensor<256xf32, #blocked0>
        // Load 8 elements from B with two vectorized load instruction
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load {{.*}} : vector<4xf32>
        %10 = amdgpu.buffer_load %arg1[%4] : tensor<256xf32, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.store {{.*}} : vector<4xf32>
        amdgpu.buffer_store %11, %arg2[%4]: tensor<256xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: buffer_load_store_vec1
    tt.func @buffer_load_store_vec1(%arg0: !tt.ptr<f32> , %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        %5 = tt.splat %arg3 : i32 -> tensor<256xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<256xi32, #blocked0>
        // Load 8 elements from A with two vectorized load instruction
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load {{.*}} : f32
        %9 = amdgpu.buffer_load %arg0[%4], %7 : tensor<256xf32, #blocked0>
        // Load 8 elements from A with two vectorized load instruction
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load {{.*}} : f32
        %10 = amdgpu.buffer_load %arg1[%4], %7 : tensor<256xf32, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
        // Load 8 elements from A with two vectorized load instruction
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.store {{.*}} : f32
        amdgpu.buffer_store %11, %arg2[%4], %7 : tensor<256xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load_store_vec2
    tt.func @buffer_load_store_vec2(%arg0: !tt.ptr<f16> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f16>{tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f16>{tt.divisibility = 4: i32}, %arg3: i32{tt.divisibility = 4: i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        %5 = tt.splat %arg3 : i32 -> tensor<256xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<256xi32, #blocked0>
        // Load 8 elements from A with two vectorized load instruction
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}} : i32
        %9 = amdgpu.buffer_load %arg0[%4], %7 : tensor<256xf16, #blocked0>
        // Load 8 elements from A with two vectorized load instruction
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}} : i32
        %10 = amdgpu.buffer_load %arg1[%4], %7 : tensor<256xf16, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf16, #blocked0>
        // Load 8 elements from A with two vectorized load instruction
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}} : i32
        amdgpu.buffer_store %11, %arg2[%4], %7 : tensor<256xf16, #blocked0>
        tt.return
  }
}
