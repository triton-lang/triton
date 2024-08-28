// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32_scalar
  tt.func @atomic_add_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : i1, %arg2 : f32) {
    // CHECK: llvm.cond_br
    // CHECK: llvm.atomicrmw
    // CHECK: llvm.store
    // CHECK: llvm.br
    // CHECK: rocdl.barrier
    // CHECK: llvm.load
    // CHECK: llvm.store
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %0 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.cond_br
    // CHECK: llvm.atomicrmw
    // CHECK: llvm.atomicrmw
    // CHECK: %[[ADDR1:.*]] = llvm.extractvalue
    // CHECK: %[[ADDR2:.*]] = llvm.extractvalue
    // CHECK: llvm.store %{{.*}}, %[[ADDR1]]
    // CHECK: llvm.store %{{.*}}, %[[ADDR2]]
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.store %arg0, %0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

// CHECK-LABEL: v_dot_i8
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.target" = "hip:gfx942", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @v_dot_i8(%arg0: tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, %arg1: tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, %arg2: tensor<16x16xi32, #blocked>) {
    // CHECK-4: llvm.call_intrinsic "llvm.amdgcn.sdot4"
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xi32, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: v_dot_fp16
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.target" = "hip:gfx942", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @v_dot_fp16(%arg0: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, %arg1: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, %arg2: tensor<16x16xf32, #blocked>) {
    // CHECK-8: llvm.call_intrinsic "llvm.amdgcn.fdot2"
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: v_dot_fp16_fp16
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.target" = "hip:gfx942", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @v_dot_fp16_fp16(%arg0: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, %arg1: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, %arg2: tensor<16x16xf16, #blocked>) {
    // CHECK-4: llvm.call_intrinsic "llvm.amdgcn.sdot4"
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf16, #blocked>
    tt.return
  }
}
