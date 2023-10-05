// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm 2>&1 | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 4], CTAOrder = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 4], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 4 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tma_multicast_no_broadcast
  tt.func @tma_multicast_no_broadcast(%basePtr: !tt.ptr<f16> {tt.divisibility = 8 : i32},
                                        %dim0: i64, %dim1: i64,
                                        %stride0: i64, %stride1: i64,
                                        %coord0: i32, %coord1: i32) {
    %mbar = triton_nvidia_gpu.alloc_mbarrier { count = 128 : i32 } : !tt.ptr<i64, 3>
    %dst = triton_gpu.alloc_tensor : tensor<1x64x64xf16, #shared>
    %c0 = arith.constant 0 : i32
    %src = tt.make_tensor_ptr %basePtr, [%dim0, %dim1], [%stride0, %stride1], [%coord0, %coord1] {order = array<i32: 1, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>
    // CHECK: nvgpu.tma_load_tiled %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 2, 0>} : !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32
    %res = triton_nvidia_gpu.insert_slice_async_v2 %src, %dst, %c0, %mbar {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operand_segment_sizes = array<i32: 1, 1, 1, 1, 0, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<1x64x64xf16, #shared>, i32, !tt.ptr<i64, 3> -> tensor<1x64x64xf16, #shared>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 4 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tma_multicast_const_mask
  tt.func @tma_multicast_const_mask(%basePtr: !tt.ptr<f16> {tt.divisibility = 8 : i32},
                                      %dim0: i64, %dim1: i64,
                                      %stride0: i64, %stride1: i64,
                                      %coord0: i32, %coord1: i32) {
    %mbar = triton_nvidia_gpu.alloc_mbarrier { count = 128 : i32 } : !tt.ptr<i64, 3>
    %dst = triton_gpu.alloc_tensor : tensor<1x64x64xf16, #shared>
    %c0 = arith.constant 0 : i32
    %src = tt.make_tensor_ptr %basePtr, [%dim0, %dim1], [%stride0, %stride1], [%coord0, %coord1] {order = array<i32: 1, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>
    // CHECK: %[[C15:.*]] = llvm.mlir.constant(15 : i16) : i16
    // CHECK: nvgpu.tma_load_tiled %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C15]]
    %res = triton_nvidia_gpu.insert_slice_async_v2 %src, %dst, %c0, %mbar {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operand_segment_sizes = array<i32: 1, 1, 1, 1, 0, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<1x64x64xf16, #shared>, i32, !tt.ptr<i64, 3> -> tensor<1x64x64xf16, #shared>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 2], CTAOrder = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 2], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 4 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tma_multicast_variable_mask
  tt.func @tma_multicast_variable_mask(%basePtr: !tt.ptr<f16> {tt.divisibility = 8 : i32},
                                         %dim0: i64, %dim1: i64,
                                         %stride0: i64, %stride1: i64,
                                         %coord0: i32, %coord1: i32) {
    %mbar = triton_nvidia_gpu.alloc_mbarrier { count = 128 : i32 } : !tt.ptr<i64, 3>
    %dst = triton_gpu.alloc_tensor : tensor<1x64x64xf16, #shared>
    %c0 = arith.constant 0 : i32
    %src = tt.make_tensor_ptr %basePtr, [%dim0, %dim1], [%stride0, %stride1], [%coord0, %coord1] {order = array<i32: 1, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>
    // CHECK: nvgpu.cluster_id
    // CHECK: nvgpu.tma_load_tiled
    %res = triton_nvidia_gpu.insert_slice_async_v2 %src, %dst, %c0, %mbar {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operand_segment_sizes = array<i32: 1, 1, 1, 1, 0, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<1x64x64xf16, #shared>, i32, !tt.ptr<i64, 3> -> tensor<1x64x64xf16, #shared>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: @tma_store
  tt.func @tma_store(%basePtr: !tt.ptr<f32> {tt.divisibility = 8 : i32},
                       %dim0: i64, %dim1: i64,
                       %stride0: i64, %stride1: i64,
                       %coord0: i32, %coord1: i32) {
    %src = triton_gpu.alloc_tensor : tensor<64x64xf32, #shared>
    %c0 = arith.constant 0 : i32
    %dst = tt.make_tensor_ptr %basePtr, [%dim0, %dim1], [%stride0, %stride1], [%coord0, %coord1] {order = array<i32: 1, 0>} : !tt.ptr<tensor<64x64xf32, #blocked>, 1>
    // CHECK: nvgpu.tma_store_tiled %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.ptr<i8, 1>, !llvm.ptr<i8, 3>, i1, i32, i32
    triton_nvidia_gpu.store_async %dst, %src {cache = 1 : i32} : !tt.ptr<tensor<64x64xf32, #blocked>, 1>, tensor<64x64xf32, #shared>
    tt.return
  }
}
