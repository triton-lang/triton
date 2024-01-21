// RUN: triton-opt %s -split-input-file --decompose-unsupported-conversions --convert-triton-gpu-to-llvm=compute-capability=90 2>&1 | FileCheck %s

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
    // CHECK: nvgpu.tma_load_tiled %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 2, 0>} : !llvm.ptr<3>, !llvm.ptr<3>, !llvm.ptr<1>, i64, i1, i32, i32
    %res = triton_nvidia_gpu.insert_slice_tma %src, %dst, %c0, %mbar {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<1x64x64xf16, #shared>, i32, !tt.ptr<i64, 3> -> tensor<1x64x64xf16, #shared>
    tt.return
  }
}
