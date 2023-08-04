// RUN: ENABLE_TMA=1 triton-opt %s -split-input-file -triton-nvidia-gpu-materialize-load-store=compute-capability=90 -canonicalize | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @matmul_loop
  tt.func @matmul_loop(%A : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<64x16xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i64
    %c16 = arith.constant 16 : i64
    %c64 = arith.constant 64 : i64
    // CHECK: %[[TENSOR_PTR:.*]] = tt.make_tensor_ptr
    %a_tileptr_init = tt.make_tensor_ptr %A, [%c64, %c16], [%c16, %c1], [%c0, %c0] { order = array<i32: 1, 0> } : !tt.ptr<tensor<64x16xf16>, 1>
    // CHECK: %[[BUFFER:.*]] = triton_gpu.alloc_tensor : tensor<1x64x16xf16, #shared>
    // CHECK: %[[MBAR:.*]] = triton_nvidia_gpu.alloc_mbarrier {count = 1 : i32} : !tt.ptr<i64, 3>
    // CHECK: triton_nvidia_gpu.mbarrier_arrive %[[MBAR]], %{{.*}} {operand_segment_sizes = array<i32: 1, 1, 0>, trackAsyncOp = false, txCount = 2048 : i32} : !tt.ptr<i64, 3>, i1
    // CHECK: %[[INSERT:.*]] = triton_nvidia_gpu.insert_slice_async_v2 %[[TENSOR_PTR]], %[[BUFFER]], %{{.*}}, %[[MBAR]]
    // CHECK: %[[EXT:.*]] = triton_gpu.extract_slice %[[INSERT]][0, 0, 0] [1, 64, 16] [1, 1, 1] : tensor<1x64x16xf16, #shared> to tensor<64x16xf16, #shared>
    // CHECK: %[[CVT:.*]] = triton_gpu.convert_layout %[[EXT]] : (tensor<64x16xf16, #shared>) -> tensor<64x16xf16, #blocked>
    // CHECK: triton_nvidia_gpu.mbarrier_wait %[[MBAR]], %false : <i64, 3>
    // CHECK: tt.return %[[CVT]] : tensor<64x16xf16, #blocked>
    %res = tt.load %a_tileptr_init {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x16xf16>, 1> -> tensor<64x16xf16, #blocked>
    tt.return %res : tensor<64x16xf16, #blocked>
  }
}
