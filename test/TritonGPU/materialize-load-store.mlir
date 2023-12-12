// RUN: ENABLE_TMA=1 triton-opt %s -split-input-file -triton-nvidia-gpu-materialize-load-store=compute-capability=90 -canonicalize | FileCheck %s

// CHECK-LABEL: @matmul_loop
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @matmul_loop(%A : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<64x16xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i64
    %c16 = arith.constant 16 : i64
    %c64 = arith.constant 64 : i64
    // CHECK: %[[MBAR:.*]] = triton_nvidia_gpu.alloc_mbarrier {count = 1 : i32} : !tt.ptr<i64, 3>
    // CHECK: %[[TENSOR_PTR:.*]] = tt.make_tensor_ptr
    %a_tileptr_init = tt.make_tensor_ptr %A, [%c64, %c16], [%c16, %c1], [%c0, %c0] { order = array<i32: 1, 0> } : !tt.ptr<tensor<64x16xf16>, 1>
    // CHECK: %[[BUFFER:.*]] = triton_gpu.alloc_tensor : tensor<1x64x16xf16, #shared>
    // CHECK: triton_nvidia_gpu.mbarrier_arrive %[[MBAR]], %{{.*}} {operandSegmentSizes = array<i32: 1, 1, 0>, trackAsyncOp = false, txCount = 2048 : i32} : !tt.ptr<i64, 3>, i1
    // CHECK: %[[INSERT:.*]] = triton_nvidia_gpu.insert_slice_async_v2 %[[TENSOR_PTR]], %[[BUFFER]], %{{.*}}, %[[MBAR]]
    // CHECK: %[[EXT:.*]] = triton_gpu.extract_slice %[[INSERT]][0, 0, 0] [1, 64, 16] [1, 1, 1] : tensor<1x64x16xf16, #shared> to tensor<64x16xf16, #shared>
    // CHECK: triton_nvidia_gpu.mbarrier_wait %[[MBAR]], %false : <i64, 3>
    // CHECK: %[[CVT:.*]] = triton_gpu.convert_layout %[[EXT]] : (tensor<64x16xf16, #shared>) -> tensor<64x16xf16, #blocked>
    // CHECK: tt.return %[[CVT]] : tensor<64x16xf16, #blocked>
    %res = tt.load %a_tileptr_init {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x16xf16>, 1> -> tensor<64x16xf16, #blocked>
    tt.return %res : tensor<64x16xf16, #blocked>
  }
}

// -----

// CHECK-LABEL: matmul_no_scf

#blockedA0 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blockedB0 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blockedA1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blockedB1 = #triton_gpu.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 16, 16]}>
#sharedA = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#sharedB = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func public @matmul_no_scf(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #blockedA0>, 1>
    %4 = arith.extsi %arg4 : i32 to i64
    %5 = arith.extsi %arg7 : i32 to i64
    %6 = tt.make_tensor_ptr %arg1, [%1, %4], [%c1_i64, %5], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<16x16xf16, #blockedB0>, 1>
    // CHECK: %[[LOADED_A:.*]] = triton_gpu.extract_slice
    // CHECK: %[[LOADED_B:.*]] = triton_gpu.extract_slice
    // CHECK-NOT: triton_gpu.convert_layout {{.*}}#shared{{.*}}->{{.*}}#blocked
    // CHECK: tt.dot %[[LOADED_A]], %[[LOADED_B]]
    %7 = tt.load %3 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x16xf16, #blockedA0>, 1> -> tensor<64x16xf16, #blockedA1>
    %8 = tt.load %6 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x16xf16, #blockedB0>, 1> -> tensor<16x16xf16, #blockedB1>
    %9 = triton_gpu.convert_layout %7 : (tensor<64x16xf16, #blockedA1>) -> tensor<64x16xf16, #sharedA>
    %10 = triton_gpu.convert_layout %8 : (tensor<16x16xf16, #blockedB1>) -> tensor<16x16xf16, #sharedB>
    %11 = tt.dot %9, %10, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x16xf16, #sharedA> * tensor<16x16xf16, #sharedB> -> tensor<64x16xf32, #mma>
    %12 = triton_gpu.convert_layout %11 : (tensor<64x16xf32, #mma>) -> tensor<64x16xf32, #blockedA1>
    %13 = arith.truncf %12 : tensor<64x16xf32, #blockedA1> to tensor<64x16xf16, #blockedA1>
    %14 = arith.extsi %arg8 : i32 to i64
    %15 = tt.make_tensor_ptr %arg2, [%0, %4], [%14, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #blockedA0>, 1>
    tt.store %15, %13 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<64x16xf16, #blockedA0>, 1>, tensor<64x16xf16, #blockedA1>
    tt.return
  }
}
