// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck %s


#block0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [8], warpsPerCTA = [4], order = [0]}>
#block2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#block3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice2d1 = #triton_gpu.slice<{dim = 1, parent=#block2}>
#slice3d0 = #triton_gpu.slice<{dim = 0, parent=#block3}>
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v1_multictas
  func @basic_insert_slice_async_v1_multictas(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}) {
    %off0_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice2d1>
    %off1_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : (tensor<32xi32, #slice2d1>) -> tensor<32x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : (tensor<32xi32, #slice3d0>) -> tensor<1x32xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : (tensor<32x1xi32, #block2>) -> tensor<32x32xi32, #block2>
    %cst_scalar = arith.constant 32 : i32
    %cst = tt.splat %cst_scalar : (i32) -> tensor<32x32xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<32x32xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : (tensor<1x32xi32, #block3>) -> tensor<32x32xi32, #block3>
    %broadcast_off0 = triton_gpu.convert_layout %broadcast_off0_ : (tensor<32x32xi32, #block2>) -> tensor<32x32xi32, #AL>
    %broadcast_off1 = triton_gpu.convert_layout %broadcast_off1_ : (tensor<32x32xi32, #block3>) -> tensor<32x32xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<32x32xi32, #AL>
    %a_init = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<32x32x!tt.ptr<f32>, #AL>
    %tensor = triton_gpu.alloc_tensor : tensor<2x32x32xf32, #A>
    %index = arith.constant 1 : i32

    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.commit_group
    %a = triton_gpu.insert_slice_async %a_ptr, %tensor, %index {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32x!tt.ptr<f32>, #AL> -> tensor<2x32x32xf32, #A>
    return
  }
}
