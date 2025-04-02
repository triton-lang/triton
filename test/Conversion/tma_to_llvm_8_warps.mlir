// RUN: triton-opt %s --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | not FileCheck %s

#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem_layout = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

#blocked_indices_8 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked_indices_4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @tma_gather_large_tile_complex
tt.func @tma_gather_large_tile_complex(
  %arg0: !tt.ptr<i8>,
  %arg1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>,
  %arg2: tensor<512xi32, #blocked_indices_8>,
  %arg3: i32,
  %arg4: !ttg.memdesc<512x512xf32, #smem_layout, #smem, mutable>,
  %arg5: i1
) {
  // CHECK-COUNT-256: cp.async.bulk.tensor
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 :
    !tt.ptr<i8>,
    tensor<512xi32, #blocked_indices_8>,
    i32,
    !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>,
    !ttg.memdesc<512x512xf32, #smem_layout, #smem, mutable>, i1
  // CHECK-NEXT: ret void
  tt.return
}

tt.func @tma_gather_small_tile_simple(
  %arg0: !tt.ptr<i8>,
  %arg1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>,
  %arg2: tensor<64xi32, #blocked_indices_4>,
  %arg3: i32,
  %arg4: !ttg.memdesc<64x64xf32, #smem_layout, #smem, mutable>,
  %arg5: i1
) {
  // CHECK: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANE_ID:%.*]] = and i32 [[TID]], 31
  // CHECK: [[WARP_ID:%.*]] = tail call i32 @llvm.nvvm.shfl.sync

  // CHECK: [[IDX0:%.*]] = extractvalue {{.*}} 0
  // CHECK: [[IDX1:%.*]] = extractvalue {{.*}} 1
  // CHECK: [[IDX2:%.*]] = extractvalue {{.*}} 2
  // CHECK: [[IDX3:%.*]] = extractvalue {{.*}} 3

  // CHECK: [[ACTIVE_LANE0:%.*]] = and i32 [[WARP_ID]], 7
  // CHECK: [[WARP_OFFSET:%.*]] = shl i32 [[WARP_ID]], 7
  // CHECK: [[CLAMP_WARP_OFFSET:%.*]] = and i32 [[WARP_OFFSET]], 896
  // CHECK: [[PRED0_LANE:%.*]] = icmp eq i32 [[LANE_ID]], [[ACTIVE_LANE0]]
  // CHECK: [[PRED0:%.*]] = and i1 %5, [[PRED0_LANE]]
  // CHECK: [[OFFSET_i64:%.*]] = zext nneg i32 [[CLAMP_WARP_OFFSET]]
  // CHECK: [[PTR:%.*]] = getelementptr {{.*}} [[OFFSET_i64]]
  // CHECK-NEXT: cp.async.bulk.tensor
  // CHECK-SAME: (i1 [[PRED0]], ptr addrspace(3) [[PTR]], ptr addrspace(1) %0, i32 %3, i32 [[IDX0]], i32 [[IDX1]], i32 [[IDX2]], i32 [[IDX3]]

  // CHECK: [[ACTIVE_LANE1:%.*]] = or disjoint i32 [[ACTIVE_LANE0]], 8
  // CHECK: [[OFFSET:%.*]] = or disjoint i32 [[CLAMP_WARP_OFFSET]], 1024
  // CHECK: [[PRED1_LANE:%.*]] = icmp eq i32 [[LANE_ID]], [[ACTIVE_LANE1]]
  // CHECK: [[PRED1:%.*]] = and i1 %5, [[PRED1_LANE]]
  // CHECK: [[OFFSET_i64:%.*]] = zext nneg i32 [[OFFSET]]
  // CHECK: [[PTR:%.*]] = getelementptr {{.*}} [[OFFSET_i64]]
  // CHECK-NEXT: cp.async.bulk.tensor
  // CHECK-SAME: (i1 [[PRED1]], ptr addrspace(3) [[PTR]], ptr addrspace(1) %0, i32 %3, i32 [[IDX0]], i32 [[IDX1]], i32 [[IDX2]], i32 [[IDX3]]

  // CHECK: [[OFFSET:%.*]] = or disjoint i32 [[CLAMP_WARP_OFFSET]], 2048
  // CHECK: [[Y1:%.*]] = add i32 %3, 32
  // CHECK: [[OFFSET_i64:%.*]] = zext nneg i32 [[OFFSET]]
  // CHECK: [[PTR:%.*]] = getelementptr {{.*}} [[OFFSET_i64]]
  // CHECK-NEXT: cp.async.bulk.tensor
  // CHECK-SAME: (i1 [[PRED0]], ptr addrspace(3) [[PTR]], ptr addrspace(1) %0, i32 [[Y1]], i32 [[IDX0]], i32 [[IDX1]], i32 [[IDX2]], i32 [[IDX3]]

  // CHECK: [[OFFSET:%.*]] = or disjoint i32 [[CLAMP_WARP_OFFSET]], 3072
  // CHECK: [[OFFSET_i64:%.*]] = zext nneg i32 [[OFFSET]]
  // CHECK: [[PTR:%.*]] = getelementptr {{.*}} [[OFFSET_i64]]
  // CHECK-NEXT: cp.async.bulk.tensor
  // CHECK-SAME: (i1 [[PRED1]], ptr addrspace(3) [[PTR]], ptr addrspace(1) %0, i32 [[Y1]], i32 [[IDX0]], i32 [[IDX1]], i32 [[IDX2]], i32 [[IDX3]]
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 :
    !tt.ptr<i8>,
    tensor<64xi32, #blocked_indices_4>,
    i32,
    !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>,
    !ttg.memdesc<64x64xf32, #smem_layout, #smem, mutable>, i1

  // CHECK-NEXT: ret void
  tt.return
}

// CHECK-LABEL: @tma_gather_tiny_tile
tt.func @tma_gather_tiny_tile(
  %arg0: !tt.ptr<i8>,
  %arg1: !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>,
  %arg2: tensor<8xi32, #blocked_indices_4>,
  %arg3: i32,
  %arg4: !ttg.memdesc<8x32xf32, #smem_layout, #smem, mutable>,
  %arg5: i1
) {
  // CHECK: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANE_ID:%.*]] = and i32 [[TID]], 31
  // CHECK: [[WARP_ID:%.*]] = tail call i32 @llvm.nvvm.shfl.sync

  // Only warps 0 and 1 are active.

  // CHECK: [[WARP_SELECT:%.*]] = and i32 [[WARP_ID]], 6
  // CHECK: [[WARP_PRED:%.*]] = icmp eq i32 [[WARP_SELECT]], 0
  // CHECK: [[PRED:%.*]] = and i1 %5, [[WARP_PRED]]

  // CHECK: [[IDX0:%.*]] = extractvalue {{.*}} 0
  // CHECK: [[IDX1:%.*]] = extractvalue {{.*}} 1
  // CHECK: [[IDX2:%.*]] = extractvalue {{.*}} 2
  // CHECK: [[IDX3:%.*]] = extractvalue {{.*}} 3

  // offset = (warpId & 1) * 128

  // CHECK: [[WID_LSB:%.*]] = and i32 [[WARP_ID]], 1
  // CHECK: [[OFFSET:%.*]] = shl nuw nsw i32 [[WID_LSB]], 7

  // Lane 0 in warp 0 and lane 1 in warp 1 are active.

  // CHECK: [[LANE_PRED:%.*]] = icmp eq i32 [[LANE_ID]], [[WID_LSB]]
  // CHECK: [[CUR_PRED:%.*]] = and i1 [[LANE_PRED]], [[PRED]]

  // CHECK: [[OFFSET_i64:%.*]] = zext nneg i32 [[OFFSET]]
  // CHECK: [[PTR:%.*]] = getelementptr {{.*}} [[OFFSET_i64]]
  // CHECK: (i1 [[CUR_PRED]], ptr addrspace(3) [[PTR]], ptr addrspace(1) %0, i32 %3, i32 [[IDX0]], i32 [[IDX1]], i32 [[IDX2]], i32 [[IDX3]]
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 :
    !tt.ptr<i8>,
    tensor<8xi32, #blocked_indices_4>,
    i32,
    !ttg.memdesc<1xi64, #bar_layout, #smem, mutable>,
    !ttg.memdesc<8x32xf32, #smem_layout, #smem, mutable>, i1
  // CHECK-NEXT: ret void
  tt.return
}

}
