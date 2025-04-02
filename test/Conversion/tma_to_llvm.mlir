// RUN: triton-opt %s --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1], [2], [16], [0]], lane = [[0], [0], [0], [0], [0]], warp = [[4], [8]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

#linear0 = #ttg.linear<{register = [[1], [2]], lane = [[4], [8], [0], [0], [0]], warp = [[16], [0]], block = []}>
#linear1 = #ttg.linear<{register = [[1], [2], [16]], lane = [[0], [8], [0], [32], [0]], warp = [[0], [4]], block = []}>
#blocked_4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @tma_gather_simple
// CHECK-SAME: i32 [[Y0:%3]]
tt.func @tma_gather_simple(%arg0: !tt.ptr<i8>, %arg1: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %arg2: tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, %arg3: i32, %arg4: !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, %arg5: i1) {
  // There are 32 indices distributed to 4 warps, so each warp as 8 indices.

  // CHECK: [[BAR:%.*]] = extractvalue {{.*}} %1, 0
  // CHECK: [[BASE_PTR:%.*]] = extractvalue {{.*}} %4, 0

  // CHECK: [[TIDX:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANE_ID:%.*]] = and i32 [[TIDX]], 31
  // CHECK: [[WIDX:%.*]] = lshr i32 [[TIDX]], 5
  // CHECK: [[WARP_ID:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[WIDX]],

  // CHECK: [[IDX0:%.*]] = extractvalue {{.*}} %2, 0
  // CHECK: [[IDX1:%.*]] = extractvalue {{.*}} %2, 1
  // CHECK: [[IDX2:%.*]] = extractvalue {{.*}} %2, 2
  // CHECK: [[IDX3:%.*]] = extractvalue {{.*}} %2, 3

  // CHECK: [[IDX4:%.*]] = extractvalue {{.*}} %2, 4
  // CHECK: [[IDX5:%.*]] = extractvalue {{.*}} %2, 5
  // CHECK: [[IDX6:%.*]] = extractvalue {{.*}} %2, 6
  // CHECK: [[IDX7:%.*]] = extractvalue {{.*}} %2, 7

  // There are 32x128 = 4096 elements. Each gather4 will read 4*128/2 = 256
  // elements into smem. We need to issue 16 gather4 messages. Each warp will
  // execute 4 gather4 instructions.
  //
  // The 64-element (128-byte) row segments are organized into shared memory
  // by segments. I.e.
  //
  // [ t[0, 0:128], t[1: 0:128], ..., t[31: 0:128], t[0, 128:256], ..., t[31: 128:256] ].
  //
  // This is captured by the `nvmma_shared` smem layout.
  //
  // Each warp will handle 4 consecutive row segments at a time, or 4*128 bytes
  // per transaction, thus reading:
  //
  // t[warpId, 0:128], t[warpId, 128:256], t[warpId+16, 0:128], t[warpId+16, 128:256]
  //
  // Each group of 4 segments are 4*128/2 = 256 elements apart. So the starting
  // addresses are [x, x+2048, x+1024, x+3072], where `x = warpId*256`.
  //
  // Note that result smem layout has a swizzle tile of [8, 64], and 8 such
  // tiles comprise the result space. That means every other group of 4 row
  // segments land in the middle of a swizzle tile, where the 0th logical column
  // element may not be at the start of the tile.

  // CHECK: [[WARP_STRIDE_TMP:%.*]] = shl i32 [[WARP_ID]], 8
  // CHECK: [[WARP_STRIDE:%.*]] = and i32 [[WARP_STRIDE_TMP]], 768

  // CHECK: [[LANE_PRED:%.*]] = icmp eq i32 [[LANE_ID]], 0
  // CHECK: [[PRED:%.*]] = and i1 [[LANE_PRED]], %5

  // CHECK: [[OFFSET0:%.*]] = zext nneg i32 [[WARP_STRIDE]] to i64
  // CHECK: [[BASEPTR0:%.*]] = getelementptr bfloat, ptr addrspace(3) [[BASE_PTR]], i64 [[OFFSET0]]
  // CHECK: "@$0 cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4, $5, $6, $7}], [$8];", "b,r,l,r,r,r,r,r,r"
  // CHECK-SAME: (i1 [[PRED]], ptr addrspace(3) [[BASEPTR0]], ptr addrspace(1) %0, i32 [[Y0]], i32 [[IDX0]], i32 [[IDX1]], i32 [[IDX2]], i32 [[IDX3]], ptr addrspace(3) [[BAR]])

  // CHECK: [[OFFSET1_TMP:%.*]] = or disjoint i32 [[WARP_STRIDE]], 1024
  // CHECK: [[OFFSET1:%.*]] = zext nneg i32 [[OFFSET1_TMP]] to i64
  // CHECK: [[BASEPTR1:%.*]] = getelementptr bfloat, ptr addrspace(3) [[BASE_PTR]], i64 [[OFFSET1]]
  // CHECK: cp.async.bulk.tensor.2d.tile::gather4
  // CHECK-SAME: (i1 [[PRED]], ptr addrspace(3) [[BASEPTR1]], ptr addrspace(1) %0, i32 [[Y0]], i32 [[IDX4]], i32 [[IDX5]], i32 [[IDX6]], i32 [[IDX7]], ptr addrspace(3) [[BAR]])

  // CHECK: [[OFFSET2_TMP:%.*]] = or disjoint i32 [[WARP_STRIDE]], 2048
  // CHECK: [[Y1:%.*]] = add i32 [[Y0]], 64
  // CHECK: [[OFFSET2:%.*]] = zext nneg i32 [[OFFSET2_TMP]] to i64
  // CHECK: [[BASEPTR2:%.*]] = getelementptr bfloat, ptr addrspace(3) [[BASE_PTR]], i64 [[OFFSET2]]
  // CHECK: cp.async.bulk.tensor.2d.tile::gather4
  // CHECK-SAME: (i1 [[PRED]], ptr addrspace(3) [[BASEPTR2]], ptr addrspace(1) %0, i32 [[Y1]], i32 [[IDX0]], i32 [[IDX1]], i32 [[IDX2]], i32 [[IDX3]], ptr addrspace(3) [[BAR]])

  // CHECK: [[OFFSET3_TMP:%.*]] = or disjoint i32 [[WARP_STRIDE]], 3072
  // CHECK: [[OFFSET3:%.*]] = zext nneg i32 [[OFFSET3_TMP]] to i64
  // CHECK: [[BASEPTR3:%.*]] = getelementptr bfloat, ptr addrspace(3) [[BASE_PTR]], i64 [[OFFSET3]]
  // CHECK: cp.async.bulk.tensor.2d.tile::gather4
  // CHECK-SAME: (i1 [[PRED]], ptr addrspace(3) [[BASEPTR3]], ptr addrspace(1) %0, i32 [[Y1]], i32 [[IDX4]], i32 [[IDX5]], i32 [[IDX6]], i32 [[IDX7]], ptr addrspace(3) [[BAR]])
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 : !tt.ptr<i8>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, i1

  // CHECK-NEXT: ret void
  tt.return
}

// CHECK-LABEL: @tma_gather_8_consecutive_indices
tt.func @tma_gather_8_consecutive_indices(%arg0: !tt.ptr<i8>, %arg1: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %arg2: tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, %arg3: i32, %arg4: !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, %arg5: i1) {
  // Due to the `sizePerThread = [1, 8]`, each warp now handles 8 consecutive
  // rows, where each row is divided into 2 segments for a total of 4 gather4s.
  //
  // t[warpId, 0:128], t[warpId, 128:256], t[warpId+4, 0:128], t[warpId+4, 128:256]
  //
  // So the base addresses are [x, x+2048, x+256, x+2048+256], where `x = warpId*256`.

  // CHECK: [[WARP_ID:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32
  // CHECK: [[WARP_STRIDE_TMP:%.*]] = shl i32 [[WARP_ID]], 9
  // CHECK: [[OFFSET0:%.*]] = and i32 [[WARP_STRIDE_TMP]], 1536

  // CHECK: zext nneg i32 [[OFFSET0]] to i64
  // CHECK: cp.async.bulk.tensor

  // CHECK: [[OFFSET1:%.*]] = or disjoint i32 [[OFFSET0]], 256
  // CHECK: zext nneg i32 [[OFFSET1]] to i64
  // CHECK: cp.async.bulk.tensor

  // CHECK: [[OFFSET2:%.*]] = or disjoint i32 [[OFFSET0]], 2048
  // CHECK: zext nneg i32 [[OFFSET2]] to i64
  // CHECK: cp.async.bulk.tensor

  // CHECK: [[OFFSET3:%.*]] = or disjoint i32 [[OFFSET0]], 2304
  // CHECK: zext nneg i32 [[OFFSET3]] to i64
  // CHECK: cp.async.bulk.tensor
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 : !tt.ptr<i8>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, i32, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, i1

  // CHECK-NEXT: ret void
  tt.return
}

// CHECK-LABEL: @tma_gather_redundant_indices
tt.func @tma_gather_redundant_indices(%arg0: !tt.ptr<i8>, %arg1: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %arg2: tensor<32xi32, #linear>, %arg3: i32, %arg4: !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, %arg5: i1) {
  // Codegen for this case is actually incorrect due to linear layouts
  // incorrectly handling register broadcasting, but the test outcome is nonetheless
  // the same.

  // CHECK-COUNT-4: cp.async.bulk.tensor
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 : !tt.ptr<i8>, tensor<32xi32, #linear>, i32, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, i1
  // CHECK-NEXT: ret void
  tt.return
}

// CHECK-LABEL: @tma_gather_redundant_warps
tt.func @tma_gather_redundant_warps(%arg0: !tt.ptr<i8>, %arg1: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %arg2: tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>, %arg3: i32, %arg4: !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, %arg5: i1) {
  // CHECK: [[TIDX:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANE_ID:%.*]] = and i32 [[TIDX]], 31
  // CHECK: [[LANE_PRED:%.*]] = icmp eq i32 [[LANE_ID]], 0
  // CHECK: [[PRED:%.*]] = and i1 [[LANE_PRED]], %5

  // CHECK-COUNT-4: cp.async.bulk.tensor{{.*}}(i1 [[PRED]],
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 : !tt.ptr<i8>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>, i32, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>, i1

  // CHECK-NEXT: ret void
  tt.return
}

// CHECK-LABEL: @tma_scatter
tt.func @tma_scatter(%arg0: !tt.ptr<i8>, %arg1: tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, %arg2: i32, %arg3: !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>) {
  // The lowering for `async_tma_scatter` shares practically all of its logic
  // with `async_tma_gather`, so we don't need to re-test the indexing logic.

  // CHECK: [[BASE_PTR:%.*]] = extractvalue {{.*}} %3, 0

  // CHECK: [[TIDX:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANE_ID:%.*]] = and i32 [[TIDX]], 31
  // CHECK: [[PRED:%.*]] = icmp eq i32 [[LANE_ID]], 0

  // CHECK: [[PTR:%.*]] = getelementptr {{.*}} [[BASE_PTR]]
  // CHECK-NEXT: "@$0 cp.async.bulk.tensor.2d.tile::scatter4.global.shared::cta.bulk_group [$1, {$2, $3, $4, $5, $6}], [$7];"
  // CHECK-SAME: (i1 [[PRED]], ptr addrspace(1) %0, i32 %2, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, ptr addrspace(3) [[PTR]])
  ttng.async_tma_scatter %arg0[%arg1, %arg2] %arg3 : !tt.ptr<i8>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<32x128xbf16, #shared1, #smem, mutable>

  // CHECK: nvvm.cp.async.bulk.commit.group()

  // CHECK-NEXT: ret void
  tt.return
}

// CHECK-LABEL: @tma_gather_linear_warp_base
tt.func @tma_gather_linear_warp_base(
  %arg0: !tt.ptr<i8>,
  %arg1: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
  %arg2: tensor<32xi32, #linear0>,
  %arg3: i32,
  %arg4: !ttg.memdesc<32x32xf32, #shared2, #smem, mutable>,
  %arg5: i1
) {
  // tileX warpId laneId
  // 0     0      0
  // 1     2      1
  // 2     0      2
  // 3     2      3
  // 4     1      0
  // 5     3      1
  // 6     1      2
  // 7     3      3

  // CHECK: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANE_ID:%.*]] = and i32 [[TID]], 31
  // CHECK: [[WARP_ID:%.*]] = tail call i32 @llvm.nvvm.shfl.sync

  // CHECK: [[IDX0:%.*]] = extractvalue {{.*}} 0
  // CHECK: [[IDX1:%.*]] = extractvalue {{.*}} 1
  // CHECK: [[IDX2:%.*]] = extractvalue {{.*}} 2
  // CHECK: [[IDX3:%.*]] = extractvalue {{.*}} 3

  // CHECK: [[WARP_B2:%.*]] = and i32 [[WARP_ID]], 2
  // CHECK: [[ACTIVE_LANE0:%.*]] = lshr exact i32 [[WARP_B2]], 1

  // CHECK: [[ADDR_C0:%.*]] = shl nuw nsw i32 [[WARP_B2]], 6
  // CHECK: [[ADDR_C1:%.*]] = shl i32 [[WARP_ID]], 9
  // CHECK: [[ADDR_C11:%.*]] = and i32 [[ADDR_C1]], 512
  // CHECK: [[ADDR0:%.*]] = or disjoint i32 [[ADDR_C0]], [[ADDR_C11]]

  // CHECK: [[LANE_PRED0:%.*]] = icmp eq i32 [[LANE_ID]], [[ACTIVE_LANE0]]
  // CHECK: [[PRED0:%.*]] = and i1 %5, [[LANE_PRED0]]

  // CHECK: [[OFFSET0:%.*]] = zext nneg i32 [[ADDR0]] to i64
  // CHECK: [[BASEPTR0:%.*]] = getelementptr {{.*}} [[OFFSET0]]
  // CHECK: cp.async.bulk.tensor
  // CHECK-SAME: (i1 [[PRED0]], ptr addrspace(3) [[BASEPTR0]], ptr addrspace(1) %0, i32 %3, i32 [[IDX0]], i32 [[IDX1]], i32 [[IDX2]], i32 [[IDX3]]

  // CHECK: [[ACTIVE_LANE1:%.*]] = or disjoint i32 [[ACTIVE_LANE0]], 2
  // CHECK: [[ADDR1:%.*]] = or disjoint i32 [[ADDR0]], 256
  // CHECK: [[LANE_PRED1:%.*]] = icmp eq i32 [[LANE_ID]], [[ACTIVE_LANE1]]
  // CHECK: [[PRED1:%.*]] = and i1 %5, [[LANE_PRED1]]
  // CHECK: [[OFFSET1:%.*]] = zext nneg i32 [[ADDR1]] to i64
  // CHECK: [[BASEPTR1:%.*]] = getelementptr {{.*}} [[OFFSET1]]
  // CHECK: cp.async.bulk.tensor
  // CHECK-SAME: (i1 [[PRED1]], ptr addrspace(3) [[BASEPTR1]], ptr addrspace(1) %0, i32 %3, i32 [[IDX0]], i32 [[IDX1]], i32 [[IDX2]], i32 [[IDX3]]
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 :
    !tt.ptr<i8>,
    tensor<32xi32, #linear0>,
    i32,
    !ttg.memdesc<1xi64, #shared, #smem, mutable>,
    !ttg.memdesc<32x32xf32, #shared2, #smem, mutable>, i1
  // CHECK-NEXT: ret void
  tt.return
}

// CHECK-LABEL: @tma_gather_strange_layout
tt.func @tma_gather_strange_layout(
  %arg0: !tt.ptr<i8>,
  %arg1: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
  %arg2: tensor<32xi32, #linear1>,
  %arg3: i32,
  %arg4: !ttg.memdesc<32x32xf32, #shared2, #smem, mutable>,
  %arg5: i1
) {
  // CHECK: llvm.nvvm.shfl.sync
  // CHECK: [[IDX0:%.*]] = extractvalue {{.*}} 0
  // CHECK: [[IDX1:%.*]] = extractvalue {{.*}} 1
  // CHECK: [[IDX2:%.*]] = extractvalue {{.*}} 2
  // CHECK: [[IDX3:%.*]] = extractvalue {{.*}} 3
  // CHECK: [[IDX4:%.*]] = extractvalue {{.*}} 4
  // CHECK: [[IDX5:%.*]] = extractvalue {{.*}} 5
  // CHECK: [[IDX6:%.*]] = extractvalue {{.*}} 6
  // CHECK: [[IDX7:%.*]] = extractvalue {{.*}} 7

  // CHECK: cp.async.bulk.tensor
  // CHECK-SAME: i32 [[IDX0]], i32 [[IDX1]], i32 [[IDX2]], i32 [[IDX3]]

  // CHECK: cp.async.bulk.tensor
  // CHECK-SAME: i32 [[IDX4]], i32 [[IDX5]], i32 [[IDX6]], i32 [[IDX7]]
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 :
    !tt.ptr<i8>,
    tensor<32xi32, #linear1>,
    i32,
    !ttg.memdesc<1xi64, #shared, #smem, mutable>,
    !ttg.memdesc<32x32xf32, #shared2, #smem, mutable>, i1
  // CHECK-NEXT: ret void
  tt.return
}

// CHECK-LABEL: @tma_gather_tile_y_iter
tt.func @tma_gather_tile_y_iter(
  %arg0: !tt.ptr<i8>,
  %arg1: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
  %arg2: tensor<8xi32, #blocked_4>,
  %arg3: i32,
  %arg4: !ttg.memdesc<8x64xf32, #shared2, #smem, mutable>,
  %arg5: i1
) {
  // y = (warpId&2) * 16

  // CHECK: [[WARP_ID:%.*]] = tail call i32 @llvm.nvvm.shfl.sync
  // CHECK: [[Y_OFFSET:%.*]] = shl i32 [[WARP_ID]], 4
  // CHECK: [[Y_CLAMP:%.*]] = and i32 [[Y_OFFSET]], 32
  // CHECK: [[Y:%.*]] = add i32 [[Y_CLAMP]], %3

  // CHECK: cp.async.bulk.tensor
  // CHECK-SAME: i32 [[Y]]
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 :
    !tt.ptr<i8>,
    tensor<8xi32, #blocked_4>,
    i32,
    !ttg.memdesc<1xi64, #shared, #smem, mutable>,
    !ttg.memdesc<8x64xf32, #shared2, #smem, mutable>, i1
  // CHECK-NEXT: ret void
  tt.return
}

}
