// RUN: triton-opt %s --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

#blocked_bcast = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 0]]}>
#shared_bar_bcast = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#shared_gather_bcast = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @tma_gather_multicast
tt.func public @tma_gather_multicast(%arg0: !tt.tensordesc<1x128xbf16, #shared_gather_bcast>, %arg1: !ttg.memdesc<1xi64, #shared_bar_bcast, #smem, mutable>, %arg2: tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_bcast}>>, %arg3: i32, %arg4: !ttg.memdesc<32x128xbf16, #shared_gather_bcast, #smem, mutable>, %arg5: i1) {
  // CHECK: [[BAR:%.*]] = extractvalue {{.*}} %1, 0
  // CHECK: [[BAR_INT:%.*]] = ptrtoint ptr addrspace(3) [[BAR]] to i64
  // CHECK: [[LEADER_BAR_INT:%.*]] = and i64 [[BAR_INT]],
  // CHECK: [[LEADER_BAR:%.*]] = inttoptr i64 [[LEADER_BAR_INT]] to ptr addrspace(3)
  // CHECK: [[ELECT:%.*]] = tail call { i32, i1 } @llvm.nvvm.elect.sync
  // CHECK: [[ELECT_PRED:%.*]] = extractvalue { i32, i1 } [[ELECT]], 1
  // CHECK: [[PRED:%.*]] = and i1 {{.*}}, [[ELECT_PRED]]
  // CHECK: "@$0 cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$1], [$2, {$3, $4, $5, $6, $7}], [$8], $9;", "b,r,l,r,r,r,r,r,r,h"
  // CHECK-SAME: (i1 [[PRED]], ptr addrspace(3) {{.*}}, ptr nonnull %0, i32 %3, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, ptr addrspace(3) [[LEADER_BAR]], i32 {{(%[0-9]+|3)}})
  ttng.async_tma_gather %arg0[%arg2, %arg3] %arg4, %arg1, %arg5 {multicast} : !tt.tensordesc<1x128xbf16, #shared_gather_bcast>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_bcast}>>, i32, !ttg.memdesc<1xi64, #shared_bar_bcast, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared_gather_bcast, #smem, mutable>, i1

  // CHECK: ret void
  tt.return
}

}
