// RUN: triton-opt %s --allocate-shared-memory-nv | FileCheck %s

// The CGA layout of #src splits dim 0 across the two CTAs, but dim 0 has size
// 1, so both CTAs hold the whole tensor. The scratch buffer must hold all 1024
// elements of a CTA, not half of them.
#src = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#dst = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1]]}>

// CHECK: ttg.shared = 4096 : i32
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @convert_layout_cta_broadcast_src
tt.func @convert_layout_cta_broadcast_src(%arg0: tensor<1x1024xi32, #src>, %ptr: tensor<1x1024x!tt.ptr<i32>, #dst>) {
  // CHECK: ttg.convert_layout %arg0 {allocation.offset = 0 : i32, allocation.size = 4096 : i32}
  %0 = ttg.convert_layout %arg0 : tensor<1x1024xi32, #src> -> tensor<1x1024xi32, #dst>
  tt.store %ptr, %0 : tensor<1x1024x!tt.ptr<i32>, #dst>
  tt.return
}
}
