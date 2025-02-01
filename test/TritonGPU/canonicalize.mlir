// RUN: triton-opt %s -split-input-file -canonicalize | FileCheck %s


// CHECK-LABEL: @test_canonicalize_convert_view
// CHECK-SAME: (%[[ARG:.+]]: tensor<64x64xf32
//   CHECK-NOT:   ttg.convert_layout
//       CHECK:   %[[V:.+]] = tt.reshape %[[ARG]] allow_reorder
//       CHECK:   tt.return %[[V]]
#blocked0 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
tt.func @test_canonicalize_convert_view(%arg0: tensor<64x64xf32, #blocked0>) -> tensor<4096xf32, #blocked1> {
    %c = ttg.convert_layout %arg0 : tensor<64x64xf32, #blocked0> -> tensor<64x64xf32, #blocked2>
    %r = tt.reshape %c allow_reorder : tensor<64x64xf32, #blocked2> -> tensor<4096xf32, #blocked1>
    tt.return %r : tensor<4096xf32, #blocked1>
}
}  // end module

// -----

// test that the convert doesn't get combined with view if the resulting operations
// is an expensive view which would require moving data across threads.
// CHECK-LABEL: @test_canonicalize_convert_expensive_view
// CHECK-SAME: (%[[ARG:.+]]: tensor<256x16xf32
//       CHECK:   %[[C:.+]] = ttg.convert_layout %[[ARG]]
//       CHECK:   %[[V:.+]] = tt.reshape %[[C]] allow_reorder
//       CHECK:   tt.return %[[V]]
#blocked0 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
tt.func @test_canonicalize_convert_expensive_view(%arg0: tensor<256x16xf32, #blocked0>) -> tensor<4096xf32, #blocked1> {
    %c = ttg.convert_layout %arg0 : tensor<256x16xf32, #blocked0> -> tensor<256x16xf32, #blocked2>
    %r = tt.reshape %c allow_reorder : tensor<256x16xf32, #blocked2> -> tensor<4096xf32, #blocked1>
    tt.return %r : tensor<4096xf32, #blocked1>
}
}  // end module

// -----

// test that the convert does get combined with the view even if the resulting operation
// is an efficient view.
// CHECK-LABEL: @test_canonicalize_convert_view
// CHECK-SAME: (%[[ARG:.+]]: tensor<64x64xf32
//   CHECK-NOT:   ttg.convert_layout
//       CHECK:   %[[V:.+]] = tt.reshape %[[ARG]] allow_reorder
//       CHECK:   tt.return %[[V]]
#blocked0 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
tt.func @test_canonicalize_convert_view(%arg0: tensor<64x64xf32, #blocked0>) -> tensor<4096xf32, #blocked1> {
    %c = ttg.convert_layout %arg0 : tensor<64x64xf32, #blocked0> -> tensor<64x64xf32, #blocked2>
    %r = tt.reshape %c allow_reorder efficient_layout : tensor<64x64xf32, #blocked2> -> tensor<4096xf32, #blocked1>
    tt.return %r : tensor<4096xf32, #blocked1>
}
}  // end module

// -----

// CHECK-LABEL: @test_canonicalize_convert_histogram
// CHECK-SAME: (%[[ARG:.+]]: tensor<256xi32
//   CHECK-NOT:   ttg.convert_layout
//       CHECK:   %[[V:.+]] = tt.histogram %[[ARG]]
//   CHECK-NOT:   ttg.convert_layout
//       CHECK:   tt.return %[[V]]
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
tt.func @test_canonicalize_convert_histogram(%arg0: tensor<256xi32, #blocked1>) -> tensor<512xi32, #blocked2> {
    %0 = ttg.convert_layout %arg0 : tensor<256xi32, #blocked1> -> tensor<256xi32, #blocked>
    %1 = tt.histogram %0 : tensor<256xi32, #blocked> -> tensor<512xi32, #blocked>
    %2 = ttg.convert_layout %1 : tensor<512xi32, #blocked> -> tensor<512xi32, #blocked2>
    tt.return %2 : tensor<512xi32, #blocked2>
}
}  // end module

// -----

// CHECK-LABEL: @test_canonicalize_convert_local_load
// CHECK-NOT:   gpu.barrier
// CHECK: %[[V:.+]] = ttg.local_load
// CHECK-NEXT:  gpu.barrier
// CHECK-NEXT: tt.return %[[V]]

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.compute-capability" = 80} {
tt.func @test_canonicalize_convert_local_load() -> tensor<256xi32, #blocked1> {
    %0 = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> tensor<256xi32, #blocked>
    gpu.barrier
    %2 = ttg.convert_layout %1 : tensor<256xi32, #blocked> -> tensor<256xi32, #blocked1>
    tt.return %2 : tensor<256xi32, #blocked1>
}
}  // end module

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase=2, maxPhase=8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: local_alloc_nofold1
  tt.func @local_alloc_nofold1(%arg0: tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem> {
    // CHECK: %[[ARG:.+]] = ttg.local_alloc
    // CHECK-NEXT: %[[ARG2:.+]] = ttg.local_load %[[ARG]]
    // CHECK-NEXT: %[[ARG3:.+]] = ttg.local_alloc %[[ARG2]]
    // CHECK-NEXT: tt.return %[[ARG3]]
    %0 = ttg.local_alloc %arg0 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #shared, #smem, mutable> -> tensor<16x16xf16, #blocked>
    %2 = ttg.local_alloc %1 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem>
    tt.return %2 : !ttg.memdesc<16x16xf16, #shared, #smem>
  }
}  // end module


// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase=2, maxPhase=8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase=1, maxPhase=1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: local_alloc_nofold2
  tt.func @local_alloc_nofold2(%arg0: tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared1, #smem> {
    // CHECK: %[[ARG:.+]] = ttg.local_alloc
    // CHECK-NEXT: %[[ARG2:.+]] = ttg.local_load %[[ARG]]
    // CHECK-NEXT: %[[ARG3:.+]] = ttg.local_alloc %[[ARG2]]
    // CHECK-NEXT: tt.return %[[ARG3]]
    %0 = ttg.local_alloc %arg0 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem>
    %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #shared, #smem> -> tensor<16x16xf16, #blocked>
    %2 = ttg.local_alloc %1 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared1, #smem>
    tt.return %2 : !ttg.memdesc<16x16xf16, #shared1, #smem>
  }
}  // end module


// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase=2, maxPhase=8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  tt.func @local_alloc_fold(%arg0: tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem> {
    // CHECK-LABEL: local_alloc_fold
    // CHECK-NEXT: %[[ARG:.+]] = ttg.local_alloc
    // CHECK-NEXT: tt.return %[[ARG]]
    %0 = ttg.local_alloc %arg0 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem>
    %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #shared, #smem> -> tensor<16x16xf16, #blocked>
    %2 = ttg.local_alloc %1 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem>
    tt.return %2 : !ttg.memdesc<16x16xf16, #shared, #smem>
  }
}  // end module

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_layout_gather_src
  tt.func @convert_layout_gather_src(%arg0: tensor<16x16xf16, #blocked>, %arg1: tensor<16x16xi32, #blocked>) -> tensor<16x16xf16, #blocked> {
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #blocked1>
    // CHECK-NEXT: tt.gather %arg0[%arg1]
    %1 = tt.gather %0[%arg1] {axis = 0 : i32} : (tensor<16x16xf16, #blocked1>, tensor<16x16xi32, #blocked>) -> tensor<16x16xf16, #blocked>
    tt.return %1 : tensor<16x16xf16, #blocked>
  }

  // CHECK-LABEL: gather_efficient_layout
  tt.func @gather_efficient_layout(%arg0: tensor<16x16xf16, #blocked>, %arg1: tensor<16x16xi32, #blocked>) -> tensor<16x16xf16, #blocked> {
    // CHECK-NEXT: convert_layout
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #blocked1>
    // CHECK-NEXT: tt.gather {{.*}} (tensor<16x16xf16, #blocked1>
    %1 = tt.gather %0[%arg1] {axis = 0 : i32, efficient_layout} : (tensor<16x16xf16, #blocked1>, tensor<16x16xi32, #blocked>) -> tensor<16x16xf16, #blocked>
    tt.return %1 : tensor<16x16xf16, #blocked>
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [8, 0], [16, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[0, 8], [0, 16]], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_trans = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @infer_trans
tt.func @infer_trans(%arg0: tensor<32x32xf32, #linear>) -> tensor<32x32xf32, #blocked_trans> {
  // CHECK-NOT: ttg.convert_layout
  %0 = ttg.convert_layout %arg0 : tensor<32x32xf32, #linear> -> tensor<32x32xf32, #blocked>
  %1 = tt.trans %0  {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_trans>
  tt.return %1 : tensor<32x32xf32, #blocked_trans>
}

}
