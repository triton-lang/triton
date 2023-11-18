// RUN: triton-opt %s -split-input-file -canonicalize | FileCheck %s


// CHECK-LABEL: @test_canonicalize_convert_view
// CHECK-SAME: (%[[ARG:.+]]: tensor<64x64xf32
//   CHECK-NOT:   triton_gpu.convert_layout
//       CHECK:   %[[V:.+]] = tt.view %[[ARG]]
//       CHECK:   tt.return %[[V]]
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>

module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.compute-capability" = 80} {
tt.func @test_canonicalize_convert_view(%arg0: tensor<64x64xf32, #blocked0>) -> tensor<4096xf32, #blocked1> {
    %c = triton_gpu.convert_layout %arg0 : (tensor<64x64xf32, #blocked0>) -> tensor<64x64xf32, #blocked2>
    %r = tt.view %c : (tensor<64x64xf32, #blocked2>) -> tensor<4096xf32, #blocked1>
    tt.return %r : tensor<4096xf32, #blocked1>
}
}  // end module

// -----

// test that the convert doesn't get combined with view if the resulting operations
// is an expensive view which would require moving data across threads.
// CHECK-LABEL: @test_canonicalize_convert_expensive_view
// CHECK-SAME: (%[[ARG:.+]]: tensor<256x16xf32
//       CHECK:   %[[C:.+]] = triton_gpu.convert_layout %[[ARG]]
//       CHECK:   %[[V:.+]] = tt.view %[[C]]
//       CHECK:   tt.return %[[V]]
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.compute-capability" = 80} {
tt.func @test_canonicalize_convert_expensive_view(%arg0: tensor<256x16xf32, #blocked0>) -> tensor<4096xf32, #blocked1> {
    %c = triton_gpu.convert_layout %arg0 : (tensor<256x16xf32, #blocked0>) -> tensor<256x16xf32, #blocked2>
    %r = tt.view %c : (tensor<256x16xf32, #blocked2>) -> tensor<4096xf32, #blocked1>
    tt.return %r : tensor<4096xf32, #blocked1>
}
}  // end module
