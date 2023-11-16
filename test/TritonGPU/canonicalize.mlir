// RUN: triton-opt %s -split-input-file -canonicalize | FileCheck %s


// CHECK-LABEL: @test_canonicalize_convert_view
// CHECK-SAME: (%[[ARG:.+]]: tensor<64x64xf32
//   CHECK-NOT:   triton_gpu.convert_layout
//       CHECK:   %[[V:.+]] = tt.view %[[ARG]]
//       CHECK:   tt.return %[[V]]
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @test_canonicalize_convert_view(%arg0: tensor<64x64xf32, #blocked0>) -> tensor<4096xf32, #blocked1> {
    %c = triton_gpu.convert_layout %arg0 : (tensor<64x64xf32, #blocked0>) -> tensor<64x64xf32, #blocked2>
    %r = tt.view %c : (tensor<64x64xf32, #blocked2>) -> tensor<4096xf32, #blocked1>
    tt.return %r : tensor<4096xf32, #blocked1>
}

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
tt.func @test_canonicalize_convert_expensive_view(%arg0: tensor<256x16xf32, #blocked0>) -> tensor<4096xf32, #blocked1> {
    %c = triton_gpu.convert_layout %arg0 : (tensor<256x16xf32, #blocked0>) -> tensor<256x16xf32, #blocked2>
    %r = tt.view %c : (tensor<256x16xf32, #blocked2>) -> tensor<4096xf32, #blocked1>
    tt.return %r : tensor<4096xf32, #blocked1>
}


// -----

// Test that the convert doesn't get combined with view if the either the
// operand or result has a dot operand ecnoding. Part of values with dot operand
// encoding will be packed/unpacked as i32 elements when lowering to LLVM. To
// avoid errors, skip this folding when either the operand or result of view has
// a dot operand encoding.
// CHECK-LABEL: @test_canonicalize_convert_view_with_dot_operand_encoding
// CHECK-SAME: (%[[ARG:.+]]: tensor<32x4x32xbf16
//       CHECK:   %[[V:.+]] = tt.view %[[ARG]]
//       CHECK:   %[[C:.+]] = triton_gpu.convert_layout %[[V]]
//       CHECK:   tt.return %[[C]]
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 1, 4], order = [0, 1, 2], CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [0, 1, 2]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
tt.func @test_canonicalize_convert_view_with_dot_operand_encoding(%arg0: tensor<32x4x32xbf16, #blocked3>) -> tensor<32x128xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked5}>> {
    %v = tt.view %arg0 : (tensor<32x4x32xbf16, #blocked3>) -> tensor<32x128xbf16, #blocked4>
    %c0 = triton_gpu.convert_layout %v : (tensor<32x128xbf16, #blocked4>) -> tensor<32x128xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked5}>>
    tt.return %c0 : tensor<32x128xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked5}>>
}
