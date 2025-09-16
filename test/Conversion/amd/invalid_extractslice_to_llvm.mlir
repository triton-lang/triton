// RUN: triton-opt -split-input-file %s --convert-triton-amdgpu-to-llvm='arch=gfx942' -verify-diagnostics

// Invalid size
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_size_input(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result shape must be multiple of shapePerCTATile}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x2xi32, #blocked1>
  tt.return
}

// -----

// Invalid offset, not multiple of shapePerTile
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_offset_input(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{offset must be multiple of shapePerCTATile}}
  %1 = amdgpu.extract_slice %arg0 [0,5] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
  tt.return
}

// -----

// Invalid offset, out of bounds for dimension
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_offset_input(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{invalid offset at dimension 1}}
  %1 = amdgpu.extract_slice %arg0 [0,128] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
  tt.return
}

// -----

// Invalid result layout
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_layout(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{CTA tile shapes must match between source and destination tensors.}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked2>
  tt.return
}
// -----

// Invalid result element type
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_element_type(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result element type must match source element type}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x16xi64, #blocked1>
  tt.return
}

// -----

// Invalid result rank
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_rank(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result rank must be equal to source rank}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x16x2xi32, #blocked1>
  tt.return
}

// -----

// Invalid result shape
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_rank(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result shape cannot exceed source shape at dimension 1}}
  %1 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x256xi32, #blocked1>
  tt.return
}

// -----

// Invalid non static offset
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_non_static_offset(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}, %arg1: i32) {
  // expected-error @+2 {{expected ']'}}
  // expected-error @+1 {{expected integer value}}
  %2 = amdgpu.extract_slice %arg0 [%arg1, 0] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
  tt.return
}

// -----

// Invalid layout 1
#dst_layout = #ttg.linear<{register=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 64], [64, 0]], lane=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp=[[0, 32], [32, 0]], block=[]}>
#src_layout = #ttg.linear<{register=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 64], [0, 128], [64, 0], [128, 0]], lane=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4], [0, 0]], warp=[[0, 32], [32, 0]], block=[]}>
tt.func @invalid_lane_warp_basis(%arg0: tensor<256x256xi32, #src_layout> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{Lane and warp dim basis must match between source and destination layout}}
  %2 = amdgpu.extract_slice %arg0 [0, 0] : tensor<256x256xi32, #src_layout> to tensor<128x128xi32, #dst_layout>
  tt.return
}

// -----

// Invalid layout 2
// Case when src and dst layouts have same CTA tile shape, but different number of registers
#src_layout = #ttg.linear<{register=[[1, 0], [2, 0]], lane=[[4, 0], [8, 0], [16, 0], [0, 1], [0, 2], [0, 4]], warp=[[0, 0], [0, 8]], block=[]}>
#dst_layout = #ttg.linear<{register=[[1, 0]], lane=[[4, 0], [8, 0], [16, 0], [0, 1], [0, 2], [0, 4]], warp=[[2, 0], [0, 8]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @invalid_concat(%arg0: tensor<64x32xi32, #src_layout>) {
    // expected-error @+1 {{Register basis must match on a CTA tile between source and destination.}}
    %1 = amdgpu.extract_slice %arg0 [0, 0] : tensor<64x32xi32, #src_layout> to tensor<32x16xi32, #dst_layout>
    tt.return
  }
}
