// RUN: triton-opt -split-input-file %s --convert-triton-amdgpu-to-llvm='arch=gfx942' -verify-diagnostics

// Invalid size
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_size_input(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{sizes [256, 2] must be a multiple of shapePerCTA [256, 16]}}
  %1 = amdgpu.view_slice %arg0[0,0] [256, 2] [1,1] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
  tt.return
}

// -----

// Invalid offset
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_offset_input(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{offset [0, 5] must be a multiple of shapePerCTA [256, 16]}}
  %1 = amdgpu.view_slice %arg0[0,5] [256, 16] [1,1] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
  tt.return
}

// -----

// Invalid result layout
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_layout(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result layout must match source layout}}
  %1 = amdgpu.view_slice %arg0[0,0] [256, 16] [1,1] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked2>
  tt.return
}

// -----

// Invalid result element type
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_element_type(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result element type must match source element type}}
  %1 = amdgpu.view_slice %arg0[0,0] [256, 16] [1,1] : tensor<256x128xi32, #blocked1> to tensor<256x16xi64, #blocked1>
  tt.return
}

// -----

// Invalid result rank
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_result_rank(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{result rank must be equal to source rank}}
  %1 = amdgpu.view_slice %arg0[0,0] [256, 16] [1,1] : tensor<256x128xi32, #blocked1> to tensor<256x16x2xi32, #blocked1>
  tt.return
}

// -----

// Invalid rank
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_rank(%arg0: tensor<256x128x2xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{currently only 2D tensors are supported}}
  %1 = amdgpu.view_slice %arg0[0,0,0] [256,16,2] [1,1,1] : tensor<256x128x2xi32, #blocked1> to tensor<256x16x2xi32, #blocked1>
  tt.return
}

// -----

// Invalid stride
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
tt.func @invalid_stride(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
  // expected-error @+1 {{expected unit strides but found unsupported stride [1, 2]}}
  %1 = amdgpu.view_slice %arg0[0,0] [256, 16] [1,2] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
  tt.return
}
