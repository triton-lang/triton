// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx942" | FileCheck %s

#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @extract_2d_blocked_tensor(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: llvm.func @extract_2d_blocked_tensor
    // CHECK-COUNT-64: %{{.*}} = llvm.extractvalue  %{{.*}} : !llvm.struct
    // CHECK-COUNT-8:  %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %72 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
    tt.return
  }
}

// -----

#ll1 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32], [0, 64]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>
#ll2 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>

module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @extract_2d_linear_tensor(%arg0: tensor<256x128xi32, #ll1> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: llvm.func @extract_2d_linear_tensor
    // CHECK-COUNT-64: %{{.*}} = llvm.extractvalue  %arg0[{{[0-9]*}}] : !llvm.struct
    // CHECK-COUNT-8:  %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %72 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #ll1> to tensor<256x16xi32, #ll2>
    tt.return
  }
}

// -----

#ll1 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 0, 16], [0, 0, 32], [0, 0, 64], [1, 0, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 8, 0], [0, 16, 0]], warp = [[0, 32, 0], [0, 64, 0], [0, 128, 0]], block = []}>
#ll2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 0, 16], [0, 0, 32], [0, 0, 64]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 8, 0], [0, 16, 0]], warp = [[0, 32, 0], [0, 64, 0], [0, 128, 0]], block = []}>

module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @extract_3d_linear_tensor(%arg0: tensor<2x256x128xi32, #ll1> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: llvm.func @extract_3d_linear_tensor
    // CHECK-COUNT-128: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-64: %{{[0-9]*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %72 = amdgpu.extract_slice %arg0 [0,0,0] : tensor<2x256x128xi32, #ll1> to tensor<1x256x128xi32, #ll2>
    tt.return
  }
}

// -----

#ll1 = #ttg.linear<{register=[[1], [256], [512]], lane=[[2], [4], [8], [16], [32], [64]], warp=[[128]], block=[]}>
#ll2 = #ttg.linear<{register=[[1]], lane=[[2], [4], [8], [16], [32], [64]], warp=[[128]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @extract_1d_linear_tensor(%arg0: tensor<1024xi32, #ll1> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: llvm.func @extract_1d_linear_tensor
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-2: %{{[0-9]*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %72 = amdgpu.extract_slice %arg0 [0] : tensor<1024xi32, #ll1> to tensor<256xi32, #ll2>
    tt.return
  }
}

// -----

// Input tensor broadcasts 4 registers along dimension 1, resulting in total 32 values in tensor and 16 values per [128x1] tile.
// Output tensor do not have redundancy in register and holds 4 values.
// Test checks that extract slice copies only 4 values from input to output.
#blocked1 = #ttg.linear<{register=[[0, 0], [0, 0], [1, 0], [2, 0], [128, 0]], lane=[[0, 0], [0, 0], [0, 0], [4, 0], [8, 0], [16, 0]], warp=[[0, 0], [32, 0], [64, 0]], block=[]}>
#blocked2 = #ttg.linear<{register=[                [1, 0], [2, 0]],           lane=[[0, 0], [0, 0], [0, 0], [4, 0], [8, 0], [16, 0]], warp=[[0, 0], [32, 0], [64, 0]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @extract_from_broadcasted_tensor(%arg0: tensor<256x1xi32, #blocked1> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: llvm.func @extract_from_broadcasted_tensor
    // CHECK-COUNT-32: %{{.*}} = llvm.extractvalue  %{{.*}} : !llvm.struct
    // CHECK-COUNT-4:  %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %0 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x1xi32, #blocked1> to tensor<128x1xi32, #blocked2>
    tt.return
  }
}

// -----

// Input tensor do not have broadcasted registers, resulting in total 8 values in tensor and 4 values per [128x1] tile.
// Output tensor broadcasts 4 registers along dimension 1 and total 16 values.
// Test checks that extract slice duplicates 4 values from input in 16 output values.
#blocked1 = #ttg.linear<{register=[                [1, 0], [2, 0], [128, 0]], lane=[[0, 0], [0, 0], [0, 0], [4, 0], [8, 0], [16, 0]], warp=[[0, 0], [32, 0], [64, 0]], block=[]}>
#blocked2 = #ttg.linear<{register=[[0, 0], [0, 0], [1, 0], [2, 0]],           lane=[[0, 0], [0, 0], [0, 0], [4, 0], [8, 0], [16, 0]], warp=[[0, 0], [32, 0], [64, 0]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @extract_to_broadcasted_tensor(%arg0: tensor<256x1xi32, #blocked1> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: llvm.func @extract_to_broadcasted_tensor
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue  %{{.*}} : !llvm.struct
    // CHECK-COUNT-16:  %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %72 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x1xi32, #blocked1> to tensor<128x1xi32, #blocked2>
    tt.return
  }
}
