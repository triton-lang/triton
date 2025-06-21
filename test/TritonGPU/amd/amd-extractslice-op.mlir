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
