// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-transpose-loads --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx1200" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 2], isTranspose = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot0
  tt.func @wmma_dot0(%arg0: !tt.ptr<bf16>) {
    %addr = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked>
    // CHECK-NOT: llvm.amdgcn.global.load.tr.b128
    %tensor_data = tt.load %addr : tensor<32x32x!tt.ptr<bf16>, #blocked>
    %tensor_dot = ttg.convert_layout %tensor_data : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
#mma = #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 2], isTranspose = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot1
  tt.func @wmma_dot1(%arg0: !tt.ptr<bf16>) {
    %addr = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked1>
    %tensor_data = tt.load %addr : tensor<32x32x!tt.ptr<bf16>, #blocked1>
    // CHECK-NOT: llvm.amdgcn.global.load.tr.b128
    %tensor_dot = ttg.convert_layout %tensor_data : tensor<32x32xbf16, #blocked1> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
#mma = #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 2], isTranspose = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot0_transpose
  tt.func @wmma_dot0_transpose(%arg0: !tt.ptr<bf16>) {
    %addr = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked1>
    // CHECK-COUNT-2: llvm.amdgcn.global.load.tr.b128
    %tensor_data = tt.load %addr : tensor<32x32x!tt.ptr<bf16>, #blocked1>
    %tensor_dot = ttg.convert_layout %tensor_data : tensor<32x32xbf16, #blocked1> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 2], isTranspose = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot1_transpose
  tt.func @wmma_dot1_transpose(%arg0: !tt.ptr<bf16>) {
    %addr = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked>
    // CHECK-COUNT-2: llvm.amdgcn.global.load.tr.b128
    %tensor_data = tt.load %addr : tensor<32x32x!tt.ptr<bf16>, #blocked>
    %tensor_dot = ttg.convert_layout %tensor_data : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    tt.return
  }
}
