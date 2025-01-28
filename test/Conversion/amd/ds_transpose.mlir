// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_n_t = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true, transA = true, transB = true}>
#mma_t_t = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true, transA = false, transB = true}>
#mma_t_n = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true, transA = false, transB = false}>
#mma_n_n = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true, transA = true, transB = false}>
#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  //  CHECK-LABEL: ds_transpose_n_t_fp16
  tt.func @ds_transpose_n_t_fp16(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>) {
    // CHECK-COUNT-32: llvm.call_intrinsic "llvm.amdgcn.ds.read.tr16.b64.v4f16.p3"({{.*}}) : (!llvm.ptr<3>) -> vector<4xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_n_t, kWidth = 4}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_n_t, kWidth = 4}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_fp16
  tt.func @ds_transpose_t_t_fp16(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>) {
    // CHECK-COUNT-16: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<4xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_t_t, kWidth = 4}>>
    // CHECK-COUNT-16: llvm.call_intrinsic "llvm.amdgcn.ds.read.tr16.b64.v4f16.p3"({{.*}}) : (!llvm.ptr<3>) -> vector<4xf16>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_t_t, kWidth = 4}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_fp16
  tt.func @ds_transpose_n_n_fp16(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>) {
    // CHECK-COUNT-16: llvm.call_intrinsic "llvm.amdgcn.ds.read.tr16.b64.v4f16.p3"({{.*}}) : (!llvm.ptr<3>) -> vector<4xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_n_n, kWidth = 4}>>
    // CHECK-COUNT-16: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<4xf16>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_n_n, kWidth = 4}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_n_fp16
  tt.func @ds_transpose_t_n_fp16(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>) {
    // CHECK-NOT: llvm.call_intrinsic "llvm.amdgcn.ds.read.tr16.b64.v4f16.p3"({{.*}}) : (!llvm.ptr<3>) -> vector<4xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_t_n, kWidth = 4}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_t_n, kWidth = 4}>>
    tt.return
  }

}
