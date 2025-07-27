// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx950 --convert-builtin-func-to-llvm | FileCheck %s

#mma16 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#mma32 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  //  CHECK-LABEL: ds_transpose_n_t_fp16_mfma_16
  tt.func @ds_transpose_n_t_fp16_mfma_16(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-32: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_fp16_mfma_16_small_kWidth
  tt.func @ds_transpose_n_t_fp16_mfma_16_small_kWidth(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-32: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 4}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 4}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 4}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 4}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 4}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 4}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_fp16_mfma_16
  tt.func @ds_transpose_t_t_fp16_mfma_16(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    // CHECK-COUNT-16: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_fp16_mfma_16_small_kWdith
  tt.func @ds_transpose_t_t_fp16_mfma_16_small_kWdith(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    // CHECK-COUNT-16: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 4}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 4}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 4}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_fp16_mfma_16
  tt.func @ds_transpose_n_n_fp16_mfma_16(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_fp16_mfma_16_small_kWidth
  tt.func @ds_transpose_n_n_fp16_mfma_16_small_kWidth(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 4}>>
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 4}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 4}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_n_fp16_mfma_16
  tt.func @ds_transpose_t_n_fp16_mfma_16(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-NOT: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_fp16_mfma32
  tt.func @ds_transpose_n_t_fp16_mfma32(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-32: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_fp16_mfma32_small_kWidth
  tt.func @ds_transpose_n_t_fp16_mfma32_small_kWidth(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-32: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 4}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 4}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 4}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 4}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 4}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 4}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_fp16_mfma32
  tt.func @ds_transpose_t_t_fp16_mfma32(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    // CHECK-COUNT-16: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_fp16_mfma32_small_kWidth
  tt.func @ds_transpose_t_t_fp16_mfma32_small_kWidth(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    // CHECK-COUNT-16: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 4}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 4}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 4}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_fp16_mfma32
  tt.func @ds_transpose_n_n_fp16_mfma32(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_fp16_mfma32_small_kWidth
  tt.func @ds_transpose_n_n_fp16_mfma32_small_kWidth(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    // CHECK-NOT: rocdl.ds.read.tr16.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 4}>>
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 4}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 4}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_n_fp16_mfma32
  tt.func @ds_transpose_t_n_fp16_mfma32(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-NOT: rocdl.ds.read.tr16.b64 %{{.*}} : <3> -> vector<4xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_i8_mfma_16
  tt.func @ds_transpose_n_t_i8_mfma_16(%arg0: !ttg.memdesc<128x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared1, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_i8_mfma_16_small_kWidth
  tt.func @ds_transpose_n_t_i8_mfma_16_small_kWidth(%arg0: !ttg.memdesc<128x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared1, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_i8_mfma_16
  tt.func @ds_transpose_t_t_i8_mfma_16(%arg0: !ttg.memdesc<128x64xi8, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared1, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    // CHECK-COUNT-8: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared1, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_i8_mfma_16_small_kWidth
  tt.func @ds_transpose_t_t_i8_mfma_16_small_kWidth(%arg0: !ttg.memdesc<128x64xi8, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared1, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    // CHECK-COUNT-8: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared1, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_i8_mfma_16
  tt.func @ds_transpose_n_n_i8_mfma_16(%arg0: !ttg.memdesc<128x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_i8_mfma_16_small_kWidth
  tt.func @ds_transpose_n_n_i8_mfma_16_small_kWidth(%arg0: !ttg.memdesc<128x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_n_i8_mfma_16
  tt.func @ds_transpose_t_n_i8_mfma_16(%arg0: !ttg.memdesc<128x64xi8, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-NOT: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared1, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_i8_mfma32
  tt.func @ds_transpose_n_t_i8_mfma32(%arg0: !ttg.memdesc<128x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared1, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_i8_mfma32_small_kWidth
  tt.func @ds_transpose_n_t_i8_mfma32_small_kWidth(%arg0: !ttg.memdesc<128x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared1, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_i8_mfma32
  tt.func @ds_transpose_t_t_i8_mfma32(%arg0: !ttg.memdesc<128x64xi8, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared1, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    // CHECK-COUNT-8: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared1, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_i8_mfma32_small_kWidth
  tt.func @ds_transpose_t_t_i8_mfma32_small_kWidth(%arg0: !ttg.memdesc<128x64xi8, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared1, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    // CHECK-COUNT-8: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared1, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_i8_mfma32
  tt.func @ds_transpose_n_n_i8_mfma32(%arg0: !ttg.memdesc<128x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_i8_mfma32_small_kWidth
  tt.func @ds_transpose_n_n_i8_mfma32_small_kWidth(%arg0: !ttg.memdesc<128x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_n_i8_mfma32
  tt.func @ds_transpose_t_n_i8_mfma32(%arg0: !ttg.memdesc<128x64xi8, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-NOT: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared1, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_fp8_mfma_16
  tt.func @ds_transpose_n_t_fp8_mfma_16(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-32: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_fp8_mfma_16_small_kWidth
  tt.func @ds_transpose_n_t_fp8_mfma_16_small_kWidth(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-32: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_fp8_mfma_16
  tt.func @ds_transpose_t_t_fp8_mfma_16(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_fp8_mfma_16_small_kWidth
  tt.func @ds_transpose_t_t_fp8_mfma_16_small_kWidth(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_fp8_mfma_16
  tt.func @ds_transpose_n_n_fp8_mfma_16(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_fp8_mfma_16_small_kWidth
  tt.func @ds_transpose_n_n_fp8_mfma_16_small_kWidth(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_n_fp8_mfma_16
  tt.func @ds_transpose_t_n_fp8_mfma_16(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-NOT: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_fp8_mfma32
  tt.func @ds_transpose_n_t_fp8_mfma32(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-32: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_t_fp8_mfma32_small_kWidth
  tt.func @ds_transpose_n_t_fp8_mfma32_small_kWidth(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-32: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_fp8_mfma32
  tt.func @ds_transpose_t_t_fp8_mfma32(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_t_fp8_mfma32_small_kWidth
  tt.func @ds_transpose_t_t_fp8_mfma32_small_kWidth(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_fp8_mfma32
  tt.func @ds_transpose_n_n_fp8_mfma32(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_n_n_fp8_mfma32_small_kWidth
  tt.func @ds_transpose_n_n_fp8_mfma32_small_kWidth(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_n_fp8_mfma32
  tt.func @ds_transpose_t_n_fp8_mfma32(%arg0: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-NOT: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_fp4_mfma_32
  tt.func @ds_transpose_fp4_mfma_32(%arg0: !ttg.memdesc<128x128xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x128xi8, #shared1, #smem, mutable>, %arg2: !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>) {
    // CHECK-COUNT-32: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr8.b64 %{{.*}} : <3> -> vector<2xi32>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x128xi8, #shared, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<128x128xi8, #shared1, #smem, mutable> -> tensor<128x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma32>
    %3 = tt.dot_scaled %1, %2, %cst_2 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>> * tensor<128x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>> -> tensor<128x128xf32, #mma32>
    ttg.local_store %3, %arg2 : tensor<128x128xf32, #mma32> -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_fp4_mfma32_small
  tt.func @ds_transpose_t_fp4_mfma32_small(%arg0: !ttg.memdesc<16x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x16xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-4: rocdl.ds.read.tr4.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr4.b64
    %1 = amdgpu.local_load_packed_tranposed %arg0 : !ttg.memdesc<16x64xi8, #shared, #smem, mutable> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %2 = amdgpu.local_load_packed_tranposed %arg1 : !ttg.memdesc<64x16xi8, #shared1, #smem, mutable> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<32x32x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<32x32x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<32x32x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<32x32x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_fp4_mfma16
  tt.func @ds_transpose_t_fp4_mfma16(%arg0: !ttg.memdesc<8x128xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x8xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-4: rocdl.ds.read.tr4.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr4.b64
    %1 = amdgpu.local_load_packed_tranposed %arg0 : !ttg.memdesc<8x128xi8, #shared, #smem, mutable> -> tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %2 = amdgpu.local_load_packed_tranposed %arg1 : !ttg.memdesc<128x8xi8, #shared1, #smem, mutable> -> tensor<64x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<16x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x16x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<16x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x16x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_t_fp4_mfma32
  tt.func @ds_transpose_t_fp4_mfma32(%arg0: !ttg.memdesc<256x256xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<256x256xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-128: rocdl.ds.read.tr4.b64 %{{.*}} : <3> -> vector<2xi32>
    // CHECK-NOT: rocdl.ds.read.tr4.b64
    %1 = amdgpu.local_load_packed_tranposed %arg0 : !ttg.memdesc<256x256xi8, #shared, #smem, mutable> -> tensor<512x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %2 = amdgpu.local_load_packed_tranposed %arg1 : !ttg.memdesc<256x256xi8, #shared1, #smem, mutable> -> tensor<128x512xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<512x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x512x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.store %ptr1, %1 : tensor<512x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<128x512x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }
}
