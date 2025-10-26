// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s

#mma_b16 = #ttg.amd_wmma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 32]}> // b16
#mma_b8 = #ttg.amd_wmma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 64]}> // b8
#mma_b8_2x = #ttg.amd_wmma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 128]}> // b8
#linear_ds_tr = #ttg.linear<{register = [[0, 64], [16, 0], [0, 1], [32, 0], [0, 2], [0, 4], [64, 0], [0, 8], [0, 32]],
                             lane = [[1, 0], [2, 0], [4, 0], [0, 16], [8, 0]], warp = [[0, 0], [0, 0]], block = []}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#padding = #ttg.padded_shared<[512:+16] {order = [0, 1], shape = [128, 64]}>
#padding_vec1 = #ttg.padded_shared<[1:+4] {order = [0, 1], shape = [128, 64]}>
#smem = #ttg.shared_memory

#linear_ds_tr_tile_out = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear_ds_tr_tile_invalid = #ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 4]], lane = [[1, 0], [4, 0], [2, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  //  CHECK-LABEL: b16_tests
  tt.func @b16_tests(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-32: llvm.call_intrinsic "llvm.amdgcn.ds.load.tr16.b128"(%{{.*}}) : (!llvm.ptr<3>) -> vector<8xf16>
    // CHECK-NOT: ds.load.tr16.b128
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_b16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma_b16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma_b16, kWidth = 8}>>
    tt.return
  }
  //  CHECK-LABEL: b16_tests_with_neg
  tt.func @b16_tests_with_neg(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>
    // CHECK-COUNT-16: llvm.call_intrinsic "llvm.amdgcn.ds.load.tr16.b128"(%{{.*}}) : (!llvm.ptr<3>) -> vector<8xf16>
    // CHECK-NOT: ds.load.tr16.b128
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_b16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma_b16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma_b16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: b8_tests
  tt.func @b8_tests(%arg0: !ttg.memdesc<128x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-48: llvm.call_intrinsic "llvm.amdgcn.ds.load.tr8.b64"(%{{.*}}) : (!llvm.ptr<3>) -> vector<2xi32>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma_b8_2x, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared1, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma_b8, kWidth = 8}>>
    // CHECK-NOT: ds.load.tr8.b64
    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma_b8_2x, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma_b8, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma_b8_2x, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma_b8, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: no_ds_read_tr
  tt.func @no_ds_read_tr(%arg0: !ttg.memdesc<128x64xi8, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xi8, #shared, #smem, mutable>, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    // CHECK-NOT: ds.load.tr8.b64
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xi8, #shared1, #smem, mutable> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma_b8_2x, kWidth = 16}>>
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xi8, #shared, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma_b8, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma_b8_2x, kWidth = 16}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma_b8, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 0, parent = #mma_b8_2x, kWidth = 16}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<i8>, #ttg.dot_op<{opIdx = 1, parent = #mma_b8, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_ll
  tt.func @ds_transpose_ll(%arg0: !ttg.memdesc<64x16xbf16, #shared, #smem>, %arg1: !tt.ptr<bf16>) {
    // CHECK-COUNT-4: llvm.call_intrinsic "llvm.amdgcn.ds.load.tr16.b128"(%{{.*}}) : (!llvm.ptr<3>) -> vector<8xbf16>
    // CHECK-NOT: ds.load.tr16.b128
    %a1 = ttg.local_load %arg0 : !ttg.memdesc<64x16xbf16, #shared, #smem> -> tensor<64x16xbf16, #linear_ds_tr_tile_out>

    %ptr1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x16x!tt.ptr<bf16>, #linear_ds_tr_tile_out>
    tt.store %ptr1, %a1 : tensor<64x16x!tt.ptr<bf16>, #linear_ds_tr_tile_out>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_ll_complex
  tt.func @ds_transpose_ll_complex(%arg0: !ttg.memdesc<64x16xbf16, #shared, #smem>, %arg1: !tt.ptr<bf16>) {
    // CHECK-COUNT-8: llvm.call_intrinsic "llvm.amdgcn.ds.load.tr16.b128"(%{{.*}}) : (!llvm.ptr<3>) -> vector<8xbf16>
    %a1 = ttg.local_load %arg0 : !ttg.memdesc<64x16xbf16, #shared, #smem> -> tensor<64x16xbf16, #linear_ds_tr>

    %ptr1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x16x!tt.ptr<bf16>, #linear_ds_tr>
    tt.store %ptr1, %a1 : tensor<64x16x!tt.ptr<bf16>, #linear_ds_tr>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_ll_invalid
  tt.func @ds_transpose_ll_invalid(%arg0: !ttg.memdesc<64x16xbf16, #shared, #smem>, %arg1: !tt.ptr<bf16>) {
    %a1 = ttg.local_load %arg0 : !ttg.memdesc<64x16xbf16, #shared, #smem> -> tensor<64x16xbf16, #linear_ds_tr_tile_invalid>
    // CHECK-NOT: ds.load.tr16.b128
    %ptr1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x16x!tt.ptr<bf16>, #linear_ds_tr_tile_invalid>
    tt.store %ptr1, %a1 : tensor<64x16x!tt.ptr<bf16>, #linear_ds_tr_tile_invalid>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_with_padding
  tt.func @ds_transpose_with_padding(%arg0: !ttg.memdesc<128x64xf16, #padding, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-COUNT-16: llvm.call_intrinsic "llvm.amdgcn.ds.load.tr16.b128"(%{{.*}}) : (!llvm.ptr<3>) -> vector<8xf16>
    // CHECK-NOT: ds.load.tr16.b128
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #padding, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: ds_transpose_padding_interval_too_small
  tt.func @ds_transpose_padding_interval_too_small(%arg0: !ttg.memdesc<128x64xf16, #padding_vec1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK-NOT: ds.load.tr16.b128
    %1 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #padding_vec1, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>

    %ptr1 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>
    tt.store %ptr1, %1 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma_b16, kWidth = 8}>>
    tt.return
  }
}
