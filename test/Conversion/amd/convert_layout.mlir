// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  tt.func @convert_layout_general_swizzling(%arg0: tensor<64x64xf32, #blocked0>, %arg1: tensor<64x64x!tt.ptr<f32>, #blocked1>) {

    // verify that following convert layout uses general swizzling

    // CHECK-DAG: [[SMEM:%.*]] = llvm.mlir.addressof @global_smem
    // CHECK-DAG: [[CST_0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NOT: llvm.add
    // CHECK-COUNT-6: llvm.select
    // CHECK-NOT: llvm.add
    // CHECK-DAG: [[OFFSET_0:%.*]] = llvm.xor
    // CHECK-DAG: [[OFFSET_1:%.*]] = llvm.xor [[CST_0]], [[OFFSET_0]] : i32
    // CHECK-DAG: [[OFFSET_2:%.*]] = llvm.xor [[OFFSET_1]], [[CST_0]] : i32
    // CHECK-DAG: [[OFFSET_3:%.*]] = llvm.add [[OFFSET_2]], [[CST_0]] : i32
    // CHECK-DAG: llvm.getelementptr inbounds [[SMEM]]{{\[}}[[OFFSET_3]]{{\]}}

    %0 = ttg.convert_layout %arg0 : tensor<64x64xf32, #blocked0> -> tensor<64x64xf32, #blocked1>
    tt.store %arg1, %0 : tensor<64x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: convert_layout_padding_swizzling
  tt.func @convert_layout_padding_swizzling(%arg0: tensor<64x64xf32, #blocked0>, %arg1: tensor<64x64x!tt.ptr<f32>, #blocked1>) {

    // verify that following convert layout uses padded path
    // see getVecAddr lambda in transferWithinBlockImpl function

    // CHECK-DAG: [[SMEM:%.*]] = llvm.mlir.addressof @global_smem
    // CHECK-DAG: [[CST_5:%.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK-DAG: [[CST_0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: [[OFFSET_0:%.*]] = llvm.lshr {{.*}}, [[CST_5]] : i32
    // CHECK-DAG: [[OFFSET_1:%.*]] = llvm.shl [[OFFSET_0]], [[CST_0]] : i32
    // CHECK-DAG: [[OFFSET_2:%.*]] = llvm.add [[OFFSET_1]], {{.*}} : i32
    // CHECK-DAG: llvm.getelementptr inbounds [[SMEM]]{{\[}}[[OFFSET_2]]{{\]}}

    %0 = ttg.convert_layout %arg0 {amdgpu.use_padded_scratch_shmem} : tensor<64x64xf32, #blocked0> -> tensor<64x64xf32, #blocked1>
    tt.store %arg1, %0 : tensor<64x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
