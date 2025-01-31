// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 4, 8, 1, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 1, 2, 3, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[32, 0], [64, 0], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$TMEM:.+]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
  // CHECK-DAG: #[[$TMEM1:.+]] = #ttng.tensor_memory_scales_encoding
  // CHECK-LABEL: mmav5_block_scaled
  //   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
  //   CHECK-DAG:   %[[A:.+]] = ttg.local_alloc %{{.*}} : (tensor<128x64xi8, #{{.*}}>) -> !ttg.memdesc<128x64xi8, #{{.*}}, #smem
  //   CHECK-DAG:   %[[B:.+]] = ttg.local_alloc %{{.*}} : (tensor<64x128xi8, #{{.*}}>) -> !ttg.memdesc<64x128xi8, #{{.*}}, #smem
  //   CHECK-DAG:   %[[ACC:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x128xf32, #{{.*}}>) -> !ttg.memdesc<128x128xf32, #{{.*}}, #ttng.tensor_memory, mutable>
  //       CHECK:   %[[SCALEA:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x2xi8, #{{.*}}>) -> !ttg.memdesc<128x2xi8, #[[$TMEM1]], #ttng.tensor_memory>
  //       CHECK:   %[[SCALEB:.+]] = ttng.tmem_alloc %{{.*}} : (tensor<128x2xi8, #{{.*}}>) -> !ttg.memdesc<128x2xi8, #[[$TMEM1]], #ttng.tensor_memory>
  //       CHECK:   ttng.tc_gen5_mma_scaled %[[A]], %[[B]], %[[ACC]], %[[SCALEA]], %[[SCALEB]], %[[TRUE]], %[[TRUE]] lhs = e4m3 rhs = e4m3
  tt.func public @mmav5_block_scaled(%a: tensor<128x128xi8, #blocked2>, %scale_a_ptr: tensor<1x1x32x4x4x!tt.ptr<i8>, #blocked3>, %b: tensor<128x128xi8, #blocked>, %scale_b_ptr: tensor<1x1x32x4x4x!tt.ptr<i8>, #blocked3>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %scale_a_5d = tt.load %scale_a_ptr: tensor<1x1x32x4x4x!tt.ptr<i8>, #blocked3>
    %scale_a_trans = tt.trans %scale_a_5d {order = array<i32: 0, 3, 2, 1, 4>} : tensor<1x1x32x4x4xi8, #blocked3> -> tensor<1x4x32x1x4xi8, #blocked4>
    %scale_a = tt.reshape %scale_a_trans : tensor<1x4x32x1x4xi8, #blocked4> -> tensor<128x4xi8, #linear>
    %scale_b_5d = tt.load %scale_b_ptr: tensor<1x1x32x4x4x!tt.ptr<i8>, #blocked3>
    %scale_b_trans = tt.trans %scale_b_5d {order = array<i32: 0, 3, 2, 1, 4>} : tensor<1x1x32x4x4xi8, #blocked3> -> tensor<1x4x32x1x4xi8, #blocked4>
    %scale_b = tt.reshape %scale_b_trans : tensor<1x4x32x1x4xi8, #blocked4> -> tensor<128x4xi8, #linear>
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xi8, #blocked2>, tensor<128x4xi8, #linear> * tensor<128x128xi8, #blocked>, tensor<128x4xi8, #linear> -> tensor<128x128xf32, #blocked>
    tt.return %d : tensor<128x128xf32, #blocked>
  }
}
