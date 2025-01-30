// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
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
  tt.func public @mmav5_block_scaled(%a: tensor<128x64xi8, #blocked2>, %scale_a_ptr: tensor<128x2x!tt.ptr<i8>, #blocked1>, %b: tensor<64x128xi8, #blocked>, %scale_b_ptr: tensor<128x2x!tt.ptr<i8>, #blocked1>) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %scale_a = tt.load %scale_a_ptr: tensor<128x2x!tt.ptr<i8>, #blocked1>
    %scale_b = tt.load %scale_b_ptr: tensor<128x2x!tt.ptr<i8>, #blocked1>
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x64xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xi8, #blocked>, tensor<128x2xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.return %d : tensor<128x128xf32, #blocked>
  }
}
