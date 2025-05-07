// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:100 num-warps=4' | FileCheck %s

#tmem0 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 256, blockN = 64, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

// CHECK-DAG: [[BLOCKN64:#.*]] = #ttg.blocked<{sizePerThread = [1, 64]
// CHECK-DAG: [[BLOCKN128:#.*]] = #ttg.blocked<{sizePerThread = [1, 128]
// CHECK-DAG: [[SCALES:#.*]] = #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [32, 0], [64, 0], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64{{]]}}, lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0], [16, 0{{]]}}, warp = {{\[\[}}0, 0], [0, 0{{]]}}, block = []}>

// CHECK: @tmem_alloc
tt.func @tmem_alloc() {
  %cst = arith.constant dense<1.0> : tensor<128x128xf32>
  // CHECK: ttng.tmem_alloc {{.*}} (tensor<128x128xf32, [[BLOCKN128]]>) ->
  %result = ttng.tmem_alloc %cst : (tensor<128x128xf32>) -> !ttg.memdesc<128x128xf32, #tmem0, #ttng.tensor_memory>
  tt.return
}

// CHECK: @tmem_load
tt.func @tmem_load(%desc: !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory>) {
  // CHECK: ttng.tmem_load {{.*}} -> tensor<128x64xf32, [[BLOCKN64]]>
  %result = ttng.tmem_load %desc : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory> -> tensor<128x64xf32>
  tt.return
}

// CHECK: @tmem_store
tt.func @tmem_store(%desc: !ttg.memdesc<256x64xf32, #tmem2, #ttng.tensor_memory, mutable>) {
  %cst = arith.constant dense<1.0> : tensor<256x64xf32>
  %true = arith.constant true
  // CHECK: ttng.tmem_store {{.*}} tensor<256x64xf32, [[BLOCKN64]]> ->
  ttng.tmem_store %cst, %desc, %true : tensor<256x64xf32> -> !ttg.memdesc<256x64xf32, #tmem2, #ttng.tensor_memory, mutable>
  tt.return
}

// CHECK: @tmem_scales_layout
tt.func @tmem_scales_layout() {
  %cst = arith.constant dense<0> : tensor<128x128xi8>
  // CHECK: ttng.tmem_alloc {{.*}} (tensor<128x128xi8, [[SCALES]]>) ->
  %result = ttng.tmem_alloc %cst : (tensor<128x128xi8>) -> !ttg.memdesc<128x128xi8, #tmem_scales, #ttng.tensor_memory>
  tt.return
}
