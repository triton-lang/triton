// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:100 num-warps=4 enable-source-remat=true' -relayout-tritongpu | FileCheck %s

#tmem0 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

// CHECK-DAG: [[BLOCKN64:#.*]] = #ttg.blocked<{sizePerThread = [1, 64]
// CHECK-DAG: [[BLOCKN128:#.*]] = #ttg.blocked<{sizePerThread = [1, 128]
// CHECK-DAG: [[SCALES:#.*]] = #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [32, 0], [64, 0], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64{{]]}}, lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0], [16, 0{{]]}}, warp = {{\[\[}}0, 0], [0, 0{{]]}}, block = []}>
// CHECK-DAG: [[BLOCK64_SPLIT:#.*]] = #ttg.blocked<{sizePerThread = [1, 32]

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
tt.func @tmem_store(%desc: !ttg.memdesc<64x64xf32, #tmem2, #ttng.tensor_memory, mutable>) {
  %cst = arith.constant dense<1.0> : tensor<64x64xf32>
  %true = arith.constant true
  // CHECK: ttng.tmem_store {{.*}} tensor<64x64xf32, [[BLOCK64_SPLIT]]> ->
  ttng.tmem_store %cst, %desc, %true : tensor<64x64xf32> -> !ttg.memdesc<64x64xf32, #tmem2, #ttng.tensor_memory, mutable>
  tt.return
}

// CHECK: @tmem_scales_layout
tt.func @tmem_scales_layout() {
  %cst = arith.constant dense<0> : tensor<128x128xi8>
  // CHECK: ttng.tmem_alloc {{.*}} (tensor<128x128xi8, [[SCALES]]>) ->
  %result = ttng.tmem_alloc %cst : (tensor<128x128xi8>) -> !ttg.memdesc<128x128xi8, #tmem_scales, #ttng.tensor_memory>
  tt.return
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

// CHECK: [[SLICE_PARENT:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK: @async_tma_gather
tt.func @async_tma_gather(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %y_offset: i32,
                          %bar: !ttg.memdesc<1xi64, #bar_layout, #ttg.shared_memory, mutable>,
                          %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>,
                          %pred: i1) {
  %x_offsets = arith.constant dense<1> : tensor<32xi32>
  // CHECK: [[IDX:%.*]] = ttg.convert_layout %cst : tensor<32xi32, #{{.*}}> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = [[SLICE_PARENT]]}>>
  ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32>, i32, !ttg.memdesc<1xi64, #bar_layout, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>, i1
  tt.return
}

// CHECK: @async_tma_scatter
tt.func @async_tma_scatter(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %y_offset: i32,
                           %src: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>) {
  %x_offsets = arith.constant dense<1> : tensor<32xi32>
  // CHECK: [[IDX:%.*]] = ttg.convert_layout %cst : tensor<32xi32, #{{.*}}> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = [[SLICE_PARENT]]}>>
  ttng.async_tma_scatter %desc[%x_offsets, %y_offset] %src : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32>, i32, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>
  tt.return
}
