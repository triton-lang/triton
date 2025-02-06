// RUN: triton-opt %s -split-input-file -optimize-amd-lds-usage=target-arch=gfx90a | FileCheck %s
// RUN: triton-opt %s -split-input-file -optimize-amd-lds-usage=target-arch=gfx90a -optimize-amd-lds-usage=lds-limit=32768 | FileCheck %s --check-prefix=CHECK-32KLIMIT

// Check that optimization detects overflow of LDS and decomposes layout convert so kernel fits into LDS
// CHECK-LABEL: alloc_convert_load
// CHECK-32KLIMIT-LABEL: alloc_convert_load
// CHECK: %0 = ttg.local_alloc %arg0 : {{.*}}#blocked{{.*}}#shared
// CHECK: %1 = ttg.convert_layout %arg1 : {{.*}}#blocked{{.*}}#blocked1
// CHECK: %2 = ttg.convert_layout %1 : {{.*}}#blocked1{{.*}}#mma
// CHECK: %3 = ttg.local_load %0 : {{.*}}#shared{{.*}}#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [32, 32], isTransposed = false}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @alloc_convert_load(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x128xf32, #blocked>) attributes {noinline = false} {
    %1 = ttg.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %2 = ttg.convert_layout %arg1 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #mma>
    %3 = ttg.local_load %1 : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

// Check that optimization detects overflow of LDS and decomposes layout convert so kernel fits into LDS
// in case of relatively small scratch buffer
// CHECK-LABEL: alloc_convert_small_load
// CHECK-32KLIMIT-LABEL: alloc_convert_small_load
// CHECK: %0 = ttg.local_alloc %arg0 : {{.*}}#blocked{{.*}}#shared
// CHECK: %1 = ttg.convert_layout %arg1 : {{.*}}#blocked{{.*}}#blocked1
// CHECK: %2 = ttg.convert_layout %1 : {{.*}}#blocked1{{.*}}#mma
// CHECK: %3 = ttg.local_load %0 : {{.*}}#shared{{.*}}#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [32, 32], isTransposed = false}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @alloc_convert_small_load(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x128xf16, #blocked>) attributes {noinline = false} {
    %1 = ttg.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %2 = ttg.convert_layout %arg1 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #mma>
    %3 = ttg.local_load %1 : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

// Check that optimization triggers with custom LDS limit and do not triggers with default one
// CHECK-LABEL: alloc_convert_32k_limit
// CHECK: %0 = ttg.local_alloc %arg0 : {{.*}}#blocked{{.*}}#shared
// CHECK: %1 = ttg.convert_layout %arg1 : {{.*}}#blocked{{.*}}#mma
// CHECK: %2 = ttg.local_load %0 : {{.*}}#shared{{.*}}#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK-32KLIMIT-LABEL: alloc_convert_32k_limit
// CHECK-32KLIMIT: %0 = ttg.local_alloc %arg0 : {{.*}}#blocked{{.*}}#shared
// CHECK-32KLIMIT: %1 = ttg.convert_layout %arg1 : {{.*}}#blocked{{.*}}#blocked1
// CHECK-32KLIMIT: %2 = ttg.convert_layout %1 : {{.*}}#blocked1{{.*}}#mma
// CHECK-32KLIMIT: %3 = ttg.local_load %0 : {{.*}}#shared{{.*}}#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [32, 32], isTransposed = false}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @alloc_convert_32k_limit(%arg0: tensor<64x128xf16, #blocked>, %arg1: tensor<64x128xf16, #blocked>) attributes {noinline = false} {
    %1 = ttg.local_alloc %arg0 : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %2 = ttg.convert_layout %arg1 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #mma>
    %3 = ttg.local_load %1 : !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, kWidth = 4, parent = #mma}>>
    tt.return
  }
}


// -----

// Checks that optimization do not crash on 1d tensor
// CHECK-LABEL: convert_1d
// CHECK: ttg.local_alloc
// CHECK-NEXT: ttg.convert_layout
// CHECK-NEXT: ttg.local_load
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @convert_1d(%arg0: tensor<128xf32, #ttg.slice<{dim = 0, parent = #mma}>>, %arg1: tensor<128x128xf32, #mma>) attributes {noinline = false} {
    %alloc = ttg.local_alloc %arg1 : (tensor<128x128xf32, #mma>) -> !ttg.memdesc<128x128xf32, #shared, #smem>
    %1 = ttg.convert_layout %arg0 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<128xf32, #blocked>
    %load = ttg.local_load %alloc : !ttg.memdesc<128x128xf32, #shared, #smem> -> tensor<128x128xf32, #mma>
    tt.return
  }
}
