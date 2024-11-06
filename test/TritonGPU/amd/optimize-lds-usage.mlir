// RUN: triton-opt %s -split-input-file -optimize-amd-lds-usage=target-arch=gfx90a | FileCheck %s
// RUN: triton-opt %s -split-input-file -optimize-amd-lds-usage=target-arch=gfx90a -optimize-amd-lds-usage=lds-limit=32768 | FileCheck %s --check-prefix=CHECK-32KLIMIT

// Check that optimization detects overflow of LDS and decomposes layout convert so kernel fits into LDS
// CHECK-LABEL: alloc_convert_load
// CHECK-32KLIMIT-LABEL: alloc_convert_load
// CHECK: %0 = triton_gpu.local_alloc %arg0 : {{.*}}#blocked{{.*}}#shared
// CHECK: %1 = triton_gpu.convert_layout %arg1 : {{.*}}#blocked{{.*}}#blocked1
// CHECK: %2 = triton_gpu.convert_layout %1 : {{.*}}#blocked1{{.*}}#mma
// CHECK: %3 = triton_gpu.local_load %0 : {{.*}}#shared{{.*}}#triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [32, 32], isTransposed = false}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @alloc_convert_load(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x128xf32, #blocked>) attributes {noinline = false} {
    %1 = triton_gpu.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory>
    %2 = triton_gpu.convert_layout %arg1 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #mma>
    %3 = triton_gpu.local_load %1 : !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

// Check that optimization detects overflow of LDS and decomposes layout convert so kernel fits into LDS
// in case of relatively small scratch buffer
// CHECK-LABEL: alloc_convert_small_load
// CHECK-32KLIMIT-LABEL: alloc_convert_small_load
// CHECK: %0 = triton_gpu.local_alloc %arg0 : {{.*}}#blocked{{.*}}#shared
// CHECK: %1 = triton_gpu.convert_layout %arg1 : {{.*}}#blocked{{.*}}#blocked1
// CHECK: %2 = triton_gpu.convert_layout %1 : {{.*}}#blocked1{{.*}}#mma
// CHECK: %3 = triton_gpu.local_load %0 : {{.*}}#shared{{.*}}#triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [32, 32], isTransposed = false}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @alloc_convert_small_load(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x128xf16, #blocked>) attributes {noinline = false} {
    %1 = triton_gpu.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory>
    %2 = triton_gpu.convert_layout %arg1 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #mma>
    %3 = triton_gpu.local_load %1 : !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

// Check that optimization works with 3d tensors
// in case of relatively small scratch buffer
// CHECK-LABEL: alloc_convert_3d_load
// CHECK-32KLIMIT-LABEL: alloc_convert_3d_load
// CHECK: %0 = triton_gpu.local_alloc %arg0 : {{.*}}#blocked{{.*}}#shared
// CHECK: %1 = triton_gpu.convert_layout %arg1 : {{.*}}#blocked{{.*}}#mma
// CHECK: %2 = triton_gpu.convert_layout %1 : {{.*}}#mma{{.*}}#mma1
// CHECK: %3 = triton_gpu.local_load %0 : {{.*}}#shared{{.*}}#triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [1, 16, 4], warpsPerCTA = [1, 1, 8], order = [0, 1, 2]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1, 8], instrShape = [32, 32], isTransposed = false}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1, 2], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @alloc_convert_3d_load(%arg0: tensor<1x128x128xf16, #blocked>, %arg1: tensor<1x128x128xf16, #blocked>) attributes {noinline = false} {
    %1 = triton_gpu.local_alloc %arg0 : (tensor<1x128x128xf16, #blocked>) -> !tt.memdesc<1x128x128xf16, #shared, #triton_gpu.shared_memory>
    %2 = triton_gpu.convert_layout %arg1 : tensor<1x128x128xf16, #blocked> -> tensor<1x128x128xf16, #mma>
    %3 = triton_gpu.local_load %1 : !tt.memdesc<1x128x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<1x128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

// Check that optimization triggers with custom LDS limit and do not triggers with default one
// CHECK-LABEL: alloc_convert_32k_limit
// CHECK: %0 = triton_gpu.local_alloc %arg0 : {{.*}}#blocked{{.*}}#shared
// CHECK: %1 = triton_gpu.convert_layout %arg1 : {{.*}}#blocked{{.*}}#mma
// CHECK: %2 = triton_gpu.local_load %0 : {{.*}}#shared{{.*}}#triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK-32KLIMIT-LABEL: alloc_convert_32k_limit
// CHECK-32KLIMIT: %0 = triton_gpu.local_alloc %arg0 : {{.*}}#blocked{{.*}}#shared
// CHECK-32KLIMIT: %1 = triton_gpu.convert_layout %arg1 : {{.*}}#blocked{{.*}}#blocked1
// CHECK-32KLIMIT: %2 = triton_gpu.convert_layout %1 : {{.*}}#blocked1{{.*}}#mma
// CHECK-32KLIMIT: %3 = triton_gpu.local_load %0 : {{.*}}#shared{{.*}}#triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [32, 32], isTransposed = false}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @alloc_convert_32k_limit(%arg0: tensor<64x128xf16, #blocked>, %arg1: tensor<64x128xf16, #blocked>) attributes {noinline = false} {
    %1 = triton_gpu.local_alloc %arg0 : (tensor<64x128xf16, #blocked>) -> !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory>
    %2 = triton_gpu.convert_layout %arg1 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #mma>
    %3 = triton_gpu.local_load %1 : !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, kWidth = 4, parent = #mma}>>
    tt.return
  }
}

// -----

// Check that optimization correctly handles LDS shortcut (see #mma2 -> #dotop2 conversion)
// CHECK-DAG: [[BLOCKED_1:#[a-z0-9]*]] = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
// CHECK-DAG: [[BLOCKED_2:#[a-z0-9]*]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [0, 1]}>
// CHECK-DAG: [[MMA_1:#[a-z0-9]*]] = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
// CHECK-DAG: [[MMA_2:#[a-z0-9]*]] = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [32, 32], isTransposed = false}>
// CHECK-DAG: [[SHARED:#[a-z0-9]*]] = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>

// CHECK: tt.func public @mfma_dot_shortcut([[ARG_0:%[a-z0-9]*]]: {{.*}}, [[ARG_1:%[a-z0-9]*]]: {{.*}}, [[ARG_2:%[a-z0-9]*]]: {{.*}})
// CHECK: [[ALLOC:%[0-9]+]] = triton_gpu.local_alloc [[ARG_0]] : (tensor<128x128xf16, [[BLOCKED_1]]>) -> !tt.memdesc<128x128xf16, [[SHARED]], #triton_gpu.shared_memory>
// CHECK: [[INTERMEDIATE_CONV:%[0-9]+]] = triton_gpu.convert_layout [[ARG_1]] : tensor<128x128xf32, [[BLOCKED_1]]> -> tensor<128x128xf32, [[BLOCKED_2]]>
// CHECK: [[CONVERT_1:%[0-9]+]] = triton_gpu.convert_layout [[INTERMEDIATE_CONV]] : tensor<128x128xf32, [[BLOCKED_2]]> -> tensor<128x128xf32, [[MMA_2]]>
// CHECK: [[CONVERT_2:%[0-9]+]] = triton_gpu.convert_layout [[ARG_2]] : tensor<256x128xf16, [[MMA_1]]> -> tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[MMA_1]], kWidth = 4}>>
// CHECK: [[LOAD:%[0-9]+]] = triton_gpu.local_load [[ALLOC]] : !tt.memdesc<128x128xf16, [[SHARED]], #triton_gpu.shared_memory> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[MMA_2]], kWidth = 4}>>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma1 = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [32, 32], isTransposed = false}>
#mma2 = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#dotop1 = #triton_gpu.dot_op<{opIdx=0, parent=#mma1, kWidth=4}>
#dotop2 = #triton_gpu.dot_op<{opIdx=0, parent=#mma2, kWidth=4}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_shortcut(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x128xf32, #blocked>, %arg2: tensor<256x128xf16, #mma2>) attributes {noinline = false} {
    %alloc = triton_gpu.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory>
    %convert_1 = triton_gpu.convert_layout %arg1 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #mma1>
    %convert_2 = triton_gpu.convert_layout %arg2 : tensor<256x128xf16, #mma2> -> tensor<256x128xf16, #dotop2>
    %load = triton_gpu.local_load %alloc : !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf16, #dotop1>
    tt.return
  }
}

// Checks that optimization do not crash on 1d tensor
// CHECK-LABEL: convert_1d
// CHECK: triton_gpu.local_alloc
// CHECK-NEXT: triton_gpu.convert_layout
// CHECK-NEXT: triton_gpu.local_load
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @convert_1d(%arg0: tensor<128xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>>, %arg1: tensor<128x128xf32, #mma>) attributes {noinline = false} {
    %alloc = triton_gpu.local_alloc %arg1 : (tensor<128x128xf32, #mma>) -> !tt.memdesc<128x128xf32, #shared, #triton_gpu.shared_memory>
    %1 = triton_gpu.convert_layout %arg0 : tensor<128xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<128xf32, #blocked>
    %load = triton_gpu.local_load %alloc : !tt.memdesc<128x128xf32, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
    tt.return
  }
}
