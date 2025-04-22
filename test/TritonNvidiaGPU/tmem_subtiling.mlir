// RUN: triton-opt %s -split-input-file --triton-nvidia-optimize-tmem-subtiling --allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 2, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 2], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @subtile_tmem_load
  tt.func public @subtile_tmem_load(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> (tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked>) {
    // CHECK: %[[S0:.+]] = ttng.tmem_subslice %{{.+}} {N = 0 : i32}
    // CHECK: %[[S1:.+]] = ttng.tmem_subslice %{{.+}} {N = 64 : i32}
    // CHECK: %[[L0:.+]] = ttng.tmem_load %[[S0]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[C0:.+]] = ttg.convert_layout %[[L0]]
    // CHECK: %[[L1:.+]] = ttng.tmem_load %[[S1]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[C1:.+]] = ttg.convert_layout %[[L1]]
    // CHECK: tt.return %[[C0]], %[[C1]]
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %1 = tt.reshape %0 : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked2>
    %2 = tt.trans %1 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked3>
    %3 = ttg.convert_layout %2 : tensor<128x64x2xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    %outLHS, %outRHS = tt.split %3 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked>
    tt.return %outLHS, %outRHS : tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [4, 1, 2], order = [1, 2, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 2, 1], order = [0, 2, 1]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 128, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 2], order = [0, 1, 2]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [
[0, 32, 0], [1, 0, 0], [2, 0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 64], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [1, 0], [2, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @subtile4_tmem_load
  tt.func public @subtile4_tmem_load(%arg0: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>) -> (tensor<128x64xf32, #blocked4>, tensor<128x64xf32, #blocked4>, tensor<128x64xf32, #blocked4>, tensor<128x64xf32, #blocked4>) {
    // CHECK: %[[S0:.+]] = ttng.tmem_subslice %{{.+}} {N = 0 : i32}
    // CHECK: %[[S1:.+]] = ttng.tmem_subslice %{{.+}} {N = 128 : i32}
    // CHECK: %[[S2:.+]] = ttng.tmem_subslice %[[S0]] {N = 0 : i32}
    // CHECK: %[[S3:.+]] = ttng.tmem_subslice %[[S0]] {N = 64 : i32}
    // CHECK: %[[S4:.+]] = ttng.tmem_subslice %[[S1]] {N = 0 : i32}
    // CHECK: %[[S5:.+]] = ttng.tmem_subslice %[[S1]] {N = 64 : i32}
    // CHECK: %[[L4:.+]] = ttng.tmem_load %[[S4]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[C4:.+]] = ttg.convert_layout %[[L4]]
    // CHECK: %[[L5:.+]] = ttng.tmem_load %[[S5]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[L2:.+]] = ttng.tmem_load %[[S2]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[C2:.+]] = ttg.convert_layout %[[L2]]
    // CHECK: %[[L3:.+]] = ttng.tmem_load %[[S3]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[C3:.+]] = ttg.convert_layout %[[L3]]
    // CHECK: %[[C5:.+]] = ttg.convert_layout %[[L5]]
    // CHECK: tt.return %[[C2]], %[[C3]], %[[C4]], %[[C5]]
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
    %1 = tt.reshape %0 : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked7>
    %2 = tt.trans %1 {order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked7> -> tensor<128x128x2xf32, #blocked8>
    %3 = ttg.convert_layout %2 : tensor<128x128x2xf32, #blocked8> -> tensor<128x128x2xf32, #linear>
    %outLHS, %outRHS = tt.split %3 : tensor<128x128x2xf32, #linear> -> tensor<128x128xf32, #linear1>
    %4 = tt.reshape %outLHS : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #blocked2>
    %5 = tt.trans %4 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked3>
    %outLHS_1, %outRHS_1 = tt.split %5 : tensor<128x64x2xf32, #blocked3> -> tensor<128x64xf32, #blocked4>
    %6 = tt.reshape %outRHS : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #blocked2>
    %7 = tt.trans %6 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked3>
    %outLHS_2, %outRHS_2 = tt.split %7 : tensor<128x64x2xf32, #blocked3> -> tensor<128x64xf32, #blocked4>
    tt.return %outLHS_1, %outRHS_1, %outLHS_2, %outRHS_2 : tensor<128x64xf32, #blocked4>, tensor<128x64xf32, #blocked4>, tensor<128x64xf32, #blocked4>, tensor<128x64xf32, #blocked4>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 2, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 2], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @subtile_tmem_load_256
  // CHECK-NOT: ttng.tmem_subslice
  // CHECK: tt.return
  tt.func public @subtile_tmem_load_256(%arg0: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> (tensor<256x64xf32, #blocked>, tensor<256x64xf32, #blocked>) {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #blocked1>
    %1 = tt.reshape %0 : tensor<256x128xf32, #blocked1> -> tensor<256x2x64xf32, #blocked2>
    %2 = tt.trans %1 {order = array<i32: 0, 2, 1>} : tensor<256x2x64xf32, #blocked2> -> tensor<256x64x2xf32, #blocked3>
    %3 = ttg.convert_layout %2 : tensor<256x64x2xf32, #blocked3> -> tensor<256x64x2xf32, #blocked4>
    %outLHS, %outRHS = tt.split %3 : tensor<256x64x2xf32, #blocked4> -> tensor<256x64xf32, #blocked>
    tt.return %outLHS, %outRHS : tensor<256x64xf32, #blocked>, tensor<256x64xf32, #blocked>
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 2, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 2], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {
  tt.func public @sink_load(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                            %arg1: tensor<128x128xf16, #blocked>,
                            %arg2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>)
                            -> (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>) {

    // CHECK: ttg.local_alloc
    // CHECK: ttng.tmem_load
    // CHECK: ttg.convert_layout
    // CHECK: arith.truncf
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %1 = tt.reshape %0 : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked2>
    %2 = tt.trans %1 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked3>
    %3 = ttg.convert_layout %2 : tensor<128x64x2xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    %outLHS, %outRHS = tt.split %3 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked>

    // CHECK: ttng.tmem_load
    // CHECK: ttg.convert_layout
    // CHECK: ttng.tmem_store
    // CHECK: arith.truncf
    %4 = ttg.local_alloc %arg1 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %5 = arith.truncf %outLHS : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked5>
    ttng.tmem_store %cst, %arg2, %true : tensor<128x128xf32, #blocked5> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %6 = arith.truncf %outRHS : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

    // CHECK: ttng.tmem_load
    // CHECK: ttg.convert_layout
    // CHECK: "unknow_may_side_effect"() : () -> ()
    // CHECK: arith.truncf
    %7 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %8 = tt.reshape %7 : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked2>
    %9 = tt.trans %8 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked3>
    %10 = ttg.convert_layout %9 : tensor<128x64x2xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    %outLHS2, %outRHS3 = tt.split %10 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked>
    "unknow_may_side_effect"() : () -> ()
    %11 = arith.truncf %outLHS2 : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

    ttg.local_dealloc %4 : !ttg.memdesc<128x128xf16, #shared, #smem>
    tt.return %5, %6, %11 : tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>
  }
}
