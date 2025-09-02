// RUN: triton-opt %s -split-input-file --triton-nvidia-optimize-tmem-layouts --allow-unregistered-dialect | FileCheck %s

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
    // CHECK: %[[L0:.+]] = ttng.tmem_load %[[S0]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[C0:.+]] = ttg.convert_layout %[[L0]]
    // CHECK: %[[S1:.+]] = ttng.tmem_subslice %{{.+}} {N = 64 : i32}
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
    // CHECK: %[[S1:.+]] = ttng.tmem_subslice %[[S0]] {N = 0 : i32}
    // CHECK: %[[L1:.+]] = ttng.tmem_load %[[S1]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[C1:.+]] = ttg.convert_layout %[[L1]]
    // CHECK: %[[S2:.+]] = ttng.tmem_subslice %[[S0]] {N = 64 : i32}
    // CHECK: %[[L2:.+]] = ttng.tmem_load %[[S2]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[C2:.+]] = ttg.convert_layout %[[L2]]
    // CHECK: %[[S3:.+]] = ttng.tmem_subslice %{{.+}} {N = 128 : i32}
    // CHECK: %[[S4:.+]] = ttng.tmem_subslice %[[S3]] {N = 0 : i32}
    // CHECK: %[[L4:.+]] = ttng.tmem_load %[[S4]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[C4:.+]] = ttg.convert_layout %[[L4]]
    // CHECK: %[[S5:.+]] = ttng.tmem_subslice %[[S3]] {N = 64 : i32}
    // CHECK: %[[L5:.+]] = ttng.tmem_load %[[S5]] : !ttg.memdesc<128x64xf32
    // CHECK: %[[C5:.+]] = ttg.convert_layout %[[L5]]
    // CHECK: tt.return %[[C1]], %[[C2]], %[[C4]], %[[C5]]
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

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>
#linear = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @subtile_tmem_store
  tt.func public @subtile_tmem_store(
    %arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    %arg1: tensor<128x64xf32, #blocked5>,
    %arg2: tensor<128x64xf32, #blocked5>
  ) {
    // CHECK: [[S0:%.+]] = ttng.tmem_subslice %arg0 {N = 0 : i32}
    // CHECK: [[V0:%.+]] = ttg.convert_layout %arg1
    // CHECK: ttng.tmem_store [[V0]], [[S0]]
    // CHECK: [[S1:%.+]] = ttng.tmem_subslice %arg0 {N = 64 : i32}
    // CHECK: [[V1:%.+]] = ttg.convert_layout %arg2
    // CHECK: ttng.tmem_store [[V1]], [[S1]]
    %true = arith.constant true
    %joined = tt.join %arg1, %arg2 : tensor<128x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked6>
    %trans = tt.trans %joined {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked6> -> tensor<128x2x64xf32, #blocked7>
    %reshaped = tt.reshape %trans : tensor<128x2x64xf32, #blocked7> -> tensor<128x128xf32, #linear>
    %cvt = ttg.convert_layout %reshaped : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked>
    ttng.tmem_store %cvt, %arg0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [8, 1, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [8, 1, 1], order = [0, 1, 2]}>
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

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[32, 0], [64, 0], [16, 0]], block = []}>
// CHECK-LABEL: tmem_load_reduce
tt.func public @tmem_load_reduce(%arg0: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
  %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory> -> tensor<128x64xf32, #blocked>
  // CHECK: ttng.tmem_load %{{.*}} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory> -> tensor<128x64xf32, #linear>
  %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
  ^bb0(%arg2: f32, %arg3: f32):
    %2 = arith.addf %arg2, %arg3 : f32
    tt.reduce.return %2 : f32
  }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
}

}


// -----

#blocked = #ttg.blocked<{sizePerThread = [64, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [8, 0], [0, 8], [0, 16], [0, 32], [16, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[32, 0], [64, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABLE: test_tmem_store_dist_layout
  tt.func public @test_tmem_store_dist_layout(%arg0: f32, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    %0 = tt.splat %arg0 : f32 -> tensor<64x128xf32, #blocked>
    %1 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #blocked>
    %2 = arith.extf %1 : tensor<64x128xf16, #blocked> to tensor<64x128xf32, #blocked>
    %3 = arith.mulf %2, %0 : tensor<64x128xf32, #blocked>
    %4 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<64x128xf32, #blocked> -> tensor<128x64xf32, #blocked1>
    // CHECK: %[[C:.+]] = ttg.convert_layout %{{.+}} : tensor<128x64xf32, #{{.+}}> -> tensor<128x64xf32, #linear>
    // CHECK: ttng.tmem_store %[[C]], %{{.+}}, %{{.+}} : tensor<128x64xf32, #linear> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %4, %arg2, %true : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [64, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABLE: test_tmem_store_dist_layout_negative
  tt.func public @test_tmem_store_dist_layout_negative(%arg0: f32, %arg1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    %1 = ttg.local_load %arg1 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked1>
    %2 = arith.extf %1 : tensor<128x64xf16, #blocked1> to tensor<128x64xf32, #blocked1>
    // CHECK: %[[C:.+]] = arith.extf
    // CHECK: ttng.tmem_store %[[C]]
    ttng.tmem_store %2, %arg2, %true : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
