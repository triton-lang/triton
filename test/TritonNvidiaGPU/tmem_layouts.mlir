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

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
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
