
// RUN: triton-opt %s -split-input-file --triton-nvidia-interleave-tmem --allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#subtile_layout = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {
  tt.func public @sink_load(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                            %arg1: tensor<128x128xf16, #blocked>,
                            %arg2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>)
                            -> (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>, tensor<128x128xf16, #blocked>) {

    // CHECK: ttg.local_alloc
    // CHECK: ttng.tmem_load
    // CHECK: ttg.convert_layout
    // CHECK: arith.truncf
    %subslice0 = ttng.tmem_subslice %arg0 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %subtile0 = ttng.tmem_load %subslice0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #subtile_layout>
    %outLHS = ttg.convert_layout %subtile0 : tensor<128x64xf32, #subtile_layout> -> tensor<128x64xf32, #blocked>
    %subslice1 = ttng.tmem_subslice %arg0 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %subtile1 = ttng.tmem_load %subslice1 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #subtile_layout>
    %outRHS = ttg.convert_layout %subtile1 : tensor<128x64xf32, #subtile_layout> -> tensor<128x64xf32, #blocked>

    // CHECK: ttng.tmem_load
    // CHECK: ttg.convert_layout
    // CHECK: ttng.tmem_store
    // CHECK: arith.truncf
    %4 = ttg.local_alloc %arg1 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %5 = arith.truncf %outLHS : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    ttng.tmem_store %cst, %arg2, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %6 = arith.truncf %outRHS : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

    // CHECK: ttng.tmem_load
    // CHECK: ttg.convert_layout
    // CHECK: "unknow_may_side_effect"() : () -> ()
    // CHECK: arith.truncf
    %7 = ttng.tmem_load %arg2 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %8 = ttg.convert_layout %7 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked>
    "unknow_may_side_effect"() : () -> ()
    %9 = arith.truncf %8 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>

    ttg.local_dealloc %4 : !ttg.memdesc<128x128xf16, #shared, #smem>
    tt.return %5, %6, %9 : tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>, tensor<128x128xf16, #blocked>
  }
}
