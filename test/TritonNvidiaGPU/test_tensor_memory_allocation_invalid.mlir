// RUN: triton-opt %s -split-input-file -triton-tensor-memory-allocation -verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100"} {
  tt.func private @ws_tmem_helper() {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  tt.func public @ws_kernel() {
    ttg.warp_specialize()
    default {
      // expected-error @below {{calls to functions that use tensor memory are not supported inside warp specialize regions}}
      tt.call @ws_tmem_helper() : () -> ()
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100"} {
  // expected-error @below {{cannot allocate tensor memory for recursive function calls}}
  tt.func private @tmem_recursive() {
    tt.call @tmem_recursive() : () -> ()
    tt.return
  }
}
