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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Callee-owned storage must not escape and be overwritten by a later call.
  // expected-error @below {{returning tensor memory descriptors from functions that allocate tensor memory is not supported}}
  tt.func private @return_tmem() -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> {
    %one = arith.constant dense<1.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc %one : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return %buf : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  }

  tt.func private @overwrite_tmem() {
    %two = arith.constant dense<2.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc %two : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %read = ttng.tmem_load %buf : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  tt.func public @kernel() {
    %live = tt.call @return_tmem() : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.call @overwrite_tmem() : () -> ()
    %read = ttng.tmem_load %live : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func private @tmem_leaf() {
    %one = arith.constant dense<1.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc %one : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %read = ttng.tmem_load %buf : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  // The restriction also applies when the allocation is in a nested callee.
  // expected-error @below {{returning tensor memory descriptors from functions that allocate tensor memory is not supported}}
  tt.func private @return_after_nested_call(%arg: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> {
    tt.call @tmem_leaf() : () -> ()
    tt.return %arg : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The hardware requires 64-row A and accumulator allocations in the same
  // row half. Per-function allocation cannot constrain an incoming address.
  tt.func private @tmem_mma_incoming_rows(%a: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, %b: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>) attributes {noinline = true} {
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @below {{64-row MMA tensor memory operands passed through function arguments or results are not supported}}
    ttng.tc_gen5_mma %a, %b, %acc, %false, %true, %bar[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    ttng.wait_barrier %bar, %c0 deps %acc : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func private @tmem_mma_identity(%a: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>) -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable> attributes {noinline = true} {
    tt.return %a : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
  }
  tt.func public @tmem_mma_returned_rows(%values: tensor<64x64xf16, #blocked>, %b: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>) {
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %a = ttng.tmem_alloc %values : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %alias = tt.call @tmem_mma_identity(%a) : (!ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>) -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @below {{64-row MMA tensor memory operands passed through function arguments or results are not supported}}
    ttng.tc_gen5_mma %alias, %b, %acc, %false, %true, %bar[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    ttng.wait_barrier %bar, %c0 deps %acc : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
