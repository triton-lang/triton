// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer -verify-diagnostics

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem   = #ttg.shared_memory

// Test 1: local_alloc used in a function call should emit a warning and
//         get skipped by the sanitizer.
module attributes { "ttg.num-ctas" = 1 : i32,
                  "ttg.num-warps" = 1 : i32,
                  ttg.shared = 65544 : i32,
                  ttg.target = "cuda:90",
                  ttg.tensor_memory_size = 0 : i32,
                  "ttg.threads-per-warp" = 32 : i32,
                  "ttg.total-num-warps" = 1 : i32 } {
  // Dummy callee that takes the memdesc (made private to avoid multiple public funcs).
  tt.func private @callee(%arg: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    tt.return
  }

  // CHECK-LABEL: @test_call
  tt.func public @test_call() {
    %c0 = arith.constant 0 : i32
    // expected-warning@+1 {{Allocation is used in a function call, cannot instrument}}
    %buf = ttg.local_alloc {allocation.offset = 0 : i32}
           : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.call @callee(%buf) : (!ttg.memdesc<32x32xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem   = #ttg.shared_memory

// Test 2: local_alloc used in a non-trivial memdesc_index should emit a warning.
module attributes { "ttg.num-ctas" = 1 : i32,
                  "ttg.num-warps" = 1 : i32,
                  ttg.shared = 65544 : i32,
                  ttg.target = "cuda:90",
                  ttg.tensor_memory_size = 0 : i32,
                  "ttg.threads-per-warp" = 32 : i32,
                  "ttg.total-num-warps" = 1 : i32 } {
  // CHECK-LABEL: @inconsistent_subview
  tt.func public @inconsistent_subview() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-warning@+1 {{Allocation is used in an inconsistent way, cannot instrument}}
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32}
             : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %sub = ttg.memdesc_index %alloc[%c1]
           : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
           -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    "memdesc_use" (%alloc) : (!ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
