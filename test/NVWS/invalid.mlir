// RUN: triton-opt --split-input-file %s --verify-diagnostics

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_release_duplicate_async() {
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{async_ops contains duplicate async kind}}
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<none>, #nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_leading_dims_mismatch(%d : !ttg.memdesc<1x1xi32, #shared0, #smem>, %e : !ttg.memdesc<2x1xi32, #shared0, #smem>) {
    // expected-error @below {{inconsistent semaphore buffer depths}}
    %sem = nvws.semaphore.create %d, %e true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem>, !ttg.memdesc<2x1xi32, #shared0, #smem>]>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_buffer_used_elsewhere(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>) {
    // expected-error @below {{Semaphore buffer is used elsewhere, Semaphore cannot guarantee async safety}}
    %sem = nvws.semaphore.create %d true : !nvws.semaphore<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>]>
    %tmp = ttng.tmem_alloc %d : (!ttg.memdesc<1x64x16xf16, #shared0, #smem>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_partial_overlap_buffer_tuple_mismatch() {
    %c0_i32 = arith.constant 0 : i32
    %a = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %c = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem0 = nvws.semaphore.create %a, %b true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    // expected-error @below {{semaphores sharing a backing buffer must use identical ordered buffer operands}}
    %sem1 = nvws.semaphore.create %a, %c false : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem1[%c0_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    ttg.local_dealloc %a : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %b : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %c : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_permuted_buffer_tuple_mismatch() {
    %c0_i32 = arith.constant 0 : i32
    %a = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem0 = nvws.semaphore.create %a, %b true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    // expected-error @below {{semaphores sharing a backing buffer must use identical ordered buffer operands}}
    %sem1 = nvws.semaphore.create %b, %a false : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem1[%c0_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    ttg.local_dealloc %a : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %b : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_buffer_arity_mismatch() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{Semaphore has different number of arguments than buffer}}
    %views:2 = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_buffer_dimensions_mismatch() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{Dimensions don't match}}
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_inconsistent_pending_count() {
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    // expected-error @below {{inconsistent pending-count contribution for partition 1}}
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<none>, #nvws.async_op<wgmma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}
