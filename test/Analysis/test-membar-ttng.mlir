// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefixes=CHECK,CF
// RUN: triton-opt %s -split-input-file                     --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefixes=CHECK,SCF

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @async_store_wait
tt.func @async_store_wait(%arg: tensor<32x16xf16, #AL>) {
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: async_tma_store_wait
  ttng.async_tma_store_wait {pendings = 0 : i32}
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttg.local_store
  ttg.local_store %arg, %alloc : tensor<32x16xf16, #AL> -> !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  tt.return
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 18944 : i32} {
// CHECK-LABEL: tma_special_cases
tt.func @tma_special_cases(%arg1: !tt.tensordesc<tensor<256x64xf16, #shared>>) -> (tensor<256x64xf16, #blocked>){
  %true = arith.constant 1 : i1
  %cx = arith.constant dense<1> : tensor<32xi32>
  %c0 = arith.constant 0 : i32
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  //      CHECK: ttng.init_barrier
  // CHECK-NEXT: ttng.init_barrier
  ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<tensor<256x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttng.wait_barrier
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<tensor<256x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttg.local_load
  %t = ttg.local_load %alloc : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #blocked>

  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<tensor<256x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: memdesc_subslice
  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttng.async_tma_gather
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttng.wait_barrier
  %view = ttg.memdesc_subslice %alloc [0, 0]  : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_gather %arg1[%cx, %c0] %view, %barrier, %true : !tt.tensordesc<tensor<256x64xf16, #shared>>, tensor<32xi32>, i32, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x64xf16, #shared, #ttg.shared_memory, mutable>, i1
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: ttng.inval_barrier
  // CHECK-NEXT: ttng.inval_barrier
  ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  tt.return %t : tensor<256x64xf16, #blocked>
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 18944 : i32} {
// CHECK-LABEL: tma_special_cases_cf
tt.func @tma_special_cases_cf(%arg1: !tt.tensordesc<tensor<256x64xf16, #shared>>, %i1 : i1, %arg2: tensor<256x64xf16, #blocked>) -> (tensor<256x64xf16, #blocked>){
  %true = arith.constant 1 : i1
  %c0 = arith.constant 0 : i32
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  // CF: cf.cond_br
  // SCF: scf.if
  scf.if %i1 {
    //  CHECK-NOT: gpu.barrier
    //      CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NEXT: ttng.barrier_expect
    // CHECK-NEXT: ttng.wait_barrier
    // CF-NEXT: cf.br
    // SCF-NEXT: } else {
    ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<tensor<256x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
    ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  } else {
    //  CHECK-NOT: gpu.barrier
    //      CHECK: ttg.local_store
    // CF-NEXT: cf.br
    // SCF-NEXT: }
    ttg.local_store %arg2, %alloc : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  }
  //      CHECK: gpu.barrier
  // CHECK-NEXT: ttg.local_load
  %t = ttg.local_load %alloc : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #blocked>
  tt.return %t : tensor<256x64xf16, #blocked>
}
}

// -----

// CHECK-LABEL: tmem_copy_after_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>

//#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @tmem_copy_after_alloc(%arg0: tensor<1x2048xf8E4M3FN, #blocked>) {
    // CHECK: local_alloc
    %0 = ttg.local_alloc %arg0 {allocation.offset = 53248 : i32} : (tensor<1x2048xf8E4M3FN, #blocked>) -> !ttg.memdesc<1x2048xf8E4M3FN, #shared, #smem>
    // CHECK: tmem_alloc
    %1 = ttng.tmem_alloc  {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    // gpu.barrier
    // CHECK: tmem_copy
    ttng.tmem_copy %0, %1 : !ttg.memdesc<1x2048xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}
