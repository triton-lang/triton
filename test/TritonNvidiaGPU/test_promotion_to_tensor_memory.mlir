// RUN:triton-opt %s -split-input-file -tritongpu-promote-lhs-to-tmem | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
// Incompatible access layout for tmem; tmem access requires one thread per datapath
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @no_tmem_promotion
  tt.func public @no_tmem_promotion(
    %lhs: tensor<128x32xf16, #blocked1>,
    %rhs: tensor<32x256xf16, #blocked2>
  ) {
    %true = arith.constant true
    %cst = arith.constant dense<0.0> : tensor<128x256xf32, #blocked>
    // CHECK: ttng.tmem_alloc %[[CST:.*]] : (tensor<128x256xf32, #[[BLOCKED:blocked[0-9]*]]>) -> !ttg.memdesc<128x256xf32, #tmem
    %tmem = ttng.tmem_alloc %cst :
      (tensor<128x256xf32, #blocked>) ->
      !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NOT: ttng.tmem_alloc %[[ARG0:.*]] : (tensor<128x32xf32, #[[BLOCKED:blocked[0-9]*]]>) -> !ttg.memdesc<128x32xf32, #[[TMEM:tmem[0-9]*]]
    %lhs_shared = ttg.local_alloc %lhs : (tensor<128x32xf16, #blocked1>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>
    %rhs_shared = ttg.local_alloc %rhs : (tensor<32x256xf16, #blocked2>) -> !ttg.memdesc<32x256xf16, #shared1, #ttg.shared_memory>

    ttng.tc_gen5_mma %lhs_shared, %rhs_shared, %tmem, %true, %true :
       !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<32x256xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
// Compatible layout for tmem access
#blocked3 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @promote_lhs_to_tmem
  tt.func public @promote_lhs_to_tmem(
    %lhs: tensor<128x32xf16, #blocked3>,
    %rhs: tensor<32x256xf16, #blocked2>
  ) {
    %true = arith.constant true
    %cst = arith.constant dense<0.0> : tensor<128x256xf32, #blocked>
    // CHECK: ttng.tmem_alloc %[[CST:.*]] : (tensor<128x256xf32, #[[BLOCKED:blocked[0-9]*]]>) -> !ttg.memdesc<128x256xf32, #tmem
    %tmem = ttng.tmem_alloc %cst :
      (tensor<128x256xf32, #blocked>) ->
      !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %[[ARG0:.*]] : (tensor<128x32xf16, #[[BLOCKED:blocked[0-9]*]]>) -> !ttg.memdesc<128x32xf16, #[[TMEM:tmem[0-9]*]]
    %lhs_shared = ttg.local_alloc %lhs : (tensor<128x32xf16, #blocked3>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>
    %rhs_shared = ttg.local_alloc %rhs : (tensor<32x256xf16, #blocked2>) -> !ttg.memdesc<32x256xf16, #shared1, #ttg.shared_memory>

    ttng.tc_gen5_mma %lhs_shared, %rhs_shared, %tmem, %true, %true :
       !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<32x256xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

    tt.return
  }
}

// -----

#lhs_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#rhs_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#lhs_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#rhs_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 32, CGALayout = [[0, 1]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1, CGALayout = [[0, 1]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-DAG: #[[LHS_TMEM:tmem[0-9]*]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 16, colStride = 1, CGALayout = {{\[\[0, 0\]\]}}>
  // CHECK-LABEL: @promote_lhs_uses_memdesc_cga
  tt.func public @promote_lhs_uses_memdesc_cga(
    %lhs: tensor<128x16xf32, #lhs_src>,
    %rhs: tensor<16x128xf32, #rhs_src>
  ) {
    %true = arith.constant true
    %lhs_src = arith.addf %lhs, %lhs : tensor<128x16xf32, #lhs_src>
    %lhs_shared = ttg.local_alloc %lhs_src :
      (tensor<128x16xf32, #lhs_src>) ->
      !ttg.memdesc<128x16xf32, #lhs_shared, #ttg.shared_memory>
    %rhs_shared = ttg.local_alloc %rhs :
      (tensor<16x128xf32, #rhs_src>) ->
      !ttg.memdesc<16x128xf32, #rhs_shared, #ttg.shared_memory>
    %acc_tmem = ttng.tmem_alloc :
      () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.*}} : (tensor<128x16xf32, #{{.*}}>) -> !ttg.memdesc<128x16xf32, #[[LHS_TMEM]], #ttng.tensor_memory>
    ttng.tc_gen5_mma %lhs_shared, %rhs_shared, %acc_tmem, %true, %true :
       !ttg.memdesc<128x16xf32, #lhs_shared, #ttg.shared_memory>,
       !ttg.memdesc<16x128xf32, #rhs_shared, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    tt.return
  }
}
