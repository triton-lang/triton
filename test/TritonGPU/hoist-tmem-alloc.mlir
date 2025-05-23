// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-hoist-tmem-alloc -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @chained_mma
  // CHECK: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[C0]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: %[[RES_TOK:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]])
  // CHECK-NOT: ttng.tmem_alloc
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC_TM]][%[[TOK]]]
  // CHECK-NOT: ttng.tmem_load
  // CHECK:   "end_of_loop"
  // CHECK:   yield %[[MMA_TOK]]
  // CHECK: %[[ACC_TM_LOAD:.*]], %{{.*}} = ttng.tmem_load %[[ACC_TM]][%[[RES_TOK]]]
  // CHECK: arith.truncf %[[ACC_TM_LOAD]]
  tt.func public @chained_mma(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "end_of_loop"() : () -> ()
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @changed_acc
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[C0]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: %[[RES_TOK:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]])
  // CHECK-NOT: ttng.tmem_alloc
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC_TM]][%[[TOK]]]
  // CHECK:   %[[ACC:.*]], %[[LOAD_TOK:.*]] = ttng.tmem_load %[[ACC_TM]][%[[MMA_TOK]]]
  // CHECK:   %[[ACC_MUL:.*]] = arith.mulf %[[ACC]]
  // CHECK:   %[[STORE_TOK:.*]] = ttng.tmem_store %[[ACC_MUL]], %[[ACC_TM]][%[[LOAD_TOK]]], %[[TRUE]]
  // CHECK:   yield %[[STORE_TOK]]
  // CHECK: %[[ACC_TM_LOAD:.*]], %{{.*}} = ttng.tmem_load %[[ACC_TM]]
  // CHECK: arith.truncf %[[ACC_TM_LOAD]]
  tt.func public @changed_acc(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc_if = arith.mulf %acc_res, %cst2 : tensor<128x128xf32, #blocked>
      scf.yield %acc_if : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @changed_acc_before_mma
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[C0]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: %[[RES_TOK:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]])
  // CHECK:   %[[ACC:.*]], %[[LOAD_TOK:.*]] = ttng.tmem_load %[[ACC_TM]][%[[TOK]]]
  // CHECK:   %[[ACC_MUL:.*]] = arith.mulf %[[ACC]]
  // CHECK:   %[[STORE_TOK:.*]] = ttng.tmem_store %[[ACC_MUL]], %[[ACC_TM]][%[[LOAD_TOK]]], %[[TRUE]]
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC_TM]][%[[STORE_TOK]]]
  // CHECK:   yield %[[MMA_TOK]]
  // CHECK: %[[ACC_TM_LOAD:.*]], %{{.*}} = ttng.tmem_load %[[ACC_TM]][%[[RES_TOK]]]
  // CHECK: arith.truncf %[[ACC_TM_LOAD]]
  tt.func public @changed_acc_before_mma(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_mul = arith.mulf %acc, %cst2 : tensor<128x128xf32, #blocked>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc_mul : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @select_after_mma
  // CHECK: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK: %[[CND:.*]] = "cnd"() : () -> i1
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[C0]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: %[[RES_TOK:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]])
  // CHECK-NOT: ttng.tmem_alloc
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC_TM]][%[[TOK]]]
  // CHECK-NOT: ttng.tmem_load
  // CHECK:   %[[CND_NEG:.*]] = arith.xori %[[CND]]
  // CHECK:   %[[STORE_TOK:.*]] = ttng.tmem_store {{.*}}, %[[ACC_TM]][%[[MMA_TOK]]], %[[CND_NEG]]
  // CHECK:   yield %[[STORE_TOK]]
  // CHECK: %[[ACC_TM_LOAD:.*]], %{{.*}} = ttng.tmem_load %[[ACC_TM]][%[[RES_TOK]]]
  // CHECK: arith.truncf %[[ACC_TM_LOAD]]
  tt.func public @select_after_mma(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cnd = "cnd"() : () -> i1
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc_if = arith.select %cnd, %acc_res, %cst2 : tensor<128x128xf32, #blocked>
      scf.yield %acc_if : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 =  #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @two_dots
  // CHECK: %[[ACC_TM1:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[ACC_TM2:.*]] = ttng.tmem_alloc : ()
  // CHECK: scf.for
  // CHECK:   ttng.tmem_store
  // CHECK:   ttng.tc_gen5_mma
  // CHECK:   ttng.tmem_load
  // CHECK:   ttng.tmem_store
  // CHECK:   ttng.tc_gen5_mma
  // CHECK:   ttng.tmem_load
  tt.func public @two_dots(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %acc_ptr: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %i = %c0_i32 to %arg3 step %c1_i32  : i32 {
      %3 = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %4 = ttg.local_alloc %3 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %5 = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %6 = ttg.local_alloc %5 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc = tt.load %acc_ptr : tensor<128x128x!tt.ptr<f32>, #blocked>

      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %4, %6, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

      %acc_tm2, %acc_tok2 = ttng.tmem_alloc %acc_res : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok2 = ttng.tc_gen5_mma %4, %6, %acc_tm2[%acc_tok2], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res2, %load_tok2 = ttng.tmem_load %acc_tm2[%mma_tok2] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

      tt.store %res_ptr, %acc_res2 : tensor<128x128x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, fp4Padded = true}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @hoist_constant_inputs
  tt.func public @hoist_constant_inputs(%arg0: !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem>, %arg2: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, %arg3: i32, %arg4: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: arith.trunci
    // CHECK: tt.splat
    // CHECK: ttng.tmem_alloc
    // CHECK: scf.for
    // CHECK:  ttng.tc_gen5_mma_scaled
    scf.for %arg5 = %c0_i32 to %arg3 step %c1_i32  : i32 {
      %0 = arith.trunci %arg3 : i32 to i8
      %1 = tt.splat %0 : i8 -> tensor<128x4xi8, #blocked1>
      %2 = ttng.tmem_alloc %1 : (tensor<128x4xi8, #blocked1>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
      ttng.tc_gen5_mma_scaled %arg0, %arg1, %arg4, %arg2, %2, %true, %true lhs = e5m2 rhs = e2m1 : !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, !ttg.memdesc<64x128xi8, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @use_in_conditional
  // CHECK: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK: %[[CND:.*]] = "cnd"() : () -> i1
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[C0]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: %[[RES_TOK:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]])
  // CHECK-NOT: ttng.tmem_alloc
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC_TM]][%[[TOK]]]
  // CHECK:   %[[CND_TOK:.*]] = scf.if %[[CND]]
  // CHECK:     "epilogue"()
  // CHECK:     %[[RESULT:.*]], %[[LOAD_TOK:.*]] = ttng.tmem_load %[[ACC_TM]][%[[MMA_TOK]]]
  // CHECK:     yield %[[LOAD_TOK]]
  // CHECK:   else
  // CHECK:     yield %[[MMA_TOK]]
  // CHECK:   %[[CND_NEG:.*]] = arith.xori %[[CND]]
  // CHECK:   %[[STORE_TOK:.*]] = ttng.tmem_store {{.*}}, %[[ACC_TM]][%[[CND_TOK]]], %[[CND_NEG]]
  // CHECK:   yield %[[STORE_TOK]]
  // CHECK: %[[ACC_TM_LOAD:.*]], %{{.*}} = ttng.tmem_load %[[ACC_TM]][%[[RES_TOK]]]
  // CHECK: arith.truncf %[[ACC_TM_LOAD]]
  tt.func public @use_in_conditional(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cnd = "cnd"() : () -> i1
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.if %cnd {
        "epilogue"() : () -> ()
        "user"(%acc_res) : (tensor<128x128xf32, #blocked>) -> ()
      }
      %acc_if = arith.select %cnd, %acc_res, %cst2 : tensor<128x128xf32, #blocked>
      scf.yield %acc_if : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  // CHECK-LABEL: @matmul_kernel_tma_persistent_nested
  tt.func public @matmul_kernel_tma_persistent_nested(%arg0: !tt.tensordesc<tensor<128x32xf8E4M3FN, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x32xf8E4M3FN, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
      // CHECK: %[[ACC:.*]], %[[TOK:.*]] = ttng.tmem_alloc
      // CHECK: %[[DIFF:.*]] = arith.subi %[[LIMIT:.*]], %[[START:.*]] : i32
      // CHECK: %[[COND:.*]] = arith.cmpi sle, %[[DIFF]], %[[ZERO]] : i32
      // CHECK-NEXT: %[[NTOK:.*]] = ttng.tmem_store %[[CST:.*]], %[[ACC]][%[[TOK]]], %[[COND]]
      // CHECK-NEXT: scf.for %[[ITER:.*]] = %[[START]] to %[[LIMIT]] step
      %20:3 = scf.for %arg19 = %arg11 to %arg12 step %c1_i32 iter_args(%arg20 = %cst, %arg21 = %c0_i32, %arg22 = %false) -> (tensor<128x128xf32, #blocked>, i32, i1)  : i32 {
        %28 = tt.descriptor_load %arg0[%arg19, %arg21] : !tt.tensordesc<tensor<128x32xf8E4M3FN, #shared>> -> tensor<128x32xf8E4M3FN, #blocked1>
        %29 = ttg.local_alloc %28 : (tensor<128x32xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x32xf8E4M3FN, #shared, #smem>
        %30 = tt.descriptor_load %arg5[%arg19, %arg21] : !tt.tensordesc<tensor<128x32xf8E4M3FN, #shared>> -> tensor<128x32xf8E4M3FN, #blocked1>
        %31 = ttg.local_alloc %30 : (tensor<128x32xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x32xf8E4M3FN, #shared, #smem>
        %32 = ttg.memdesc_trans %31 {order = array<i32: 1, 0>} : !ttg.memdesc<128x32xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<32x128xf8E4M3FN, #shared2, #smem>
        %acc, %acc_tok = ttng.tmem_alloc %arg20 : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %mma_tok = ttng.tc_gen5_mma %29, %32, %acc[%acc_tok], %arg22, %true : !ttg.memdesc<128x32xf8E4M3FN, #shared, #smem>, !ttg.memdesc<32x128xf8E4M3FN, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %34, %load_tok = ttng.tmem_load %acc[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %35 = arith.addi %arg21, %c32_i32 : i32
        scf.yield %34, %35, %true : tensor<128x128xf32, #blocked>, i32, i1
      }
      %21 = tt.reshape %20#0 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked2>
      %22 = tt.trans %21 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked3>
      %outLHS, %outRHS = tt.split %22 : tensor<128x64x2xf32, #blocked3> -> tensor<128x64xf32, #blocked4>
      %23 = tt.fp_to_fp %outLHS, rounding = rtne : tensor<128x64xf32, #blocked4> -> tensor<128x64xf8E4M3FN, #blocked4>
      %24 = ttg.convert_layout %23 : tensor<128x64xf8E4M3FN, #blocked4> -> tensor<128x64xf8E4M3FN, #blocked5>
      tt.descriptor_store %arg10[%c0_i32, %c0_i32], %24 : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared1>>, tensor<128x64xf8E4M3FN, #blocked5>
      %25 = tt.fp_to_fp %outRHS, rounding = rtne : tensor<128x64xf32, #blocked4> -> tensor<128x64xf8E4M3FN, #blocked4>
      %26 = ttg.convert_layout %25 : tensor<128x64xf8E4M3FN, #blocked4> -> tensor<128x64xf8E4M3FN, #blocked5>
      tt.descriptor_store %arg10[%c0_i32, %c0_i32], %26 : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared1>>, tensor<128x64xf8E4M3FN, #blocked5>
    }
    tt.return
  }
}
