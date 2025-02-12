// RUN: triton-opt %s -split-input-file -tritongpu-test-pipeline-lower-loop -canonicalize | FileCheck %s

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep
// CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] token %[[A_TOK3]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[RES:.*]] = arith.addf %[[ACC]], %[[A_VAL]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[RES]], %[[INS_NEXT]], %[[EXT_NEXT]]

tt.func @one_dep_async(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> tensor<128x32xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<128x32xf16, #A>) {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    %res = arith.addf %acc, %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x32xf16, #A>
    scf.yield %res : tensor<128x32xf16, #A>
  }
  tt.return %loop#0 : tensor<128x32xf16, #A>
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep_sync
// CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
// CHECK: scf.for
// CHECK:   tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
tt.func @one_dep_sync(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<1x!tt.ptr<f16>, #A>) -> tensor<1xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<1xf16, #A>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<1xf16, #A>) {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x!tt.ptr<f16>, #A>
    %res = arith.addf %acc, %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1xf16, #A>
    scf.yield %res : tensor<1xf16, #A>
  }
  tt.return %loop#0 : tensor<1xf16, #A>
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK: #[[SHARED:.*]] = #ttg.nvmma_shared
// CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x32xf16, #[[SHARED]], #
// CHECK:   %[[RES:.*]] = arith.addf %[[ACC]], %[[A_VAL]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[RES]], %[[INS_NEXT]], %[[EXT_NEXT]]

tt.func @one_dep_local_alloc(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> tensor<128x32xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<128x32xf16, #A>) {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    %a_alloc = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory, mutable>
    %a_load = ttg.local_load %a_alloc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory, mutable> -> tensor<128x32xf16, #A>
    %res = arith.addf %acc, %a_load {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x32xf16, #A>
    scf.yield %res : tensor<128x32xf16, #A>
  }
  tt.return %loop#0 : tensor<128x32xf16, #A>
}
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, instrShape = [16, 16, 16]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @shmem_pipelining_mmav3
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128
  // CHECK: %[[B:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_INS:.*]] = ttg.memdesc_subview %[[B]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[B_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK2:.*]] = ttg.async_commit_group %[[B_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK3:.*]] = ttg.async_wait %[[B_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[B_EXT:.*]] = ttg.memdesc_subview %[[B]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.warp_group_dot %[[A_EXT]], %[[B_EXT]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield {{.*}}, %[[INS_NEXT]], %[[EXT_NEXT]]

  tt.func public @shmem_pipelining_mmav3(%lb : index, %ub : index, %step : index,
                                              %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                                              %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>) -> tensor<128x128xf16, #mma> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : index {
      %A = tt.load %A_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B = tt.load %B_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_res = ttng.warp_group_dot %A_sh, %B_sh, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %acc_res : tensor<128x128xf32, #mma>
    }
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    tt.return %res_f16 : tensor<128x128xf16, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, instrShape = [16, 16, 16]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @no_shmem_pipelining_incompat_layout
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128
  // CHECK: %[[B:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_INS:.*]] = ttg.memdesc_subview %[[B]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[B_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK2:.*]] = ttg.async_commit_group %[[B_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK3:.*]] = ttg.async_wait %[[B_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[B_EXT:.*]] = ttg.memdesc_subview %[[B]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.warp_group_dot %[[A_EXT]], %[[B_EXT]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield {{.*}}, %[[INS_NEXT]], %[[EXT_NEXT]]

  tt.func public @no_shmem_pipelining_incompat_layout(%lb : index, %ub : index, %step : index,
                                              %A_ptr: tensor<128x128x!tt.ptr<f32>, #blocked1>,
                                              %B_ptr: tensor<128x128x!tt.ptr<f32>, #blocked1>) -> tensor<128x128xf32, #mma> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : index {
      %A = tt.load %A_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked1>
      %B = tt.load %B_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #shared1, #ttg.shared_memory>
      %acc_res = ttng.warp_group_dot %A_sh, %B_sh, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory> * !ttg.memdesc<128x128xf32, #shared1, #ttg.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %acc_res : tensor<128x128xf32, #mma>
    }
    tt.return %res : tensor<128x128xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @shmem_pipelining_mmav5
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128
  // CHECK: %[[B:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_INS:.*]] = ttg.memdesc_subview %[[B]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[B_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK2:.*]] = ttg.async_commit_group %[[B_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK3:.*]] = ttg.async_wait %[[B_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[B_EXT:.*]] = ttg.memdesc_subview %[[B]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.tc_gen5_mma %[[A_EXT]], %[[B_EXT]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield {{.*}}, %[[INS_NEXT]], %[[EXT_NEXT]]

  tt.func public @shmem_pipelining_mmav5(%lb : index, %ub : index, %step : index,
                                              %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                                              %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>) : index {
      %A = tt.load %A_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B = tt.load %B_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_tm = ttng.tmem_alloc %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    }
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}


// Tests to write:
// - mmav5 scaled with pipelined scale loads
// - load and mma layouts that don't match
// - load with multiple uses
// - assymetric loads
// - handling "other" for cp.async
// - not pipelining in shmem when "other" is used
