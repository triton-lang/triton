// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-test-pipeline-lower-loop -canonicalize | FileCheck %s
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @unscheduled_loop
// CHECK: scf.for
// CHECK:   tt.load
// CHECK:   "use"
tt.func @unscheduled_loop(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>
    "use"(%a) : (tensor<128x32xf16, #A>) -> ()
  }
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep_async
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi sge, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[ZERO]], %[[INS_P1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi sge, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[ZERO]], %[[EXT_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] token %[[A_TOK3]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use"(%[[A_VAL]]) {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[INS_NEXT]], %[[EXT_NEXT]]
// CHECK-DAG:   ttg.local_dealloc %[[A]]
// CHECK-DAG:   ttg.async_wait  {num = 0 : i32}

tt.func @one_dep_async(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep_barrier_wait
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x64
tt.func @one_dep_barrier_wait(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x64x!tt.ptr<f16>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                 %bar : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>,
                 %phase : i32) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x64x!tt.ptr<f16>, #A>
    %sh = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xf16, #A>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xf16, #A>) -> ()
    ttng.wait_barrier %bar, %phase deps %sh {loop.cluster = 3 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
  } {tt.scheduled_max_stage = 3 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep_barrier_wait_trans
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x64
tt.func @one_dep_barrier_wait_trans(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x64x!tt.ptr<f16>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                 %bar : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>,
                 %phase : i32) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x64x!tt.ptr<f16>, #A>
    %sh = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xf16, #A>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
    %trans = ttg.memdesc_trans %sh {order = array<i32: 1, 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xf16, #A>) -> ()
    ttng.wait_barrier %bar, %phase deps %trans {loop.cluster = 3 : i32, loop.stage = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>
  } {tt.scheduled_max_stage = 3 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @different_use_stages
// CHECK: scf.for
// CHECK:   ttg.async_copy_global_to_local %{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttg.async_wait {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   ttg.memdesc_subview {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_VAL:.*]] = ttg.local_load {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use1"(%[[A_VAL]]) {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use2"(%[[A_VAL]]) {loop.cluster = 0 : i32, loop.stage = 3 : i32}
tt.func @different_use_stages(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    "use1"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
    "use2"(%a) {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 3 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @used_by_if_yield
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK: scf.for
// CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   ttg.local_load {{.*}} token %[[A_TOK3]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use"{{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}

tt.func @used_by_if_yield(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                 %init_a : tensor<128x32xf16, #A>,
                 %cnd : i1) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    %a_if = scf.if %cnd -> tensor<128x32xf16, #A> {
      scf.yield %a : tensor<128x32xf16, #A>
    } else {
      scf.yield %init_a : tensor<128x32xf16, #A>
    } {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    "use"(%a_if) {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 3 : i32}
  tt.return
}
}
// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @dist1_load
tt.func @dist1_load(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                 %init_a : tensor<128x32xf16, #A>) -> () {
  %_ = scf.for %iv = %lb to %ub step %step iter_args(%prev_a = %init_a) -> (tensor<128x32xf16, #A>) : index {
    "use_next_iter"(%prev_a) {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x32xf16, #A>) -> ()
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
    scf.yield %a : tensor<128x32xf16, #A>
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep_sync
// CHECK: scf.for
// CHECK:   tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
tt.func @one_dep_sync(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<1x!tt.ptr<f16>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x!tt.ptr<f16>, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<1xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK: #[[SHARED:.*]] = #ttg.swizzled_shared
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi sge, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[ZERO]], %[[INS_P1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi sge, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[ZERO]], %[[EXT_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x32xf16, #[[SHARED]], #
// CHECK:   "use"(%[[A_VAL]]) {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[INS_NEXT]], %[[EXT_NEXT]]
// CHECK-DAG:   ttg.local_dealloc %[[A]]
// CHECK-DAG:   ttg.async_wait  {num = 0 : i32}
tt.func @one_dep_local_alloc(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    %a_alloc = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory, mutable>
    %a_load = ttg.local_load %a_alloc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory, mutable> -> tensor<128x32xf16, #A>
    "use"(%a_load) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_load_group
tt.func @one_load_group(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f32>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                       %b_ptr_init : tensor<128x32x!tt.ptr<f32>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> () {
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
  // Only one insert and extract index is used.
  // CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]]) ->
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]]
    // CHECK: %[[INS_CMP:.*]] = arith.cmpi sge, %[[INS_P1]], %[[NUM_BUFS]]
    // CHECK: %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[ZERO]], %[[INS_P1]]
    // CHECK: %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]]
    // CHECK: %[[EXT_CMP:.*]] = arith.cmpi sge, %[[EXT_P1]], %[[NUM_BUFS]]
    // CHECK: %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[ZERO]], %[[EXT_P1]]
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    %b = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    "use1"(%a){loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> ()
    "use2"(%b){loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @two_load_groups
tt.func @two_load_groups(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f32>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                       %b_ptr_init : tensor<128x32x!tt.ptr<f32>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                       %c_ptr_init : tensor<128x32x!tt.ptr<f32>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> () {
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS2:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK-DAG: %[[NUM_BUFS3:.*]] = arith.constant {{.*}} 3 : i32
  // Two insert and extract indices are used.
  // CHECK: scf.for {{.*}} iter_args(%[[INS2:.*]] = %[[MINUS_ONE]], %[[EXT2:.*]] = %[[MINUS_ONE]], %[[INS3:.*]] = %[[MINUS_ONE]], %[[EXT3:.*]] = %[[MINUS_ONE]]) ->
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK-DAG: %[[INS3_P1:.*]] = arith.addi %[[INS3]], %[[ONE]]
    // CHECK-DAG: %[[INS3_CMP:.*]] = arith.cmpi sge, %[[INS3_P1]], %[[NUM_BUFS3]]
    // CHECK-DAG: %[[INS3_NEXT:.*]] = arith.select %[[INS3_CMP]], %[[ZERO]], %[[INS3_P1]]
    // CHECK-DAG: %[[EXT3_P1:.*]] = arith.addi %[[EXT3]], %[[ONE]]
    // CHECK-DAG: %[[EXT3_CMP:.*]] = arith.cmpi sge, %[[EXT3_P1]], %[[NUM_BUFS3]]
    // CHECK-DAG: %[[EXT3_NEXT:.*]] = arith.select %[[EXT3_CMP]], %[[ZERO]], %[[EXT3_P1]]
    // CHECK-DAG: %[[INS2_P1:.*]] = arith.addi %[[INS2]], %[[ONE]]
    // CHECK-DAG: %[[INS2_CMP:.*]] = arith.cmpi sge, %[[INS2_P1]], %[[NUM_BUFS2]]
    // CHECK-DAG: %[[INS2_NEXT:.*]] = arith.select %[[INS2_CMP]], %[[ZERO]], %[[INS2_P1]]
    // CHECK-DAG: %[[EXT2_P1:.*]] = arith.addi %[[EXT2]], %[[ONE]]
    // CHECK-DAG: %[[EXT2_CMP:.*]] = arith.cmpi sge, %[[EXT2_P1]], %[[NUM_BUFS2]]
    // CHECK-DAG: %[[EXT2_NEXT:.*]] = arith.select %[[EXT2_CMP]], %[[ZERO]], %[[EXT2_P1]]
    %a = tt.load %a_ptr_init {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    %b = tt.load %a_ptr_init {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    %c = tt.load %a_ptr_init {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    "use1"(%a){loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> ()
    "use2"(%b){loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> ()
    "use3"(%c){loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x32xf32, #A>) -> ()
  } {tt.scheduled_max_stage = 3 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @dependent_loads
tt.func @dependent_loads(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f32>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> () {
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf32
  // CHECK: %[[C:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf32
  // CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]]) ->
  // CHECK: %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[INS_CMP:.*]] = arith.cmpi sge, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[ZERO]], %[[INS_P1]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[EXT_CMP:.*]] = arith.cmpi sge, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[ZERO]], %[[EXT_P1]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 2 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK: %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] token %[[A_TOK3]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[B:.*]] = "pointerize"(%[[A_VAL]]) {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_INS:.*]] = ttg.memdesc_subview %[[C]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK:.*]] = ttg.async_copy_global_to_local %[[B]], %[[C_INS]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK2:.*]] = ttg.async_commit_group %[[C_TOK]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK3:.*]] = ttg.async_wait %[[C_TOK2]] {loop.cluster = 0 : i32, loop.stage = 4 : i32, num = 0 : i32}
  // CHECK: %[[C_EXT:.*]] = ttg.memdesc_subview %[[C]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32}
  // CHECK: %[[C_VAL:.*]] = ttg.local_load %[[C_EXT]] token %[[C_TOK3]] {loop.cluster = 0 : i32, loop.stage = 4 : i32}
  // CHECK: "use1"(%[[C_VAL]]) {loop.cluster = 0 : i32, loop.stage = 4 : i32}
  // CHECK: scf.yield
  // CHECK-DAG: ttg.local_dealloc %[[A]]
  // CHECK-DAG: ttg.local_dealloc %[[C]]
  // CHECK-DAG:   ttg.async_wait  {num = 0 : i32}
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    %b = "pointerize"(%a) {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> tensor<128x32x!tt.ptr<f32>, #A>
    %c = tt.load %b {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    "use1"(%c){loop.cluster = 0 : i32, loop.stage = 4 : i32} : (tensor<128x32xf32, #A>) -> ()
  } {tt.scheduled_max_stage = 4 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @dependent_loads_asymmetric
// Loads have different latencies, should create two load groups.
tt.func @dependent_loads_asymmetric(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f32>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> () {
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS2:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK-DAG: %[[NUM_BUFS3:.*]] = arith.constant {{.*}} 3 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf32
  // CHECK: %[[C:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf32
  // CHECK: scf.for {{.*}} iter_args(%[[INS2:.*]] = %[[MINUS_ONE]], %[[EXT2:.*]] = %[[MINUS_ONE]], %[[INS3:.*]] = %[[MINUS_ONE]], %[[EXT3:.*]] = %[[MINUS_ONE]]) ->
  // CHECK-DAG: %[[INS3_P1:.*]] = arith.addi %[[INS3]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK-DAG: %[[INS3_CMP:.*]] = arith.cmpi sge, %[[INS3_P1]], %[[NUM_BUFS3]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK-DAG: %[[INS3_NEXT:.*]] = arith.select %[[INS3_CMP]], %[[ZERO]], %[[INS3_P1]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK-DAG: %[[EXT3_P1:.*]] = arith.addi %[[EXT3]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK-DAG: %[[EXT3_CMP:.*]] = arith.cmpi sge, %[[EXT3_P1]], %[[NUM_BUFS3]] {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK-DAG: %[[EXT3_NEXT:.*]] = arith.select %[[EXT3_CMP]], %[[ZERO]], %[[EXT3_P1]] {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK-DAG: %[[INS2_P1:.*]] = arith.addi %[[INS2]], %[[ONE]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK-DAG: %[[INS2_CMP:.*]] = arith.cmpi sge, %[[INS2_P1]], %[[NUM_BUFS2]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK-DAG: %[[INS2_NEXT:.*]] = arith.select %[[INS2_CMP]], %[[ZERO]], %[[INS2_P1]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK-DAG: %[[EXT2_P1:.*]] = arith.addi %[[EXT2]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK-DAG: %[[EXT2_CMP:.*]] = arith.cmpi sge, %[[EXT2_P1]], %[[NUM_BUFS2]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK-DAG: %[[EXT2_NEXT:.*]] = arith.select %[[EXT2_CMP]], %[[ZERO]], %[[EXT2_P1]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS2_NEXT]]{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 2 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK: %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT2_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] token %[[A_TOK3]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[B:.*]] = "pointerize"(%[[A_VAL]]) {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_INS:.*]] = ttg.memdesc_subview %[[C]][%[[INS3_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK:.*]] = ttg.async_copy_global_to_local %[[B]], %[[C_INS]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK2:.*]] = ttg.async_commit_group %[[C_TOK]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK3:.*]] = ttg.async_wait %[[C_TOK2]] {loop.cluster = 0 : i32, loop.stage = 5 : i32, num = 0 : i32}
  // CHECK: %[[C_EXT:.*]] = ttg.memdesc_subview %[[C]][%[[EXT3_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK: %[[C_VAL:.*]] = ttg.local_load %[[C_EXT]] token %[[C_TOK3]] {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK: "use1"(%[[C_VAL]]) {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK: scf.yield
  // CHECK-DAG: ttg.local_dealloc %[[A]]
  // CHECK-DAG: ttg.local_dealloc %[[C]]
  // CHECK-DAG: ttg.async_wait  {num = 0 : i32}
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    %b = "pointerize"(%a) {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> tensor<128x32x!tt.ptr<f32>, #A>
    %c = tt.load %b {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    "use1"(%c){loop.cluster = 0 : i32, loop.stage = 5 : i32} : (tensor<128x32xf32, #A>) -> ()
  } {tt.scheduled_max_stage = 5 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @unused_load
tt.func @unused_load(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f32>, #A> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> () {
  // CHECK: scf.for
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: dummy
    %a = tt.load %a_ptr_init {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    "dummy"() : () -> ()
  } {tt.scheduled_max_stage = 1 : i32}
  tt.return
}
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, instrShape = [16, 16, 16]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @shmem_pipelining_mmav3
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 3 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: %[[B:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi sge, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[ZERO]], %[[INS_P1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi sge, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[ZERO]], %[[EXT_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
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
  // CHECK:   %[[A_EXT_TRANSP:.*]] = ttg.memdesc_trans %[[A_EXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>}
  // CHECK:   ttng.warp_group_dot %[[A_EXT_TRANSP]], %[[B_EXT]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield {{.*}}, %[[INS_NEXT]], %[[EXT_NEXT]]
  // CHECK-DAG: ttg.local_dealloc %[[A]]
  // CHECK-DAG: ttg.local_dealloc %[[B]]
  // CHECK-DAG: ttg.async_wait  {num = 0 : i32}
  tt.func public @shmem_pipelining_mmav3(%lb : index, %ub : index, %step : index,
                                              %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                                              %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> tensor<128x128xf16, #mma> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : index {
      %A = tt.load %A_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B = tt.load %B_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %A_transp = ttg.memdesc_trans %A_sh {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>
      %acc_res = ttng.warp_group_dot %A_transp, %B_sh, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory> * !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %acc_res : tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
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
  // The combination of blocked and shared layouts for operand B would result in cp.async with less than 4 bytes size.
  // We can't pipeline that using shared memory buffer.
  // CHECK-LABEL: @no_shmem_pipelining_incompat_layout
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 3 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi sge, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[ZERO]], %[[INS_P1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi sge, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[ZERO]], %[[EXT_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B:.*]] = tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_SH:.*]] = ttg.local_alloc %[[B]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.warp_group_dot %[[A_EXT]], %[[B_SH]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield {{.*}}, %[[INS_NEXT]], %[[EXT_NEXT]]
  // CHECK-DAG:   ttg.local_dealloc %[[A]]
  // CHECK-DAG:   ttg.async_wait  {num = 0 : i32}
  tt.func public @no_shmem_pipelining_incompat_layout(
                    %lb : index, %ub : index, %step : index,
                    %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                    %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> tensor<128x128xf32, #mma> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : index {
      %A = tt.load %A_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B = tt.load %B_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>
      %acc_res = ttng.warp_group_dot %A_sh, %B_sh, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %acc_res : tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    tt.return %res : tensor<128x128xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, instrShape = [16, 16, 16]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // non-zero "other" value is used in the load, while cp.async does not support it.
  // We can't feed the shared memory values directly to mma, we need other values being filled in the registers.
  // CHECK-LABEL: @no_shmem_pipelining_other_used
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128
  // CHECK: %[[B:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:[^,]*]] = %[[INIT]], %[[INS:[^,]*]] = %[[MINUS_ONE]], %[[EXT:[^,]*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi sge, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[ZERO]], %[[INS_P1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi sge, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[ZERO]], %[[EXT_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[A_LOAD:.*]] = ttg.local_load %[[A_EXT]] {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[A_MASKED:.*]] = arith.select {{.*}}, %[[A_LOAD]], {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_INS:.*]] = ttg.memdesc_subview %[[B]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[B_INS]] {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK2:.*]] = ttg.async_commit_group %[[B_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK3:.*]] = ttg.async_wait %[[B_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[B_EXT:.*]] = ttg.memdesc_subview %[[B]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_LOAD:.*]] = ttg.local_load %[[B_EXT]] {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_MASKED:.*]] = arith.select {{.*}}, %[[B_LOAD]], {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[A_SH:.*]] = ttg.local_alloc %[[A_MASKED]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_SH:.*]] = ttg.local_alloc %[[B_MASKED]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.warp_group_dot %[[A_SH]], %[[B_SH]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield {{.*}}, %[[INS_NEXT]], %[[EXT_NEXT]]
  // CHECK-DAG: ttg.local_dealloc %[[A]]
  // CHECK-DAG: ttg.local_dealloc %[[B]]
  // CHECK-DAG: ttg.async_wait  {num = 0 : i32}
  tt.func public @no_shmem_pipelining_other_used(
                      %lb : index, %ub : index, %step : index,
                      %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                      %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                      %mask: tensor<128x128xi1, #blocked1> {tt.constancy = 128 : i32},
                      %other: tensor<128x128xf16, #blocked1> {tt.constancy = 128 : i32}) -> tensor<128x128xf16, #mma> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : index {
      %A = tt.load %A_ptr, %mask, %other  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B = tt.load %B_ptr, %mask, %other {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_res = ttng.warp_group_dot %A_sh, %B_sh, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %acc_res : tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    tt.return %res_f16 : tensor<128x128xf16, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @shmem_pipelining_mmav5
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[TWO:.*]] = arith.constant{{.*}} 2 : i32
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant{{.*}}3 : i32
  // CHECK: %[[ACC_TM:.*]], %[[ACC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[INIT]], %[[ACC_TM]][%[[ACC_TOK]]]
  // CHECK: %[[BAR:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<2xi64
  // CHECK: %[[BAR_SUB1:.*]] = ttg.memdesc_subview %[[BAR]][%[[ZERO]]]
  // CHECK: ttng.init_barrier %[[BAR_SUB1]], 1
  // CHECK: %[[BAR_SUB2:.*]] = ttg.memdesc_subview %[[BAR]][%[[ONE]]]
  // CHECK: ttng.init_barrier %[[BAR_SUB2]], 1
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: %[[B:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: %[[FOR_RET:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]], %[[PHASE:.*]] = %[[ZERO]], %[[BAR_IDX:.*]] = %[[ZERO]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi sge, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[ZERO]], %[[INS_P1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi sge, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[ZERO]], %[[EXT_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
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
  // CHECK:   %[[BAR_SUB:.*]] = ttg.memdesc_subview %[[BAR]][%[[BAR_IDX]]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma %[[A_EXT]], %[[B_EXT]], %{{.*}}[%[[TOK]]], {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.wait_barrier %[[BAR_SUB]], %[[PHASE]] deps %[[A_EXT]], %[[B_EXT]] {loop.cluster = 0 : i32, loop.stage = 3 : i32}
  // CHECK:   %[[PHASE_NEG:.*]] = arith.xori %[[PHASE]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_P1:.*]] = arith.addi %[[BAR_IDX]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_CMP:.*]] = arith.cmpi sge, %[[BAR_IDX_P1]], %[[TWO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_NEXT:.*]] = arith.select %[[BAR_IDX_CMP]], %[[ZERO]], %[[BAR_IDX_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[PHASE_NEXT:.*]] = arith.select %[[BAR_IDX_CMP]], %[[PHASE_NEG]], %[[PHASE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield %[[MMA_TOK]], %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]], %[[INS_NEXT]], %[[EXT_NEXT]]
  // CHECK-DAG: ttg.local_dealloc %[[A]]
  // CHECK-DAG: ttg.local_dealloc %[[B]]
  // CHECK-DAG: %[[BAR_SUB1:.*]] = ttg.memdesc_subview %[[BAR]][%[[ZERO]]]
  // CHECK-DAG: ttng.inval_barrier %[[BAR_SUB1]]
  // CHECK-DAG: %[[BAR_SUB2:.*]] = ttg.memdesc_subview %[[BAR]][%[[ONE]]]
  // CHECK-DAG: ttng.inval_barrier %[[BAR_SUB2]]
  // CHECK-DAG: ttg.local_dealloc %[[BAR]]
  // CHECK-DAG: ttg.async_wait {num = 0 : i32}
  tt.func public @shmem_pipelining_mmav5(%lb : index, %ub : index, %step : index,
                                              %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                                              %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %acc_tm, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %acc_tm[%acc_tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %i = %lb to %ub step %step iter_args(%tok = %init_tok) -> !ttg.async.token : index {
      %A = tt.load %A_ptr {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B = tt.load %B_ptr {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%tok], %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}
    %res, %res_tok = ttng.tmem_load %acc_tm[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#nvmma_64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tma_load_lowering
// CHECK-DAG: %[[TRUE:.*]] = arith.constant {{.*}} true
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1 : i32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK-DAG: %[[BARRIER:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
// CHECK: %[[BAR1_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ZERO]]]
// CHECK: ttng.init_barrier %[[BAR1_VIEW]], 1
// CHECK: %[[BAR2_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ONE]]]
// CHECK: ttng.init_barrier %[[BAR2_VIEW]], 1
// CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]], %[[PHASE:.*]] = %[[ZERO]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi sge, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[ZERO]], %[[INS_P1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi sge, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[ZERO]], %[[EXT_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[PHASE_XOR:.*]] = arith.xori %[[PHASE]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[PHASE_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[PHASE_XOR]], %[[PHASE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[BAR_INS:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[INS_NEXT]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttng.barrier_expect %[[BAR_INS]], 8192 {loop.cluster = 2 : i32, loop.stage = 0 : i32}, %[[TRUE]]
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]], %[[ZERO]], %[[ZERO]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttng.async_tma_copy_global_to_local {{.*}}[{{.*}}] %[[A_INS]], %[[BAR_INS]], %[[TRUE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[BAR_EXT:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[EXT_NEXT]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   ttng.wait_barrier %[[BAR_EXT]], %[[PHASE_NEXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]], %[[ZERO]], %[[ZERO]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_LOAD:.*]] = ttg.local_load %[[A_EXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use"(%[[A_LOAD]]) {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[INS_NEXT]], %[[EXT_NEXT]], %[[PHASE_NEXT]] : i32, i32, i32
// CHECK:  %[[BAR1_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ZERO]]]
// CHECK:  ttng.inval_barrier %[[BAR1_VIEW]]
// CHECK:  %[[BAR2_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ONE]]]
// CHECK:  ttng.inval_barrier %[[BAR2_VIEW]]
// CHECK:  ttg.local_dealloc %[[BARRIER]]
// CHECK:  ttg.local_dealloc %[[A]]
tt.func @tma_load_lowering(%lb : index, %ub : index, %step : index,
                 %desc : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>>,
                 %offs : i32) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.descriptor_load %desc[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>> -> tensor<128x32xf16, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#offsets = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tma_gather_lowering
// CHECK-DAG: %[[TRUE:.*]] = arith.constant {{.*}} true
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1 : i32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x32x128
// CHECK-DAG: %[[BARRIER:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
// CHECK: %[[BAR1_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ZERO]]]
// CHECK: ttng.init_barrier %[[BAR1_VIEW]], 1
// CHECK: %[[BAR2_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ONE]]]
// CHECK: ttng.init_barrier %[[BAR2_VIEW]], 1
// CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]], %[[PHASE:.*]] = %[[ZERO]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi sge, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[ZERO]], %[[INS_P1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi sge, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[ZERO]], %[[EXT_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[PHASE_XOR:.*]] = arith.xori %[[PHASE]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[PHASE_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[PHASE_XOR]], %[[PHASE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[BAR_INS:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[INS_NEXT]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttng.barrier_expect %[[BAR_INS]], 16384 {loop.cluster = 2 : i32, loop.stage = 0 : i32}, %[[TRUE]]
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]], %[[ZERO]], %[[ZERO]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttng.async_tma_gather {{.*}}[{{.*}}] %[[A_INS]], %[[BAR_INS]], %[[TRUE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[BAR_EXT:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[EXT_NEXT]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   ttng.wait_barrier %[[BAR_EXT]], %[[PHASE_NEXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]], %[[ZERO]], %[[ZERO]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_LOAD:.*]] = ttg.local_load %[[A_EXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use"(%[[A_LOAD]]) {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[INS_NEXT]], %[[EXT_NEXT]], %[[PHASE_NEXT]] : i32, i32, i32
// CHECK:  %[[BAR1_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ZERO]]]
// CHECK:  ttng.inval_barrier %[[BAR1_VIEW]]
// CHECK:  %[[BAR2_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ONE]]]
// CHECK:  ttng.inval_barrier %[[BAR2_VIEW]]
// CHECK-DAG: ttg.local_dealloc %[[BARRIER]]
// CHECK-DAG: ttg.local_dealloc %[[A]]
tt.func @tma_gather_lowering(%lb : index, %ub : index, %step : index,
                 %desc : !tt.tensordesc<tensor<1x128xf32, #nvmma_128>>,
                 %x : tensor<32xi32, #offsets>,
                 %y : i32) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.descriptor_gather %desc[%x, %y] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (!tt.tensordesc<tensor<1x128xf32, #nvmma_128>>, tensor<32xi32, #offsets>, i32) -> tensor<32x128xf32, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<32x128xf32, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#nvmma_64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tma_reuse_barrier
// CHECK: scf.for
// CHECK:   ttng.barrier_expect {{.*}}, 16384
// CHECK:   ttng.async_tma_copy_global_to_local
// CHECK-NOT: ttng.wait_barrier
// CHECK:   ttng.async_tma_copy_global_to_local
// CHECK:   ttng.wait_barrier
// CHECK:   "use1"
// CHECK:   "use2"
// CHECK:   ttng.barrier_expect {{.*}}, 8192
// CHECK:   ttng.async_tma_copy_global_to_local
// CHECK:   ttng.wait_barrier
// CHECK:   "use3"
tt.func @tma_reuse_barrier(%lb : index, %ub : index, %step : index,
                 %descA : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>>,
                 %descB : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>>,
                 %descC : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>>,
                 %offs : i32) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.descriptor_load %descA[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>> -> tensor<128x32xf16, #A>
    %b = tt.descriptor_load %descB[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>> -> tensor<128x32xf16, #A>
    "use1"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
    "use2"(%b) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
    %c = tt.descriptor_load %descC[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>> -> tensor<128x32xf16, #A>
    "use3"(%c) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, instrShape = [16, 16, 16]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tma_pipelining_mmav3
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3xi64
  // CHECK: scf.for
  // CHECK:   ttng.barrier_expect
  // CHECK:   ttng.async_tma_copy_global_to_local
  // CHECK-NOT: ttng.wait_barrier
  // CHECK:   ttng.async_tma_copy_global_to_local
  // CHECK:   ttng.wait_barrier
  // CHECK-NOT: ttg.local_alloc
  // CHECK:   ttng.warp_group_dot
  tt.func public @tma_pipelining_mmav3(%lb : index, %ub : index, %step : index,
                                              %descA : !tt.tensordesc<tensor<128x128xf16, #shared>>,
                                              %descB : !tt.tensordesc<tensor<128x128xf16, #shared>>,
                                              %offs : i32) -> tensor<128x128xf16, #mma> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : index {
      %A = tt.descriptor_load %descA[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B = tt.descriptor_load %descB[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_res = ttng.warp_group_dot %A_sh, %B_sh, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %acc_res : tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    tt.return %res_f16 : tensor<128x128xf16, #mma>
  }
}

// -----
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_descriptor_lowering
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[_128:.*]] = arith.constant{{.*}} 128 : i32
  // CHECK: %[[GLOBAL_ALLOC:.*]] = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
  // CHECK: scf.for {{.*}} iter_args(%[[IDX:.*]] = %[[ZERO]])
  // CHECK:   %[[OFFS:.*]] = arith.muli %[[IDX]], %[[_128]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   %[[DESC_PTR:.*]] = tt.addptr %[[GLOBAL_ALLOC]], %[[OFFS]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   ttng.tensormap_create %[[DESC_PTR]]{{.*}} loop.cluster = 0 : i32, loop.stage = 1 : i32
  // CHECK:   ttng.tensormap_fenceproxy_acquire %[[DESC_PTR]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   %[[DESC:.*]] = ttng.reinterpret_tensor_descriptor %[[DESC_PTR]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   %[[IDX_P1:.*]] = arith.addi %[[IDX]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   %[[IDX_CMP:.*]] = arith.cmpi sge, %[[IDX_P1]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   %[[IDX_NEXT:.*]] = arith.select %[[IDX_CMP]], %[[ZERO]], %[[IDX_P1]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   "use"(%[[DESC]]) {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  tt.func @tensor_descriptor_lowering(
    %lb : index, %ub : index, %step : index,
    %A: !tt.ptr<f16>,
    %shape_x: i32,
    %shape_y: i32,
    %strides_x: i64,
    %strides_y: i64) -> (){
    scf.for %iv = %lb to %ub step %step : index {
      %desc = tt.make_tensor_descriptor %A, [%shape_x, %shape_y], [%strides_x, %strides_y] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : <f16>, <tensor<128x128xf16, #nvmma_128>>
      "use"(%desc) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (!tt.tensordesc<tensor<128x128xf16, #nvmma_128>>) -> ()
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipelining_mmav5_scaled
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf8E5M2
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf8E5M2
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x1x2x32x4x4xi8
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x1x2x32x4x4xi8
  tt.func public @pipelining_mmav5_scaled(%lb : index, %ub : index, %step : index,
                                              %A_ptr: tensor<128x128x!tt.ptr<f8E5M2>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                                              %B_ptr: tensor<128x128x!tt.ptr<f8E5M2>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                                              %A_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                                              %B_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}) -> tensor<128x128xf32, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %acc_tm, %acc_tok = ttng.tmem_alloc %cst {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %last_tok = scf.for %i = %lb to %ub step %step iter_args(%tok = %acc_tok) -> !ttg.async.token : index {
      %A = tt.load %A_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>
      %B = tt.load %B_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E5M2, #blocked1>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E5M2, #blocked1>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>

      %A_sc = tt.load %A_sc_ptr {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
      %A_sc_sh = ttg.local_alloc %A_sc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

      %B_sc = tt.load %B_sc_ptr {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
      %B_sc_sh = ttg.local_alloc %B_sc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

      %mma_tok = ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm[%tok], %A_sc_sh, %B_sc_sh, %true, %true lhs = e5m2 rhs = e5m2 {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}
    %res, %res_tok = ttng.tmem_load %acc_tm[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return %res : tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @cnd_store_before_mma
  tt.func public @cnd_store_before_mma(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = "cnd"() : () -> i1
    %1, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Do not multibuffer tmem, as all the tmem uses are in the same stage.
    // CHECK: %[[ACC_TM:.*]], %[[ACC_TOK:.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32
    %init_tok = ttng.tmem_store %cst, %1[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      %4 = arith.xori %0, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i1
      %store_tok = ttng.tmem_store %cst_0, %1[%tok], %4 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %5 = tt.load %arg0 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %7 = tt.load %arg1 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %8 = ttg.local_alloc %7 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %mma_tok = ttng.tc_gen5_mma %6, %8, %1[%store_tok], %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32}
    %2, %load_tok = ttng.tmem_load %1[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %3 = arith.truncf %2 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %3 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @simple_persistent_mmav5
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[INIT_ACC:.*]] = "init_acc"()
  // CHECK-DAG: %[[OVERRIDE_ACC:.*]] = "override_acc"()
  // CHECK-DAG: %[[CND:.*]] = "cnd"()
  // CHECK-DAG: %[[C_N1:.*]] = arith.constant -1 : i32
  // CHECK-DAG: %[[C_0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C_1:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C_2:.*]] = arith.constant 2 : i32
  // CHECK: %[[ACC_TM:.*]], %[[ACC_TOK:.*]] = ttng.tmem_alloc  : () -> (!ttg.memdesc<2x128x128xf32
  // CHECK: %[[ACC_TM_SLICE:.*]] = ttg.memdesc_subview %[[ACC_TM]][%[[C_0]]
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[INIT_ACC]], %[[ACC_TM_SLICE]][%[[ACC_TOK]]], %[[TRUE]]
  // CHECK: %[[BAR:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<2xi64
  // CHECK: %[[BAR_SLICE:.*]] = ttg.memdesc_subview %[[BAR]][%[[C_0]]
  // CHECK: ttng.init_barrier %[[BAR_SLICE]], 1
  // CHECK: %[[BAR_SLICE_2:.*]] = ttg.memdesc_subview %[[BAR]][%[[C_1]]
  // CHECK: ttng.init_barrier %[[BAR_SLICE_2]], 1
  // CHECK: %[[FOR_RES:.*]]:6 = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]], %[[PHASE:.*]] = %[[C_0]], %[[BAR_IDX:.*]] = %[[C_0]], %[[BUF_IDX:.*]] = %[[C_N1]], %[[INSERT_IDX:.*]] = %[[C_N1]], %[[EXTRACT_IDX:.*]] = %[[C_N1]]
  // CHECK:   %[[BUF_IDX_P1:.*]] = arith.addi %[[BUF_IDX]], %[[C_1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BUF_IDX_CND:.*]] = arith.cmpi sge, %[[BUF_IDX_P1]], %[[C_2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BUF_IDX_NEXT:.*]] = arith.select %[[BUF_IDX_CND]], %[[C_0]], %[[BUF_IDX_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BUF_IDX_NEXT_CND:.*]] = arith.select %[[CND]], %[[BUF_IDX]], %[[BUF_IDX_NEXT]]
  // CHECK:   %[[TM_SLICE:.*]] = ttg.memdesc_subview %[[ACC_TM]][%[[BUF_IDX_NEXT_CND]], {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[STORE_TOK:.*]] = ttng.tmem_store %[[OVERRIDE_ACC]], %[[TM_SLICE]][], {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_SLICE:.*]] = ttg.memdesc_subview %[[BAR]][%[[BAR_IDX]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[ACC_TM_SLICE:.*]] = ttg.memdesc_subview %[[ACC_TM]][%[[BUF_IDX_NEXT_CND]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma %{{.*}}, %{{.*}}, %[[ACC_TM_SLICE]][], %[[TRUE]], %[[TRUE]], %[[BAR_SLICE]][%true] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]] deps %{{.*}}, %{{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
  // CHECK:   scf.if
  // CHECK:     %[[ACC_TM_SLICE:.*]] = ttg.memdesc_subview %[[ACC_TM]][%[[BUF_IDX_NEXT_CND]]
  // CHECK:     %[[LOAD_ACC:.*]], %[[USER_TOK:.*]] = ttng.tmem_load %[[ACC_TM_SLICE]][]
  // CHECK:     "use"(%[[LOAD_ACC]])
  // CHECK:   }
  // CHECK:   %[[PHASE_NEG:.*]] = arith.xori %[[PHASE]], %[[C_1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_P1:.*]] = arith.addi %[[BAR_IDX]], %[[C_1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_CND:.*]] = arith.cmpi sge, %[[BAR_IDX_P1]], %[[C_2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_NEXT:.*]] = arith.select %[[BAR_IDX_CND]], %[[C_0]], %[[BAR_IDX_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[PHASE_NEXT:.*]] = arith.select %[[BAR_IDX_CND]], %[[PHASE_NEG]], %[[PHASE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield %{{[0-9]+}}, %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]], %[[BUF_IDX_NEXT_CND]]
  // CHECK: } {tt.scheduled_max_stage = 3 : i32}
  // CHECK: %[[ACC_TM_SLICE:.*]] = ttg.memdesc_subview %[[ACC_TM]][%[[FOR_RES]]#3,
  // CHECK: %[[LOAD_ACC:.*]], %[[RES_TOK:.*]] = ttng.tmem_load %[[ACC_TM_SLICE]][%[[FOR_RES]]#0]
  tt.func public @simple_persistent_mmav5(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = "init_acc"() : () -> tensor<128x128xf32, #blocked1>
    %cst_0 = "override_acc"() : () -> tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = "cnd"() : () -> i1
    %1, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %1[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      %4 = arith.xori %0, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i1
      %store_tok = ttng.tmem_store %cst_0, %1[%tok], %4 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %5 = tt.load %arg0 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %7 = tt.load %arg1 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %8 = ttg.local_alloc %7 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %mma_tok = ttng.tc_gen5_mma %6, %8, %1[%store_tok], %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cnd_tok = scf.if %0 -> !ttg.async.token {
        %9, %user_tok = ttng.tmem_load %1[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
        "use"(%9) : (tensor<128x128xf32, #blocked1>) -> ()
        scf.yield %user_tok : !ttg.async.token
      } else {
        scf.yield %mma_tok : !ttg.async.token
      } {loop.cluster = 3 : i32, loop.stage = 3 : i32}
      scf.yield %cnd_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32}
    %2, %res_tok = ttng.tmem_load %1[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %3 = arith.truncf %2 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %3 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @simple_persistent_mmav5_acc_flag
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[INIT_ACC:.*]] = "init_acc"()
  // CHECK-DAG: %[[OVERRIDE_ACC:.*]] = "override_acc"()
  // CHECK-DAG: %[[CND:.*]] = "cnd"()
  // CHECK-DAG: %[[C_N1:.*]] = arith.constant -1 : i32
  // CHECK-DAG: %[[C_0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C_1:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C_2:.*]] = arith.constant 2 : i32
  // CHECK: %[[ACC_TM:.*]], %[[ACC_TOK:.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<2x128x128xf32
  // CHECK: %[[ACC_TM_SLICE:.*]] = ttg.memdesc_subview %[[ACC_TM]][%[[C_0]]
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[INIT_ACC]], %[[ACC_TM_SLICE]][%[[ACC_TOK]]], %[[TRUE]]
  // CHECK: %[[BAR:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<2xi64
  // CHECK: %[[BAR_SLICE:.*]] = ttg.memdesc_subview %[[BAR]][%[[C_0]]
  // CHECK: ttng.init_barrier %[[BAR_SLICE]], 1
  // CHECK: %[[BAR_SLICE_2:.*]] = ttg.memdesc_subview %[[BAR]][%[[C_1]]
  // CHECK: ttng.init_barrier %[[BAR_SLICE_2]], 1
  // CHECK: %[[FOR_RES:.*]]:6 = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]], %[[PHASE:.*]] = %[[C_0]], %[[BAR_IDX:.*]] = %[[C_0]], %[[BUF_IDX:.*]] = %[[C_N1]], %[[INSERT_IDX:.*]] = %[[C_N1]], %[[EXTRACT_IDX:.*]] = %[[C_N1]]
  // CHECK:   %[[BAR_SLICE:.*]] = ttg.memdesc_subview %[[BAR]][%[[BAR_IDX]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BUF_IDX_P1:.*]] = arith.addi %[[BUF_IDX]], %[[C_1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BUF_IDX_CND:.*]] = arith.cmpi sge, %[[BUF_IDX_P1]], %[[C_2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BUF_IDX_NEXT:.*]] = arith.select %[[BUF_IDX_CND]], %[[C_0]], %[[BUF_IDX_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BUF_IDX_NEXT_CND:.*]] = arith.select %[[CND]], %[[BUF_IDX]], %[[BUF_IDX_NEXT]]
  // CHECK:   %[[ACC_TM_SLICE:.*]] = ttg.memdesc_subview %[[ACC_TM]][%[[BUF_IDX_NEXT_CND]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma %{{.*}}, %{{.*}}, %[[ACC_TM_SLICE]][], %[[CND]], %[[TRUE]], %[[BAR_SLICE]][%true] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]] deps %{{.*}}, %{{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
  // CHECK:   scf.if
  // CHECK:     %[[ACC_TM_SLICE:.*]] = ttg.memdesc_subview %[[ACC_TM]][%[[BUF_IDX_NEXT_CND]]
  // CHECK:     %[[LOAD_ACC:.*]], %[[USER_TOK:.*]] = ttng.tmem_load %[[ACC_TM_SLICE]][]
  // CHECK:     "use"(%[[LOAD_ACC]])
  // CHECK:   }
  // CHECK:   %[[PHASE_NEG:.*]] = arith.xori %[[PHASE]], %[[C_1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_P1:.*]] = arith.addi %[[BAR_IDX]], %[[C_1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_CND:.*]] = arith.cmpi sge, %[[BAR_IDX_P1]], %[[C_2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_NEXT:.*]] = arith.select %[[BAR_IDX_CND]], %[[C_0]], %[[BAR_IDX_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[PHASE_NEXT:.*]] = arith.select %[[BAR_IDX_CND]], %[[PHASE_NEG]], %[[PHASE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield %{{[0-9]+}}, %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]], %[[BUF_IDX_NEXT_CND]]
  // CHECK: } {tt.scheduled_max_stage = 3 : i32}
  // CHECK: %[[ACC_TM_SLICE:.*]] = ttg.memdesc_subview %[[ACC_TM]][%[[FOR_RES]]#3,
  // CHECK: %[[LOAD_ACC:.*]], %[[RES_TOK:.*]] = ttng.tmem_load %[[ACC_TM_SLICE]][%[[FOR_RES]]#0]
  tt.func public @simple_persistent_mmav5_acc_flag(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = "init_acc"() : () -> tensor<128x128xf32, #blocked1>
    %cst_0 = "override_acc"() : () -> tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = "cnd"() : () -> i1
    %1, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %1[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      %5 = tt.load %arg0 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %7 = tt.load %arg1 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %8 = ttg.local_alloc %7 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %mma_tok = ttng.tc_gen5_mma %6, %8, %1[%tok], %0, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cnd_tok = scf.if %0 -> !ttg.async.token {
        %9, %user_tok = ttng.tmem_load %1[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
        "use"(%9) : (tensor<128x128xf32, #blocked1>) -> ()
        scf.yield %user_tok : !ttg.async.token
      } else {
        scf.yield %mma_tok : !ttg.async.token
      } {loop.cluster = 3 : i32, loop.stage = 3 : i32}
      scf.yield %cnd_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32}
    %2, %res_tok = ttng.tmem_load %1[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %3 = arith.truncf %2 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %3 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mmav5_load_in_different_cluster
  tt.func public @mmav5_load_in_different_cluster(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg4: i1) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %0[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg5 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      %3 = tt.load %arg0 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %4 = ttg.local_alloc %3 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %5 = tt.load %arg1 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
      // Wait should be in the cluster right before the load
      // CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
      // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32}
      %mma_tok = ttng.tc_gen5_mma %4, %6, %0[%tok], %false, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %7, %load_tok = ttng.tmem_load %0[%mma_tok] {loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      "use"(%7) {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x128xf32, #blocked1>) -> ()
      scf.yield %load_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32}
    %1, %res_tok = ttng.tmem_load %0[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %2 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @chained_dot_wait_before_store
  // CHECK-DAG: %[[C0_F:.+]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK-DAG: %[[CN1:.+]] = arith.constant -1 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C2:.+]] = arith.constant{{.*}} 2 : i32
  // CHECK: %[[TMEM_BUF:.+]], %[[ACC_TOK:.+]] = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32
  // CHECK: %[[INIT_TOK:.+]] = ttng.tmem_store %[[C0_F]], %[[TMEM_BUF]][%[[ACC_TOK]]]
  // CHECK: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK: ttng.init_barrier %[[BAR_SLICE0]], 1
  // CHECK: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK: ttng.init_barrier %[[BAR_SLICE1]], 1
  // CHECK: %[[LHS_BUFS:.+]] = ttg.local_alloc
  // CHECK: %[[RHS_BUFS:.+]] = ttg.local_alloc
  // CHECK: %[[FOR_RES:.+]]:5 = scf.for {{.*}} iter_args(%[[TOK:[^,]+]] = %[[INIT_TOK]], %[[PHASE:[^,]+]] = %[[C0]], %[[BAR_IDX:[^,]+]] = %[[C0]],
  // CHECK:   %[[IDX0:.+]] = arith.select
  // CHECK:   %[[IDX1:.+]] = arith.select
  // CHECK:   %[[LHS_DEP:.+]] = ttg.memdesc_subview %[[LHS_BUFS]][%[[IDX1]],
  // CHECK:   %[[RHS_DEP:.+]] = ttg.memdesc_subview %[[RHS_BUFS]][%[[IDX1]],
  // CHECK:   %[[BAR_SLICE:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[BAR_IDX]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[MMA_TOK:.+]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[TMEM_BUF]][%[[TOK]]], %[[TRUE]], %[[TRUE]], %[[BAR_SLICE]][%true] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]] deps %[[LHS_DEP]], %[[RHS_DEP]] {loop.cluster = 0 : i32, loop.stage = 3 : i32}
  // CHECK:   %[[CND_TOK:.+]] = scf.if
  // CHECK:     ttng.wait_barrier %[[BAR_SLICE]], %[[PHASE]] deps %[[LHS_DEP]], %[[RHS_DEP]]
  // CHECK:     %[[ACC_RES:.+]], %[[USER_TOK:.+]] = ttng.tmem_load %[[TMEM_BUF]][%[[MMA_TOK]]]
  // CHECK:     tt.store %{{.*}}, %[[ACC_RES]]
  // CHECK:     yield %[[USER_TOK]]
  // CHECK:   } else {
  // CHECK:     yield %[[MMA_TOK]]
  // CHECK:   } {loop.cluster = 3 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[PHASE_XOR:.+]] = arith.xori %[[PHASE]], %[[C1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_P1:.+]] = arith.addi %[[BAR_IDX]], %[[C1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_WRAP:.+]] = arith.cmpi sge, %[[BAR_IDX_P1]], %[[C2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_IDX_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[C0]], %[[BAR_IDX_P1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[PHASE_NEXT:.+]] = arith.select %[[BAR_WRAP]], %[[PHASE_XOR]], %[[PHASE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   yield %[[CND_TOK]]
  // CHECK: %[[BAR_SLICE0:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C0]]]
  // CHECK: ttng.inval_barrier %[[BAR_SLICE0]]
  // CHECK: %[[BAR_SLICE1:.+]] = ttg.memdesc_subview %[[BAR_BUF]][%[[C1]]]
  // CHECK: ttng.inval_barrier %[[BAR_SLICE1]]
  // CHECK: ttg.local_dealloc %[[BAR_BUF]]
  // CHECK: ttng.tmem_load %[[TMEM_BUF]][%[[FOR_RES]]#0]
  tt.func public @chained_dot_wait_before_store(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg4: i1) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %0[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg5 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      %3 = tt.load %arg0 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %4 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %5 = tt.load %arg1 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %mma_tok = ttng.tc_gen5_mma %4, %6, %0[%tok], %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cnd_tok = scf.if %arg4 -> !ttg.async.token {
        %7, %user_tok = ttng.tmem_load %0[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
        tt.store %arg3, %7 : tensor<128x128x!tt.ptr<f32>, #blocked1>
        scf.yield %user_tok : !ttg.async.token
      } else {
        scf.yield %mma_tok : !ttg.async.token
      } {loop.cluster = 3 : i32, loop.stage = 2 : i32}
      scf.yield %cnd_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32}
    %1, %res_tok = ttng.tmem_load %0[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %2 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @multibuf_tmem1
  tt.func public @multibuf_tmem1(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: tensor<128x128x!tt.ptr<f32>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // Multibuffer tmem as users are scheduled after defs
    // CHECK: ttng.tmem_alloc : () -> (!ttg.memdesc<2x128x128xf32
    %0, %acc_tok = ttng.tmem_alloc  : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32 iter_args(%tok = %acc_tok) -> !ttg.async.token : i32 {
      %2 = tt.load %arg0 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %3 = ttg.local_alloc %2 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %4 = tt.load %arg1 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg2 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked1>
      %store_tok = ttng.tmem_store %6, %0[%tok], %true {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mma_tok = ttng.tc_gen5_mma %3, %5, %0[%store_tok], %true, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %res, %load_tok = ttng.tmem_load %0[%mma_tok] {loop.cluster = 2 : i32, loop.stage = 3 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      scf.yield %load_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @multibuf_tmem2
  tt.func public @multibuf_tmem2(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: tensor<128x128x!tt.ptr<f32>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // Multibuffer tmem as users are scheduled after defs
    // CHECK: ttng.tmem_alloc : () -> (!ttg.memdesc<2x128x128xf32
    %0, %acc_tok = ttng.tmem_alloc  : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32 iter_args(%tok = %acc_tok) -> !ttg.async.token : i32 {
      %2 = tt.load %arg0 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %3 = ttg.local_alloc %2 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %4 = tt.load %arg1 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg2 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked1>
      %store_tok = ttng.tmem_store %6, %0[%tok], %true {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mma_tok = ttng.tc_gen5_mma %3, %5, %0[%store_tok], %true, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %res, %load_tok = ttng.tmem_load %0[%mma_tok] {loop.cluster = 3 : i32, loop.stage = 3 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      scf.yield %load_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @two_dots
  tt.func public @two_dots(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: tensor<128x128x!tt.ptr<f32>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // Do not multi buffer tmem as uses are scheduled before defs
    // CHECK: ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32
    // CHECK: ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32
    // CHECK: scf.for
    // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 4 : i32, loop.stage = 2 : i32}
    // CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 2 : i32, loop.stage = 3 : i32}
    // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 3 : i32, loop.stage = 3 : i32}
    // CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32}
    %0, %acc_tok0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %1, %acc_tok1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32 iter_args(%tok0 = %acc_tok0, %tok1 = %acc_tok1) -> (!ttg.async.token, !ttg.async.token) : i32 {
      %2 = tt.load %arg0 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %3 = ttg.local_alloc %2 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %4 = tt.load %arg1 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg2 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked1>

      %store_tok0 = ttng.tmem_store %6, %0[%tok0], %true {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mma_tok0 = ttng.tc_gen5_mma %3, %5, %0[%store_tok0], %true, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %7, %load_tok0 = ttng.tmem_load %0[%mma_tok0] {loop.cluster = 1 : i32, loop.stage = 3 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>

      %store_tok1 = ttng.tmem_store %7, %1[%tok1], %true {loop.cluster = 1 : i32, loop.stage = 3 : i32} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mma_tok1 = ttng.tc_gen5_mma %3, %5, %1[%store_tok1], %true, %true {loop.cluster = 1 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8, %load_tok1 = ttng.tmem_load %1[%mma_tok1] {loop.cluster = 0 : i32, loop.stage = 4 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>

      tt.store %arg3, %8 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked1>

      scf.yield %load_tok0, %load_tok1 : !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 4 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 2, 16], warpsPerCTA = [1, 1, 1, 4, 1], order = [4, 3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 2, 4, 4], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1], threadsPerWarp = [1, 4, 2, 1, 4], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 1, 2, 3, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 0, 0, 0, 0]], lane = [[0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0]], warp = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, fp4Padded = true}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scaled_mmav5_unswizzled(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg19: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg24: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg25: !tt.ptr<i32>, %arg26: !tt.ptr<i32>, %arg27: i32, %arg28: i32, %arg29: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %true = arith.constant true
    %c16_i64 = arith.constant 16 : i64
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    %cst_0 = arith.constant dense<127> : tensor<128x4xi8, #linear>
    %0 = tt.make_tensor_descriptor %arg6, [%c32_i32, %c32_i32], [%c32_i64, %c1_i64] : <f8E4M3FN>, <tensor<1x128xf8E4M3FN, #shared>>
    %1 = tt.make_tensor_descriptor %arg9, [%c32_i32, %c32_i32, %c32_i32], [%c32_i64, %c32_i64, %c1_i64] : <i8>, <tensor<1x64x256xi8, #shared1>>
    %2 = tt.make_tensor_descriptor %arg12, [%c32_i32, %c32_i32, %c32_i32, %c32_i32, %c16_i32], [%c32_i64, %c32_i64, %c32_i64, %c16_i64, %c1_i64] : <i8>, <tensor<1x2x1x32x16xi8, #shared2>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %4 = ttng.tmem_alloc %cst_0 : (tensor<128x4xi8, #linear>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
    %5, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %5[%acc_tok], %true : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      // Scale format is not tma compatible, so we have a local_alloc inside the loop
      // CHECK: ttng.wait_barrier
      // CHECK: ttg.local_load
      // CHECK: ttg.local_alloc
      // CHECK: ttng.wait_barrier
      // CHECK: ttng.tmem_alloc
      // CHECK: ttng.tc_gen5_mma_scaled

      %7 = tt.descriptor_gather %0[%3, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (!tt.tensordesc<tensor<1x128xf8E4M3FN, #shared>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, i32) -> tensor<128x128xf8E4M3FN, #blocked2>
      %8 = ttg.local_alloc %7 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E4M3FN, #blocked2>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %9 = tt.descriptor_load %1[%arg30, %c0_i32, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<1x64x256xi8, #shared1>> -> tensor<64x256xi8, #blocked2>
      %10 = ttg.local_alloc %9 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<64x256xi8, #blocked2>) -> !ttg.memdesc<64x256xi8, #shared3, #smem>
      %11 = tt.descriptor_load %2[%arg30, %c0_i32, %c0_i32, %c0_i32, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<1x2x1x32x16xi8, #shared2>> -> tensor<1x2x1x32x16xi8, #blocked3>
      %12 = tt.reshape %11 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x2x1x32x16xi8, #blocked3> -> tensor<2x1x32x4x4xi8, #blocked4>
      %13 = tt.trans %12 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 0, 3, 2, 1, 4>} : tensor<2x1x32x4x4xi8, #blocked4> -> tensor<2x4x32x1x4xi8, #blocked5>
      %14 = ttg.convert_layout %13 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<2x4x32x1x4xi8, #blocked5> -> tensor<2x4x32x1x4xi8, #linear1>
      %15 = tt.reshape %14 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<2x4x32x1x4xi8, #linear1> -> tensor<256x4xi8, #linear2>

      %16 = ttng.tmem_alloc %15 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<256x4xi8, #linear2>) -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
      %mma_tok = ttng.tc_gen5_mma_scaled %8, %10, %5[%tok], %4, %16, %true, %true lhs = e4m3 rhs = e2m1 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<64x256xi8, #shared3, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.scheduled_max_stage = 2 : i32}

    %6, %res_tok = ttng.tmem_load %5[%last_tok] : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @changed_acc_before_mma
  // CHECK-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK: %[[TMEM_BUF:.+]], %[[ACC_TOK:.+]] = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32
  // CHECK: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK: %[[A_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16
  // CHECK: %[[B_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16
  // CHECK: scf.for
  // CHECK:   %[[ACC1:.*]], %[[LOAD_TOK:.+]] = ttng.tmem_load %[[TMEM_BUF]][%{{.*}}] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[MUL:.*]] = arith.mulf %[[ACC1]], {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[STORE_TOK:.+]] = ttng.tmem_store %[[MUL]], %[[TMEM_BUF]][%[[LOAD_TOK]]], {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[BAR_SLICE:.*]] = ttg.memdesc_subview %[[BAR_BUF]]
  // CHECK:   %[[MMA_TOK:.+]] = ttng.tc_gen5_mma %[[A_SLICE:.*]], %[[B_SLICE:.*]], %[[TMEM_BUF]][%[[STORE_TOK]]], {{.*}}, {{.*}}, %[[BAR_SLICE]][%true] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.wait_barrier %[[BAR_SLICE]], {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32}
  // CHECK:   scf.yield %[[MMA_TOK]]
  tt.func public @changed_acc_before_mma(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %cst, %0[%acc_tok], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %last_tok = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%tok = %init_tok) -> !ttg.async.token : i32 {
      %3 = tt.load %arg0 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %4 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %5 = tt.load %arg1 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %7, %load_tok = ttng.tmem_load %0[%tok] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %8 = arith.mulf %7, %cst_0 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked1>
      %store_tok = ttng.tmem_store %8, %0[%load_tok], %true {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mma_tok = ttng.tc_gen5_mma %4, %6, %0[%store_tok], %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %mma_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32}
    %1, %res_tok = ttng.tmem_load %0[%last_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %2 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Check that wait is pushed to the next stage, right before the tmem_load, and after the prologue,
  // despite mma being impossible to pipeline.
  // CHECK-LABEL: @changed_acc_unpipelineable_operand
  // CHECK: scf.for
  // CHECK: "prologue"() {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: ttg.async_copy_global_to_local {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32}
  tt.func public @changed_acc_unpipelineable_operand(%A: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32},
                                                     %B: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32},
                                                      %arg1: i32, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %true = arith.constant true
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %0, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.for %arg4 = %arg1 to %arg2 step %arg3  : i32 {
      %2 = "prologue"() {loop.cluster = 0 : i32, loop.stage = 2 : i32} : () -> tensor<128x128xf16, #blocked2>
      %3 = ttng.tmem_load %0 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %4 = "acc_modify"(%3, %2) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked1>, tensor<128x128xf16, #blocked2>) -> tensor<128x128xf32, #blocked1>
      %5 = tt.load %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %7 = tt.load %A {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %8 = ttg.local_alloc %7 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      ttng.tmem_store %4, %0, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %6, %8, %0, %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    } {tt.scheduled_max_stage = 2 : i32}
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%1) : (tensor<128x128xf32, #blocked1>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Check that wait is pushed to the next stage, right before the tmem_load, and after the prologue,
  // despite mma being impossible to pipeline.
  // CHECK-LABEL: @changed_acc_unpipelineable_operand2
  // CHECK: scf.for
  // CHECK: "prologue"() {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK: ttg.async_copy_global_to_local {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: ttng.tmem_load {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32}
  tt.func public @changed_acc_unpipelineable_operand2(%A: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32},
                                                     %B: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32},
                                                      %arg1: i32, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %true = arith.constant true
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %0, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.for %arg4 = %arg1 to %arg2 step %arg3  : i32 {
      %2 = "prologue"() {loop.cluster = 0 : i32, loop.stage = 2 : i32} : () -> tensor<128x128xf16, #blocked2>
      %5 = tt.load %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %3 = ttng.tmem_load %0 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %4 = "acc_modify"(%3, %2) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked1>, tensor<128x128xf16, #blocked2>) -> tensor<128x128xf32, #blocked1>

      %7 = tt.load %A {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %8 = ttg.local_alloc %7 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      ttng.tmem_store %4, %0, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %6, %8, %0, %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    } {tt.scheduled_max_stage = 2 : i32}
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%1) : (tensor<128x128xf32, #blocked1>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Check that wait is pushed to the next stage, right before the tmem_alloc, and after the prologue.
  // Check that tmem is hoisted out of the loop.
  // CHECK-LABEL: @wait_before_tmem_alloc
  // CHECK: ttng.tmem_alloc
  // CHECK: %[[TMEM_BUF:.+]] = ttng.tmem_alloc
  // CHECK: scf.for
  // CHECK: "prologue"() {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK: ttng.tmem_store {{.*}}, %[[TMEM_BUF]], {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: ttg.async_copy_global_to_local {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32}
  tt.func public @wait_before_tmem_alloc(%A: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32},
                                                     %B: tensor<128x128xf16, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32},
                                                      %arg1: i32, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %true = arith.constant true
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %0, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.for %arg4 = %arg1 to %arg2 step %arg3  : i32 {
      %2 = "prologue"() {loop.cluster = 0 : i32, loop.stage = 2 : i32} : () -> tensor<128x128xf16, #blocked2>
      %8 = ttng.tmem_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
      %5 = tt.load %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      ttng.tc_gen5_mma %6, %8, %0, %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    } {tt.scheduled_max_stage = 2 : i32}
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%1) : (tensor<128x128xf32, #blocked1>) -> ()
    tt.return
  }
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @load_cant_use_async_cp
// CHECK: scf.for
// CHECK:   tt.load
// CHECK:   "use"
tt.func @load_cant_use_async_cp(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    "use"(%a) {loop.cluster = 2 : i32, loop.stage = 3 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 3 : i32}
  tt.return
}
}
