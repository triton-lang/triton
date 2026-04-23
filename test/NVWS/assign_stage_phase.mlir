// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-assign-stage-phase | FileCheck %s











#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @assign_stage_basic
  tt.func @assign_stage_basic(%lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant 0 : i32
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    // CHECK: [[LOOP:%.*]]:2 = scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // Phase flip BEFORE acquire
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[C1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE]]], [[TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_load [[BUF]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{%.*}}, [[BUF]] {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      %val = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> !elt
      ttg.local_store %val, %view {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @matmul_tma_acc_with_next_iter_if_result_use_d
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false
  // CHECK: [[FOR:%.*]]:5 = scf.for {{.*}} iter_args([[FTOK:%.*]] = %{{.*}}, [[USE_D:%.*]] = %true, [[FSTAGE:%.*]] = %{{.*}}, [[FPF:%.*]] = %{{.*}}, [[FPE:%.*]] = %{{.*}}) -> (!ttg.async.token, i1, i32, i32, i32)
  // CHECK: [[BUF_MMA:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[FSTAGE]]], [[FTOK]] {ttg.partition = array<i32: 1>}
  // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA]][], [[USE_D]], %true {ttg.partition = array<i32: 1>}
  // CHECK: [[FLAG:%.*]] = arith.xori %{{.*}}, %true {ttg.partition = array<i32: 0, 1>} : i1
  // CHECK: scf.if {{.*}} -> (!ttg.async.token, i32, i32, i32)
  // CHECK: [[C1_NEXT:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
  // CHECK: [[NEXT_STAGE_RAW:%.*]] = arith.addi [[FSTAGE]], [[C1_NEXT]] {ttg.partition = array<i32: 0, 1>} : i32
  // CHECK: [[DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
  // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT_STAGE_RAW]], [[DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
  // CHECK: [[ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
  // CHECK: [[NEXT_STAGE:%.*]] = arith.select [[WRAP]], [[ZERO]], [[NEXT_STAGE_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
  // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[NEXT_STAGE]], {{.*}}] {ttg.partition = array<i32: 1>}
  // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[PTOK]], [[NEXT_STAGE]], {{.*}}
  // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %{{.*}}, [[FLAG]], {{.*}}
  tt.func @matmul_tma_acc_with_next_iter_if_result_use_d(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %empty = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %init = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %store = ttng.tmem_store %cst_0, %init[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %3:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token, %arg4 = %true) -> (!ttg.async.token, i1) : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %arg4, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %useD_next = arith.xori %11, %true {ttg.partition = array<i32: 0, 1>} : i1
      %12 = scf.if %11 -> (!ttg.async.token) {
        nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %15 = nvws.semaphore.buffer %full, %token_2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_3, %token_4 = ttng.tmem_load %15[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        %token_6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %token_6 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %12, %useD_next : !ttg.async.token, i1
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 7 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
    tt.return
  }
}








// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @shared_stage_two_semaphores
  tt.func @shared_stage_two_semaphores() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM0:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[SEM1:%.*]] = nvws.semaphore.create %{{.*}} false
    // CHECK: [[S_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant -1 : i32
    %sem0 = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %sem1 = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    // Stage advance: addi/cmpi/select wrapping at 2
    // CHECK: [[STEP:%.*]] = arith.constant 1 : i32
    // CHECK: [[NEXT:%.*]] = arith.addi [[S_INIT]], [[STEP]] : i32
    // CHECK: [[DEPTH:%.*]] = arith.constant 2 : i32
    // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT]], [[DEPTH]] : i32
    // CHECK: [[ZERO:%.*]] = arith.constant 0 : i32
    // CHECK: [[ADV:%.*]] = arith.select [[WRAP]], [[ZERO]], [[NEXT]] : i32
    // Phase flip sem0 (isReleased=true, init=0), then acquire
    // CHECK: [[C1_0:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH0:%.*]] = arith.shli [[C1_0]], [[ADV]] : i32
    // CHECK: [[PE_NEW:%.*]] = arith.xori [[PE_INIT]], [[SH0]] : i32
    // CHECK: [[PE_SHR:%.*]] = arith.shrui [[PE_NEW]], [[ADV]] : i32
    // CHECK: [[PE_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE_BIT:%.*]] = arith.andi [[PE_SHR]], [[PE_C1]] : i32
    // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[SEM0]][[[ADV]], [[PE_BIT]]]
    // Phase flip sem1 (isReleased=false, init=-1), then acquire
    // CHECK: [[C1_1:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH1:%.*]] = arith.shli [[C1_1]], [[ADV]] : i32
    // CHECK: [[PF_NEW:%.*]] = arith.xori [[PF_INIT]], [[SH1]] : i32
    // CHECK: [[PF_SHR:%.*]] = arith.shrui [[PF_NEW]], [[ADV]] : i32
    // CHECK: [[PF_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF_BIT:%.*]] = arith.andi [[PF_SHR]], [[PF_C1]] : i32
    // CHECK: [[TOK1:%.*]] = nvws.semaphore.acquire [[SEM1]][[[ADV]], [[PF_BIT]]]
    %tok0 = nvws.semaphore.acquire %sem0 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %tok1 = nvws.semaphore.acquire %sem1 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token

    // CHECK: [[BUF0:%.*]] = nvws.semaphore.buffer [[SEM0]][[[ADV]]], [[TOK0]]
    // CHECK: ttg.local_store {{%.*}}, [[BUF0]]
    %view0 = nvws.semaphore.buffer %sem0, %tok0 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
    %v = arith.constant dense<0> : !elt
    ttg.local_store %v, %view0 : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>

    // CHECK: nvws.semaphore.release [[SEM0]][[[ADV]]], [[TOK0]] [#nvws.async_op<none>]
    // CHECK: nvws.semaphore.release [[SEM1]][[[ADV]]], [[TOK1]] [#nvws.async_op<none>]
    nvws.semaphore.release %sem0, %tok0 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    nvws.semaphore.release %sem1, %tok1 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @if_observation
  tt.func @if_observation(%cond: i1, %lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant 0 : i32
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    // CHECK: scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // No stage advance here: first use is not provably Store on all paths.
      // Phase flip BEFORE acquire
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[C1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE]]], [[TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_load [[BUF]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{%.*}}, [[BUF]] {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>

      // scf.if with NO results (no stage/phase threading through if)
      scf.if %cond {
        %x = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> !elt
        "use"(%x) {ttg.partition = array<i32: 0>} : (!elt) -> ()
      } {ttg.partition = array<i32: 0>}

      %v = arith.constant {ttg.partition = array<i32: 0>} dense<0> : !elt
      ttg.local_store %v, %view {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @for_body_store_post_loop_load
  tt.func @for_body_store_post_loop_load(%lb: i32, %ub: i32, %step: i32,
                                         %lb1: i32, %ub1: i32, %step1: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant 0 : i32
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    // CHECK: scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // No stage advance here: a store inside the nested loop competes with a
      // later post-loop load of the same buffer lineage.
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[C1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE]]], [[TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: "foo_store"([[BUF]]) {ttg.partition = array<i32: 0>}
      // CHECK: "foo_load"([[BUF]]) {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      scf.for %j = %lb1 to %ub1 step %step1 : i32 {
        "foo_store"(%view) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>) -> ()
      } {ttg.partition = array<i32: 0>}
      "foo_load"(%view) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>) -> ()
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared2d_t = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @view_path_observation
  tt.func @view_path_observation(%lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant 0 : i32
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>

    // CHECK: scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // The first real access of the semaphore-buffer lineage is foo_store on
      // %view, so this advances stage even though a later alias path loads.
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[STAGE_INC:%.*]] = arith.addi [[STAGE]], [[C1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C2:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 2 : i32
      // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[STAGE_INC]], [[C2]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[STAGE_NEW:%.*]] = arith.select [[WRAP]], [[C0]], [[STAGE_INC]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_SHIFT:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1_SHIFT]], [[STAGE_NEW]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE_NEW]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[C1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE_NEW]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE_NEW]]], [[TOK]] {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2>
      %trans = ttg.memdesc_trans %view {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2> -> !ttg.memdesc<2x2xi32, #shared2d_t, #smem, mutable, 2x2x2>
      // CHECK: "foo_store"([[BUF]]) {ttg.partition = array<i32: 0>}
      // CHECK: "foo_load"(%{{.*}}) {ttg.partition = array<i32: 0>}
      "foo_store"(%view) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2>) -> ()
      "foo_load"(%trans) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<2x2xi32, #shared2d_t, #smem, mutable, 2x2x2>) -> ()
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE_NEW]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE_NEW]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @if_view_fallthrough_store
  tt.func @if_view_fallthrough_store(%cond: i1, %lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant 0 : i32
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>

    // CHECK: scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // The branch only creates a view alias; both true and false paths fall
      // through to the same later store, so this still advances stage.
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[STAGE_INC:%.*]] = arith.addi [[STAGE]], [[C1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C2:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 2 : i32
      // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[STAGE_INC]], [[C2]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[STAGE_NEW:%.*]] = arith.select [[WRAP]], [[C0]], [[STAGE_INC]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_SHIFT:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1_SHIFT]], [[STAGE_NEW]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE_NEW]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[C1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE_NEW]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE_NEW]]], [[TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: "foo_store"([[BUF]]) {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2>
      scf.if %cond {
        %trans = ttg.memdesc_trans %view {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2> -> !ttg.memdesc<2x2xi32, #shared2d_t, #smem, mutable, 2x2x2>
      } {ttg.partition = array<i32: 0>}
      "foo_store"(%view) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2>) -> ()
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE_NEW]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE_NEW]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

#shared9 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem9 = #ttg.shared_memory
#tmem9 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales9 = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @scale_a_buffer_read
  tt.func @scale_a_buffer_read(%arg0: !ttg.memdesc<128x64xf16, #shared9, #smem9>, %arg1: !ttg.memdesc<64x128xf16, #shared9, #smem9>, %arg2: !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>, %arg3: !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>) {
    %true = arith.constant true
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY_A:%.*]] = nvws.semaphore.create %arg3 true
    // CHECK: [[FULL_A:%.*]] = nvws.semaphore.create %arg3 false
    %empty = nvws.semaphore.create %arg3 true : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>
    %full = nvws.semaphore.create %arg3 false : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>
    // CHECK: [[TOK_A:%.*]] = nvws.semaphore.acquire [[FULL_A]]
    %tok = nvws.semaphore.acquire %full : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]> -> !ttg.async.token
    // CHECK: [[BUF_A:%.*]] = nvws.semaphore.buffer [[FULL_A]][
    %buf = nvws.semaphore.buffer %full, %tok : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}, [[BUF_A]], {{.*}}, %true, %true lhs = e4m3 rhs = e4m3
    ttng.tc_gen5_mma_scaled %arg0, %arg1, %acc, %buf, %arg2, %true, %true lhs = e4m3 rhs = e4m3 : !ttg.memdesc<128x64xf16, #shared9, #smem9>, !ttg.memdesc<64x128xf16, #shared9, #smem9>, !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>
    // CHECK: nvws.semaphore.release [[EMPTY_A]][
    nvws.semaphore.release %empty, %tok [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @scale_b_buffer_read
  tt.func @scale_b_buffer_read(%arg0: !ttg.memdesc<128x64xf16, #shared9, #smem9>, %arg1: !ttg.memdesc<64x128xf16, #shared9, #smem9>, %arg2: !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>, %arg3: !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>) {
    %true = arith.constant true
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY_B:%.*]] = nvws.semaphore.create %arg3 true
    // CHECK: [[FULL_B:%.*]] = nvws.semaphore.create %arg3 false
    %empty = nvws.semaphore.create %arg3 true : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>
    %full = nvws.semaphore.create %arg3 false : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>
    // CHECK: [[TOK_B:%.*]] = nvws.semaphore.acquire [[FULL_B]]
    %tok = nvws.semaphore.acquire %full : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]> -> !ttg.async.token
    // CHECK: [[BUF_B:%.*]] = nvws.semaphore.buffer [[FULL_B]][
    %buf = nvws.semaphore.buffer %full, %tok : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}, {{.*}}, [[BUF_B]], %true, %true lhs = e4m3 rhs = e4m3
    ttng.tc_gen5_mma_scaled %arg0, %arg1, %acc, %arg2, %buf, %true, %true lhs = e4m3 rhs = e4m3 : !ttg.memdesc<128x64xf16, #shared9, #smem9>, !ttg.memdesc<64x128xf16, #shared9, #smem9>, !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>
    // CHECK: nvws.semaphore.release [[EMPTY_B]][
    nvws.semaphore.release %empty, %tok [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>, !ttg.async.token
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @two_consumers
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %ub = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false
    // CHECK: [[C2_INIT:%.*]] = arith.constant 2 : i32
    // CHECK: [[PE_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant -1 : i32
    %empty = nvws.semaphore.create %0 true : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %0 false : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[LOOP:%.*]]:4 = scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C2_INIT]], [[PE:%.*]] = [[PE_INIT]], {{%.*}} = [[PF_INIT]], {{%.*}} = [[PF_INIT]]) -> (i32, i32, i32, i32)
    scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
      %2 = "op_a"() {ttg.partition = array<i32: 0>} : () -> tensor<1xi32, #blocked>
      // Stage advance: addi/cmpi/select wrapping at 3
      // CHECK: [[STEP:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 1 : i32
      // CHECK: [[NEXT:%.*]] = arith.addi [[STAGE]], [[STEP]] {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: [[C3:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 3 : i32
      // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT]], [[C3]] {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: [[ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 0 : i32
      // CHECK: [[NEW_STAGE:%.*]] = arith.select [[WRAP]], [[ZERO]], [[NEXT]] {ttg.partition = array<i32: 0, 1, 2>} : i32
      // Phase flip EMPTY (isReleased=true, init=0), then acquire
      // CHECK: [[PC1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PSHIFT:%.*]] = arith.shli [[PC1]], [[NEW_STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE_NEW:%.*]] = arith.xori [[PE]], [[PSHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE_SHR:%.*]] = arith.shrui [[PE_NEW]], [[NEW_STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PE_BIT:%.*]] = arith.andi [[PE_SHR]], [[PE_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[NEW_STAGE]], [[PE_BIT]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[NEW_STAGE]]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{%.*}}, [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]][[[NEW_STAGE]]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %token = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %buffers = nvws.semaphore.buffer %empty, %token {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      ttg.local_store %2, %buffers {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.semaphore.release %full, %token [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      // Consumer1: phase flip FULL, then acquire
      // CHECK: [[GC1_1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[GSHIFT1:%.*]] = arith.shli [[GC1_1]], [[NEW_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_NEW:%.*]] = arith.xori [[PF1:%.*]], [[GSHIFT1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_SHR:%.*]] = arith.shrui [[PF1_NEW]], [[NEW_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[PF1_BIT:%.*]] = arith.andi [[PF1_SHR]], [[PF1_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[GTOK1:%.*]] = nvws.semaphore.acquire [[FULL]][[[NEW_STAGE]], [[PF1_BIT]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[GBUF1:%.*]] = nvws.semaphore.buffer [[FULL]][[[NEW_STAGE]]], [[GTOK1]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load [[GBUF1]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]][[[NEW_STAGE]]], [[GTOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      %token_1 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %buffers_0 = nvws.semaphore.buffer %full, %token_1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      %3 = ttg.local_load %buffers_0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.semaphore.release %empty, %token_1 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%3) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()

      // Consumer2: phase flip FULL, then acquire
      // CHECK: [[GC1_2:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // CHECK: [[GSHIFT2:%.*]] = arith.shli [[GC1_2]], [[NEW_STAGE]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[PF2_NEW:%.*]] = arith.xori [[PF2:%.*]], [[GSHIFT2]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[PF2_SHR:%.*]] = arith.shrui [[PF2_NEW]], [[NEW_STAGE]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[PF2_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // CHECK: [[PF2_BIT:%.*]] = arith.andi [[PF2_SHR]], [[PF2_MASK]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[GTOK2:%.*]] = nvws.semaphore.acquire [[FULL]][[[NEW_STAGE]], [[PF2_BIT]]] {ttg.partition = array<i32: 2>}
      // CHECK: [[GBUF2:%.*]] = nvws.semaphore.buffer [[FULL]][[[NEW_STAGE]]], [[GTOK2]] {ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_load [[GBUF2]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[EMPTY]][[[NEW_STAGE]]], [[GTOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      %token_3 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %buffers_2 = nvws.semaphore.buffer %full, %token_3 {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      %4 = ttg.local_load %buffers_2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.semaphore.release %empty, %token_3 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%4) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%4) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()

      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[NEW_STAGE]], [[PE_NEW]], [[PF1_NEW]], [[PF2_NEW]] : i32, i32, i32, i32
    // CHECK-NEXT: } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>, array<i32: 0>, array<i32: 1>, array<i32: 2>], ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}

    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }

}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @semaphore_lowering
  tt.func @semaphore_lowering(%d : !ttg.memdesc<3x64x16xf16, #shared0, #smem>,
                         %e : !ttg.memdesc<3x16x32xf16, #shared0, #smem>,
                         %f : !ttg.memdesc<3x64x16xf16, #shared0, #smem>,
                         %g : !ttg.memdesc<3x16x32xf16, #shared0, #smem>,
                         %cond : i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %lb = arith.constant 0 : i32
    %ub = arith.constant 4 : i32

    // CHECK: [[E0:%.*]] = nvws.semaphore.create {{%.*}} true
    // CHECK: [[F0:%.*]] = nvws.semaphore.create {{%.*}} false
    // CHECK: [[S0_INIT:%.*]] = arith.constant 2 : i32
    // CHECK: [[PE0_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF0_INIT:%.*]] = arith.constant -1 : i32
    // CHECK: [[E1:%.*]] = nvws.semaphore.create {{%.*}} true
    // CHECK: [[F1:%.*]] = nvws.semaphore.create {{%.*}} false
    // CHECK: [[S1_INIT:%.*]] = arith.constant 2 : i32
    // CHECK: [[PE1_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF1_INIT:%.*]] = arith.constant -1 : i32
    %empty0 = nvws.semaphore.create %d, %e true : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    %full0 = nvws.semaphore.create %d, %e false : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    %empty1 = nvws.semaphore.create %f, %g true : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    %full1 = nvws.semaphore.create %f, %g false : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    // CHECK: [[LOOP:%.*]]:6 = scf.for {{.*}} iter_args([[S0:%.*]] = [[S0_INIT]], [[PE0:%.*]] = [[PE0_INIT]], [[PF0:%.*]] = [[PF0_INIT]], [[S1:%.*]] = [[S1_INIT]], [[PE1:%.*]] = [[PE1_INIT]], [[PF1:%.*]] = [[PF1_INIT]]) -> (i32, i32, i32, i32, i32, i32)
    scf.for %i = %lb to %ub step %c1_i32 : i32{
      // Group0 producer: stage advance, then phase flip pe0 BEFORE acquire E0
      // CHECK: [[PC1_P0:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S0_NEXT_RAW:%.*]] = arith.addi [[S0]], [[PC1_P0]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[P0_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // CHECK: [[S0_WRAP:%.*]] = arith.cmpi eq, [[S0_NEXT_RAW]], [[P0_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[P0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[S0_NEXT:%.*]] = arith.select [[S0_WRAP]], [[P0_ZERO]], [[S0_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[PC1_P0B:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PSH0:%.*]] = arith.shli [[PC1_P0B]], [[S0_NEXT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE0_NEW:%.*]] = arith.xori [[PE0]], [[PSH0]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE0_SHR:%.*]] = arith.shrui [[PE0_NEW]], [[S0_NEXT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE0_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PE0_BIT:%.*]] = arith.andi [[PE0_SHR]], [[PE0_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PTOK0:%.*]] = nvws.semaphore.acquire [[E0]][[[S0_NEXT]], [[PE0_BIT]]] {ttg.partition = array<i32: 0>}
      %ptok0 = nvws.semaphore.acquire %empty0 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.async.token
      // CHECK: [[PBUF0:%.*]]:2 = nvws.semaphore.buffer [[E0]][[[S0_NEXT]]], [[PTOK0]] {ttg.partition = array<i32: 0>}
      // CHECK: "op1_store"([[PBUF0]]#0) {ttg.partition = array<i32: 0>}
      // CHECK: "op2_store"([[PBUF0]]#1) {ttg.partition = array<i32: 0>}
      %1:2 = nvws.semaphore.buffer %empty0, %ptok0 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>
      "op1_store"(%1#0) {ttg.partition = array<i32: 0>}: (!ttg.memdesc<64x16xf16, #shared0, #smem, mutable>) -> ()
      "op2_store"(%1#1)  {ttg.partition = array<i32: 0>} : (!ttg.memdesc<16x32xf16, #shared0, #smem, mutable>) -> ()
      // CHECK: nvws.semaphore.release [[F0]][[[S0_NEXT]]], [[PTOK0]] [#nvws.async_op<tma_load>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %full0, %ptok0 [#nvws.async_op<tma_load>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token

      // Group0 consumer: phase flip pf0 BEFORE acquire F0
      // CHECK: [[GC1_C0:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[GSH0:%.*]] = arith.shli [[GC1_C0]], [[S0_NEXT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF0_NEW:%.*]] = arith.xori [[PF0]], [[GSH0]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF0_SHR:%.*]] = arith.shrui [[PF0_NEW]], [[S0_NEXT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF0_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[PF0_BIT:%.*]] = arith.andi [[PF0_SHR]], [[PF0_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[GTOK0:%.*]] = nvws.semaphore.acquire [[F0]][[[S0_NEXT]], [[PF0_BIT]]] {ttg.partition = array<i32: 1>}
      %gtok0 = nvws.semaphore.acquire %full0 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.async.token
      // CHECK: [[GBUF0:%.*]]:2 = nvws.semaphore.buffer [[F0]][[[S0_NEXT]]], [[GTOK0]] {ttg.partition = array<i32: 1>}
      // CHECK: "op3_load"([[GBUF0]]#0, [[GBUF0]]#1) {ttg.partition = array<i32: 1>}
      %2:2 = nvws.semaphore.buffer %full0, %gtok0 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>
      "op3_load"(%2#0, %2#1) {ttg.partition = array<i32: 1>}: (!ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>) -> ()
      // CHECK: nvws.semaphore.release [[E0]][[[S0_NEXT]]], [[GTOK0]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty0, %gtok0 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      // CHECK: [[IFRES:%.*]]:3 = scf.if {{%.*}} -> (i32, i32, i32)
      scf.if %cond {
      } else {
        %ptok1 = nvws.semaphore.acquire %empty1 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.async.token
        %4:2 = nvws.semaphore.buffer %empty1, %ptok1 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>
        "op4_store"(%4#0, %4#1) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>) -> ()
        nvws.semaphore.release %full1, %ptok1 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
        %gtok1 = nvws.semaphore.acquire %full1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.async.token
        %5:2 = nvws.semaphore.buffer %full1, %gtok1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>
        "op5_load"(%5#0, %5#1) {ttg.partition = array<i32: 1>}: (!ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>) -> ()
        nvws.semaphore.release %empty1, %gtok1 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[S1]], [[PE1]], [[PF1]] : i32, i32, i32
      // CHECK: } else {
      // Else branch advances S1, then flips pe1 BEFORE acquire E1.
      // CHECK: [[EC1_P1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S1_NEXT_RAW:%.*]] = arith.addi [[S1]], [[EC1_P1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S1_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // CHECK: [[S1_WRAP:%.*]] = arith.cmpi eq, [[S1_NEXT_RAW]], [[S1_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[S1_NEXT:%.*]] = arith.select [[S1_WRAP]], [[S1_ZERO]], [[S1_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[EC1_P1B:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[ESH_P1:%.*]] = arith.shli [[EC1_P1B]], [[S1_NEXT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE1_NEW:%.*]] = arith.xori [[PE1]], [[ESH_P1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE1_SHR:%.*]] = arith.shrui [[PE1_NEW]], [[S1_NEXT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PE1_BIT:%.*]] = arith.andi [[PE1_SHR]], [[PE1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ETOK_P1:%.*]] = nvws.semaphore.acquire [[E1]][[[S1_NEXT]], [[PE1_BIT]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[EBUF_P1:%.*]]:2 = nvws.semaphore.buffer [[E1]][[[S1_NEXT]]], [[ETOK_P1]] {ttg.partition = array<i32: 0>}
      // CHECK: "op4_store"([[EBUF_P1]]#0, [[EBUF_P1]]#1) {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[F1]][[[S1_NEXT]]], [[ETOK_P1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // Phase flip pf1 BEFORE acquire F1
      // CHECK: [[FC1_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[FSH_C1:%.*]] = arith.shli [[FC1_C1]], [[S1_NEXT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_NEW:%.*]] = arith.xori [[PF1]], [[FSH_C1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_SHR:%.*]] = arith.shrui [[PF1_NEW]], [[S1_NEXT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[PF1_BIT:%.*]] = arith.andi [[PF1_SHR]], [[PF1_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[FTOK_C1:%.*]] = nvws.semaphore.acquire [[F1]][[[S1_NEXT]], [[PF1_BIT]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[FBUF_C1:%.*]]:2 = nvws.semaphore.buffer [[F1]][[[S1_NEXT]]], [[FTOK_C1]] {ttg.partition = array<i32: 1>}
      // CHECK: "op5_load"([[FBUF_C1]]#0, [[FBUF_C1]]#1) {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[E1]][[[S1_NEXT]]], [[FTOK_C1]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[S1_NEXT]], [[PE1_NEW]], [[PF1_NEW]] : i32, i32, i32
      // CHECK-NEXT: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>, array<i32: 0>, array<i32: 1>]}
      } {ttg.partition = array<i32: 0, 1>}

      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[S0_NEXT]], [[PE0_NEW]], [[PF0_NEW]], [[IFRES]]#0, [[IFRES]]#1, [[IFRES]]#2 : i32, i32, i32, i32, i32, i32
    // CHECK-NEXT: } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1>, array<i32: 0>, array<i32: 1>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    } {ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----


#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @warp_specialize_tma_matmul
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared>, %arg4: !tt.tensordesc<128x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false
    // CHECK: [[S_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PE_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant 1 : i32
    // Pre-loop: stage advance, single-phase flip, acquire EMPTY
    // CHECK: [[SA_STEP:%.*]] = arith.constant 1 : i32
    // CHECK: [[SA_ADD:%.*]] = arith.addi [[S_INIT]], [[SA_STEP]] : i32
    // CHECK: [[SA_DEPTH:%.*]] = arith.constant 1 : i32
    // CHECK: [[SA_CMP:%.*]] = arith.cmpi eq, [[SA_ADD]], [[SA_DEPTH]] : i32
    // CHECK: [[SA_ZERO:%.*]] = arith.constant 0 : i32
    // CHECK: [[PRE_STAGE:%.*]] = arith.select [[SA_CMP]], [[SA_ZERO]], [[SA_ADD]] : i32
    // CHECK: [[PE_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE_FLIP:%.*]] = arith.xori [[PE_INIT]], [[PE_C1]] : i32
    // CHECK: [[PE_CMP:%.*]] = arith.constant 0 : i32
    // CHECK: [[PE_EQ:%.*]] = arith.cmpi eq, [[PRE_STAGE]], [[PE_CMP]] : i32
    // CHECK: [[PE_PRE:%.*]] = arith.select [[PE_EQ]], [[PE_FLIP]], [[PE_INIT]] : i32
    // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[PRE_STAGE]], [[PE_PRE]]]
    // CHECK: [[BUF_INIT:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[PRE_STAGE]]], [[TOK0]]
    // CHECK: ttng.tmem_store {{%.*}}, [[BUF_INIT]][], {{%.*}}
    %empty = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %1 = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %2 = ttng.tmem_store %cst, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: scf.for {{.*}} : i32 {
    scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32  : i32 {
      %4 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      // CHECK: tt.descriptor_load {{.*}} {ttg.partition = array<i32: 2>}
      // CHECK: tt.descriptor_load {{.*}} {ttg.partition = array<i32: 2>}
      %5 = tt.descriptor_load %arg3[%arg1, %4] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg4[%arg2, %4] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %9 = ttg.memdesc_trans %8 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // CHECK: [[BUF_MMA:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[PRE_STAGE]]], [[TOK0]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      %10 = nvws.semaphore.buffer %empty, %token {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %11 = ttng.tc_gen5_mma %7, %9, %10[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // Post-loop: release FULL, single-phase flip pf, acquire FULL, tmem_load, release EMPTY
    // CHECK: nvws.semaphore.release [[FULL]][[[PRE_STAGE]]], [[TOK0]] [#nvws.async_op<tc5mma>]
    nvws.semaphore.release %full, %token [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[PF_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF_FLIP:%.*]] = arith.xori [[PF_INIT]], [[PF_C1]] : i32
    // CHECK: [[PF_CMP:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF_EQ:%.*]] = arith.cmpi eq, [[PRE_STAGE]], [[PF_CMP]] : i32
    // CHECK: [[PF_POST:%.*]] = arith.select [[PF_EQ]], [[PF_FLIP]], [[PF_INIT]] : i32
    // CHECK: [[TOK1:%.*]] = nvws.semaphore.acquire [[FULL]][[[PRE_STAGE]], [[PF_POST]]]
    // CHECK: [[BUF_POST:%.*]] = nvws.semaphore.buffer [[FULL]][[[PRE_STAGE]]], [[TOK1]]
    // CHECK: ttng.tmem_load [[BUF_POST]][]
    %token_1 = nvws.semaphore.acquire %full : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %full, %token_1 : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_2, %token_3 = ttng.tmem_load %3[] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    // CHECK: nvws.semaphore.release [[EMPTY]][[[PRE_STAGE]]], [[TOK1]] [#nvws.async_op<none>]
    nvws.semaphore.release %empty, %token_1 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%result_2) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_unconditional_user
  tt.func @matmul_tma_acc_with_unconditional_user(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false
    // Pre-loop: stage advance, phase flips, acquire EMPTY
    // CHECK: [[S_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[SA_STEP:%.*]] = arith.constant 1 : i32
    // CHECK: [[SA_ADD:%.*]] = arith.addi [[S_INIT]], [[SA_STEP]] : i32
    // CHECK: [[SA_DEPTH:%.*]] = arith.constant 2 : i32
    // CHECK: [[SA_CMP:%.*]] = arith.cmpi eq, [[SA_ADD]], [[SA_DEPTH]] : i32
    // CHECK: [[SA_ZERO:%.*]] = arith.constant 0 : i32
    // CHECK: [[PRE_STAGE:%.*]] = arith.select [[SA_CMP]], [[SA_ZERO]], [[SA_ADD]] : i32
    // CHECK: [[PF1_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF1_FLIP:%.*]] = arith.xori [[PE_INIT]], [[PF1_C1]] : i32
    // CHECK: [[PF1_CMP:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF1_EQ:%.*]] = arith.cmpi eq, [[PRE_STAGE]], [[PF1_CMP]] : i32
    // CHECK: [[PF1_OUT:%.*]] = arith.select [[PF1_EQ]], [[PF1_FLIP]], [[PE_INIT]] : i32
    // CHECK: [[PE1_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE1_FLIP:%.*]] = arith.xori [[PE_INIT]], [[PE1_C1]] : i32
    // CHECK: [[PE1_CMP:%.*]] = arith.constant 0 : i32
    // CHECK: [[PE1_EQ:%.*]] = arith.cmpi eq, [[PRE_STAGE]], [[PE1_CMP]] : i32
    // CHECK: [[PE_PRE:%.*]] = arith.select [[PE1_EQ]], [[PE1_FLIP]], [[PE_INIT]] : i32
    // CHECK: [[PRETOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[PRE_STAGE]], [[PE_PRE]]]
    // CHECK: [[BUF_INIT:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[PRE_STAGE]]], [[PRETOK]]
    // CHECK: ttng.tmem_store {{%.*}}, [[BUF_INIT]][], {{%.*}}
    %empty = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %1 = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %2 = ttng.tmem_store %cst_0, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[FOR:%.*]]:4 = scf.for {{.*}} iter_args([[FTOK:%.*]] = [[PRETOK]], [[FSTAGE:%.*]] = [[PRE_STAGE]], {{%.*}}, {{%.*}}) -> (!ttg.async.token, i32, i32, i32)
    %3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token) -> (!ttg.async.token)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[BUF_MMA:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[FSTAGE]]], [[FTOK]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: nvws.semaphore.release [[FULL]][[[FSTAGE]]], [[FTOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Consumer: phase flip BEFORE acquire FULL
      // CHECK: [[GC1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[GFLIP:%.*]] = arith.xori [[FPF:%.*]], [[GC1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[GCMP:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[GEQ:%.*]] = arith.cmpi eq, [[FSTAGE]], [[GCMP]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[FPF_NEW:%.*]] = arith.select [[GEQ]], [[GFLIP]], [[FPF]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[FSTAGE]], [[FPF_NEW]]] {ttg.partition = array<i32: 0>}
      %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %11 = nvws.semaphore.buffer %full, %token_2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[BUF_LOAD:%.*]] = nvws.semaphore.buffer [[FULL]][[[FSTAGE]]], [[GTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_load [[BUF_LOAD]][] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[EMPTY]][[[FSTAGE]]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_3, %token_4 = ttng.tmem_load %11[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      // Stage advance, phase flip, re-acquire EMPTY
      // CHECK: [[NSA_STEP:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[NSA_ADD:%.*]] = arith.addi [[FSTAGE]], [[NSA_STEP]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[NSA_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
      // CHECK: [[NSA_CMP:%.*]] = arith.cmpi eq, [[NSA_ADD]], [[NSA_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[NSA_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[NEXT_STAGE:%.*]] = arith.select [[NSA_CMP]], [[NSA_ZERO]], [[NSA_ADD]] {ttg.partition = array<i32: 0, 1>}
      // CHECK: [[PC1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[PFLIP:%.*]] = arith.xori [[FPE:%.*]], [[PC1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PCMP:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[PEQ:%.*]] = arith.cmpi eq, [[NEXT_STAGE]], [[PCMP]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[FPE_NEW:%.*]] = arith.select [[PEQ]], [[PFLIP]], [[FPE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[NEXT_STAGE]], [[FPE_NEW]]] {ttg.partition = array<i32: 1>}
      %token_6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %12 = nvws.semaphore.buffer %empty, %token_6 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[BUF_REINIT:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[NEXT_STAGE]]], [[PTOK]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tmem_store {{%.*}}, [[BUF_REINIT]][], {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[PTOK]], [[NEXT_STAGE]], [[FPF_NEW]], [[FPE_NEW]]
      %13 = ttng.tmem_store %cst, %12[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      scf.yield %token_6 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32}
    // CHECK: nvws.semaphore.release [[FULL]][[[FOR]]#1], [[FOR]]#0 [#nvws.async_op<none>]
    nvws.semaphore.release %full, %3 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @matmul_tma_acc_with_conditional_next_iter_user
  tt.func @matmul_tma_acc_with_conditional_next_iter_user(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false
    %empty = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %init = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %store = ttng.tmem_store %cst_0, %init[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[FOR:%.*]]:5 = scf.for {{.*}} iter_args([[FTOK:%.*]] = %{{.*}}, [[USE_D:%.*]] = %true, [[FSTAGE:%.*]] = %{{.*}}, [[FPF:%.*]] = %{{.*}}, [[FPE:%.*]] = %{{.*}}) -> (!ttg.async.token, i1, i32, i32, i32)
    %3:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token, %arg4 = %true) -> (!ttg.async.token, i1)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[BUF_MMA:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[FSTAGE]]], [[FTOK]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA]][], [[USE_D]], %true {ttg.partition = array<i32: 1>}
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %arg4, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %12 = scf.if %11 -> (!ttg.async.token) {
        nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %15 = nvws.semaphore.buffer %full, %token_2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_3, %token_4 = ttng.tmem_load %15[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // Stage advance + phase flip BEFORE re-acquire EMPTY.
        // CHECK: [[PC1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
        // CHECK: [[NEXT_STAGE_RAW:%.*]] = arith.addi [[FSTAGE]], [[PC1]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
        // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT_STAGE_RAW]], [[DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
        // CHECK: [[NEXT_STAGE:%.*]] = arith.select [[WRAP]], [[ZERO]], [[NEXT_STAGE_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[PFLIP:%.*]] = arith.xori [[FPE]], {{%.*}} {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PEQ:%.*]] = arith.cmpi eq, [[NEXT_STAGE]], {{%.*}} {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PP_OUT:%.*]] = arith.select [[PEQ]], [[PFLIP]], [[FPE]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[NEXT_STAGE]], [[PP_OUT]]] {ttg.partition = array<i32: 1>}
        %token_6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[PTOK]], [[NEXT_STAGE]], {{%.*}}, [[PP_OUT]]
        scf.yield %token_6 : !ttg.async.token
      } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[FTOK]], [[FSTAGE]], [[FPF]], [[FPE]]
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %13 = arith.xori %11, %true {ttg.partition = array<i32: 0, 1>} : i1
      // CHECK: [[FLAG:%.*]] = arith.xori %{{.*}}, %true {ttg.partition = array<i32: 0, 1>} : i1
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[IF:%.*]]#0, [[FLAG]], [[IF]]#1, [[IF]]#2, [[IF]]#3
      scf.yield %12, %13 : !ttg.async.token, i1
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 6 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_next_iter_and_post_loop_read
  // CHECK: [[EMPTY2:%.*]] = nvws.semaphore.create %{{.*}} true
  // CHECK: [[FULL2:%.*]] = nvws.semaphore.create %{{.*}} false
  // CHECK: [[FOR2:%.*]]:5 = scf.for {{.*}} iter_args([[FTOK2:%.*]] = %{{.*}}, {{%.*}} = %true, [[FSTAGE2:%.*]] = %{{.*}}, {{%.*}} = %{{.*}}, {{%.*}} = %{{.*}}) -> (!ttg.async.token, i1, i32, i32, i32)
  // CHECK: [[BUF_MMA2:%.*]] = nvws.semaphore.buffer [[EMPTY2]][[[FSTAGE2]]], [[FTOK2]] {ttg.partition = array<i32: 1>}
  // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA2]][], {{%.*}}, %true {ttg.partition = array<i32: 1>}
  // CHECK: [[PTOK2:%.*]] = nvws.semaphore.acquire [[EMPTY2]][[[FSTAGE2]], {{%.*}}] {ttg.partition = array<i32: 1>}
  // CHECK: [[BUF_POST2:%.*]] = nvws.semaphore.buffer [[EMPTY2]][[[FOR2]]#2], [[FOR2]]#0 {ttg.partition = array<i32: 1>}
  // CHECK: ttng.tmem_load [[BUF_POST2]][] {ttg.partition = array<i32: 1>}
  tt.func @matmul_tma_acc_with_conditional_next_iter_and_post_loop_read(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %empty = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %init = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %store = ttng.tmem_store %cst_0, %init[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %3:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token, %arg4 = %true) -> (!ttg.async.token, i1)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %arg4, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %12 = scf.if %11 -> (!ttg.async.token) {
        nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %15 = nvws.semaphore.buffer %full, %token_2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_3, %token_4 = ttng.tmem_load %15[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        %token_6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield %token_6 : !ttg.async.token
      } else {
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %13 = arith.xori %11, %true {ttg.partition = array<i32: 0, 1>} : i1
      scf.yield %12, %13 : !ttg.async.token, i1
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 7 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
    %post = nvws.semaphore.buffer %empty, %3#0 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %loaded, %tok_end = ttng.tmem_load %post[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @attention_forward
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<64x64xf16, #shared>, %arg2: !tt.tensordesc<64x64xf16, #shared>, %arg3: f32, %arg4: i32) {
    %cst = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-DAG: [[E0:%.*]] = nvws.semaphore.create [[TMEM0:%.*]] true
    // CHECK-DAG: [[F0:%.*]] = nvws.semaphore.create [[TMEM0]] false
    // CHECK-DAG: [[E1:%.*]] = nvws.semaphore.create [[TMEM1:%.*]] true
    // CHECK-DAG: [[F1:%.*]] = nvws.semaphore.create [[TMEM1]] false
    // CHECK-DAG: [[E2:%.*]] = nvws.semaphore.create [[TMEM2:%.*]] true
    // CHECK-DAG: [[F2:%.*]] = nvws.semaphore.create [[TMEM2]] false
    %empty0 = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full0 = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty0 : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %empty1 = nvws.semaphore.create %result_2 true : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full1 = nvws.semaphore.create %result_2 false : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token_4 = nvws.semaphore.acquire %empty1 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %2 = nvws.semaphore.buffer %empty1, %token_4 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %3 = ttng.tmem_store %cst_0, %2[], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %result_5 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>
    %empty2 = nvws.semaphore.create %result_5 true : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %full2 = nvws.semaphore.create %result_5 false : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[LOOP:%.*]]:12 = scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[DATA0:%.*]] = [[INIT0:%.*]], [[DATA1:%.*]] = [[INIT1:%.*]], [[TOK0:%.*]] = [[TOK0_INIT:%.*]], [[TOK1:%.*]] = [[TOK1_INIT:%.*]], [[S0_S:%.*]] = [[S0_S_INIT:%.*]], [[S0_PF:%.*]] = [[S0_PF_INIT:%.*]], [[S0_PE:%.*]] = [[S0_PE_INIT:%.*]], [[S1_PF:%.*]] = [[S1_PF_INIT:%.*]], [[S1_PE:%.*]] = [[S1_PE_INIT:%.*]], [[S2_S:%.*]] = [[S2_S_INIT:%.*]], [[S2_PE:%.*]] = [[S2_PE_INIT:%.*]], [[S2_PF:%.*]] = [[S2_PF_INIT:%.*]]) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32, i32, i32)
    %5:4 = scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg6 = %cst, %arg7 = %cst_1, %arg8 = %token, %arg9 = %token_4) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      // CHECK: [[DESCLD1:%.*]] = tt.descriptor_load [[ARG1:%.*]][[[IV]], [[C0_LD:%.*]]] {ttg.partition = array<i32: 2>}
      %7 = tt.descriptor_load %arg1[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked1>
      %8 = ttg.local_alloc %7 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %9 = ttg.memdesc_trans %8 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      // CHECK: [[BUF_E0:%.*]] = nvws.semaphore.buffer [[E0]][[[S0_S]]], [[TOK0]] {ttg.partition = array<i32: 1>}
      %10 = nvws.semaphore.buffer %empty0, %arg8 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      // CHECK: [[MMA1:%.*]] = ttng.tc_gen5_mma [[ARG0:%.*]], [[TRANS:%.*]], [[BUF_E0]][], [[FALSE:%.*]], [[TRUE:%.*]] {ttg.partition = array<i32: 1>}
      %11 = ttng.tc_gen5_mma %arg0, %9, %10[], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      // CHECK: nvws.semaphore.release [[F0]][[[S0_S]]], [[TOK0]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %full0, %arg8 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Single-phase flip BEFORE acquire F0
      // CHECK: [[S0_PF_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[S0_PF_FLIP:%.*]] = arith.xori [[S0_PF]], [[S0_PF_C1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[S0_PF_CMP:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[S0_PF_EQ:%.*]] = arith.cmpi eq, [[S0_S]], [[S0_PF_CMP]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[S0_PF_NEW:%.*]] = arith.select [[S0_PF_EQ]], [[S0_PF_FLIP]], [[S0_PF]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[GTOK0:%.*]] = nvws.semaphore.acquire [[F0]][[[S0_S]], [[S0_PF_NEW]]] {ttg.partition = array<i32: 0>}
      %token_11 = nvws.semaphore.acquire %full0 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %12 = nvws.semaphore.buffer %full0, %token_11 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      // CHECK: [[BUF_F0:%.*]] = nvws.semaphore.buffer [[F0]][[[S0_S]]], [[GTOK0]] {ttg.partition = array<i32: 0>}
      // CHECK: [[TLOAD0:%.*]], [[TLTOK0:%.*]] = ttng.tmem_load [[BUF_F0]][] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[E0]][[[S0_S]]], [[GTOK0]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_12, %token_13 = ttng.tmem_load %12[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64> -> tensor<256x64xf32, #blocked>
      nvws.semaphore.release %empty0, %token_11 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %13 = "compute_row_max"(%result_12, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = "sub_row_max"(%result_12, %13, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %15 = math.exp2 %14 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked>
      %16 = arith.subf %arg7, %13 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = arith.subf %arg7, %13 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %18 = math.exp2 %16 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %19 = math.exp2 %17 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %20 = "tt.reduce"(%15) <{axis = 1 : i32}> ({
      ^bb0(%arg10: f32, %arg11: f32):
        %36 = arith.addf %arg10, %arg11 {ttg.partition = array<i32: 0>}: f32
        tt.reduce.return %36 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %21 = arith.mulf %arg6, %19 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %22 = arith.addf %21, %20 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %23 = tt.expand_dims %18 {axis = 1 : i32, ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %24 = tt.broadcast %23 {ttg.partition = array<i32: 3>} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      // CHECK: [[BUF_E1:%.*]] = nvws.semaphore.buffer [[E1]][{{.*}}], [[TOK1]] {ttg.partition = array<i32: 3>}
      %25 = nvws.semaphore.buffer %empty1, %arg9 {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: ttng.tmem_load [[BUF_E1]][] {ttg.partition = array<i32: 3>}
      %result_14, %token_15 = ttng.tmem_load %25[] {ttg.partition = array<i32: 3>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
      %26 = arith.mulf %result_14, %24 {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked>
      %27 = tt.descriptor_load %arg2[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked1>
      %28 = ttg.local_alloc %27 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %29 = arith.truncf %15 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      %token_17 = nvws.semaphore.acquire %empty2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %30 = nvws.semaphore.buffer %empty2, %token_17 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // Phase flip BEFORE acquire E2
      // CHECK: [[S2_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S2_NEXT_RAW:%.*]] = arith.addi [[S2_S]], [[S2_C1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S2_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S2_WRAP:%.*]] = arith.cmpi eq, [[S2_NEXT_RAW]], [[S2_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S2_C0:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[S2_STAGE:%.*]] = arith.select [[S2_WRAP]], [[S2_C0]], [[S2_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S2_PE_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[S2_PE_FLIP:%.*]] = arith.xori [[S2_PE]], [[S2_PE_C1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[S2_PE_CMP:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[S2_PE_EQ:%.*]] = arith.cmpi eq, [[S2_STAGE]], [[S2_PE_CMP]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[S2_PE_NEW:%.*]] = arith.select [[S2_PE_EQ]], [[S2_PE_FLIP]], [[S2_PE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ATOK_E2:%.*]] = nvws.semaphore.acquire [[E2]][[[S2_STAGE]], [[S2_PE_NEW]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[BUF_E2:%.*]] = nvws.semaphore.buffer [[E2]][[[S2_STAGE]]], [[ATOK_E2]] {ttg.partition = array<i32: 0>}
      // CHECK: [[TSTORE:%.*]] = ttng.tmem_store [[TRUNCF:%.*]], [[BUF_E2]][[[ATOK_E2]]], [[TRUE]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[F2]][[[S2_STAGE]]], [[ATOK_E2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %31 = ttng.tmem_store %29, %30[%token_17], %true {ttg.partition = array<i32: 0>} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %full2, %token_17 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %32 = ttng.tmem_store %26, %25[], %true {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: nvws.semaphore.release [[F1]][{{.*}}], [[TOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 3>}
      nvws.semaphore.release %full1, %arg9 [#nvws.async_op<none>] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // S1 phase_f flip without stage advance
      // CHECK: [[S1_PF_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S1_PF_FLIP:%.*]] = arith.xori [[S1_PF]], [[S1_PF_C1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S1_PF_CMP:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[S1_PF_EQ:%.*]] = arith.cmpi eq, {{.*}}, [[S1_PF_CMP]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S1_PF_NEW:%.*]] = arith.select [[S1_PF_EQ]], [[S1_PF_FLIP]], [[S1_PF]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ATOK_F1:%.*]] = nvws.semaphore.acquire [[F1]][{{.*}}, [[S1_PF_NEW]]] {ttg.partition = array<i32: 1>}
      %token_19 = nvws.semaphore.acquire %full1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %33 = nvws.semaphore.buffer %full1, %token_19 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: [[BUF_F1:%.*]] = nvws.semaphore.buffer [[F1]][{{.*}}], [[ATOK_F1]] {ttg.partition = array<i32: 1>}
      // S2 phase_f flip before acquire F2
      // CHECK: [[S2_PF_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[S2_PF_FLIP:%.*]] = arith.xori [[S2_PF]], [[S2_PF_C1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S2_PF_CMP:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[S2_PF_EQ:%.*]] = arith.cmpi eq, [[S2_STAGE]], [[S2_PF_CMP]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S2_PF_NEW:%.*]] = arith.select [[S2_PF_EQ]], [[S2_PF_FLIP]], [[S2_PF]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK_F2:%.*]] = nvws.semaphore.acquire [[F2]][[[S2_STAGE]], [[S2_PF_NEW]]] {ttg.partition = array<i32: 1>}
      %token_21 = nvws.semaphore.acquire %full2 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %34 = nvws.semaphore.buffer %full2, %token_21 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: [[BUF_F2:%.*]] = nvws.semaphore.buffer [[F2]][[[S2_STAGE]]], [[ATOK_F2]] {ttg.partition = array<i32: 1>}
      // CHECK: [[MMA2:%.*]] = ttng.tc_gen5_mma [[BUF_F2]], [[ALLOC2:%.*]], [[BUF_F1]][], [[TRUE]], [[TRUE]] {ttg.partition = array<i32: 1>}
      %35 = ttng.tc_gen5_mma %34, %28, %33[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: nvws.semaphore.release [[E2]][[[S2_STAGE]]], [[ATOK_F2]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[E1]][{{.*}}], [[ATOK_F1]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty2, %token_21 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      nvws.semaphore.release %empty1, %token_19 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // S0 stage advance
      // CHECK: [[S0_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S0_NEXT:%.*]] = arith.addi [[S0_S]], [[S0_C1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S0_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
      // CHECK: [[S0_WRAP:%.*]] = arith.cmpi eq, [[S0_NEXT]], [[S0_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S0_C0:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[S0_STAGE:%.*]] = arith.select [[S0_WRAP]], [[S0_C0]], [[S0_NEXT]] {ttg.partition = array<i32: 0, 1>} : i32
      // Phase flip before acquire E0
      // CHECK: [[S0_PE_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[S0_PE_FLIP:%.*]] = arith.xori [[S0_PE]], [[S0_PE_C1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S0_PE_CMP:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[S0_PE_EQ:%.*]] = arith.cmpi eq, [[S0_STAGE]], [[S0_PE_CMP]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S0_PE_NEW:%.*]] = arith.select [[S0_PE_EQ]], [[S0_PE_FLIP]], [[S0_PE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK_E0:%.*]] = nvws.semaphore.acquire [[E0]][[[S0_STAGE]], [[S0_PE_NEW]]] {ttg.partition = array<i32: 1>}
      // Phase flip before acquire E1
      // CHECK: [[S1_PE_C1:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 1 : i32
      // CHECK: [[S1_PE_FLIP:%.*]] = arith.xori [[S1_PE]], [[S1_PE_C1]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[S1_PE_CMP:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[S1_PE_EQ:%.*]] = arith.cmpi eq, {{.*}}, [[S1_PE_CMP]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[S1_PE_NEW:%.*]] = arith.select [[S1_PE_EQ]], [[S1_PE_FLIP]], [[S1_PE]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[ATOK_E1:%.*]] = nvws.semaphore.acquire [[E1]][{{.*}}, [[S1_PE_NEW]]] {ttg.partition = array<i32: 3>}
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} [[YIELD_D0:%.*]], [[YIELD_D1:%.*]], [[ATOK_E0]], [[ATOK_E1]], [[S0_STAGE]], [[S0_PF_NEW]], [[S0_PE_NEW]], [[S1_PF_NEW]], [[S1_PE_NEW]], [[S2_STAGE]], [[S2_PE_NEW]], [[S2_PF_NEW]]
      %token_23 = nvws.semaphore.acquire %empty0 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %token_25 = nvws.semaphore.acquire %empty1 {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      scf.yield %22, %13, %token_23, %token_25 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 3>]}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 3>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>, array<i32: 0, 1>, array<i32: 3>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>]
    // CHECK: nvws.semaphore.release [[F1]][{{.*}}], [[LOOP]]#3 [#nvws.async_op<tc5mma>]
    // CHECK: nvws.semaphore.release [[F0]][[[LOOP]]#4], [[LOOP]]#2 [#nvws.async_op<none>]
    nvws.semaphore.release %full1, %5#3 [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.semaphore.release %full0, %5#2 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // Phase flip BEFORE acquire F1
    // CHECK: [[TOK_END:%.*]] = nvws.semaphore.acquire [[F1]]
    // CHECK: [[BUF_F1_END:%.*]] = nvws.semaphore.buffer [[F1]][{{.*}}], [[TOK_END]]
    // CHECK: ttng.tmem_load [[BUF_F1_END]][]
    // CHECK: nvws.semaphore.release [[E1]][{{.*}}], [[TOK_END]] [#nvws.async_op<none>]
    %token_7 = nvws.semaphore.acquire %full1 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %6 = nvws.semaphore.buffer %full1, %token_7 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %result_8, %token_9 = ttng.tmem_load %6[] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
    nvws.semaphore.release %empty1, %token_7 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%5#0, %result_8, %5#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @matmul_tma_acc_with_conditional_user
    tt.func @matmul_tma_acc_with_conditional_user(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false
    // CHECK: %{{.*}} = nvws.semaphore.acquire [[EMPTY]][{{%.*}}, {{%.*}}]
    // CHECK: %{{.*}} = nvws.semaphore.buffer [[EMPTY]][{{%.*}}], %{{.*}}
    // CHECK: ttng.tmem_store {{%.*}}, %{{.*}}[], {{%.*}}
    %empty = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %1 = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %2 = ttng.tmem_store %cst_0, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[FOR:%.*]]:4 = scf.for {{.*}} iter_args([[FTOK:%.*]] = %{{.*}}, [[FSTAGE:%.*]] = %{{.*}}, [[FPF:%.*]] = %{{.*}}, [[FPE:%.*]] = %{{.*}}) -> (!ttg.async.token, i32, i32, i32)
    %3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token) -> (!ttg.async.token)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[BUF_MMA:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[FSTAGE]]], [[FTOK]] {ttg.partition = array<i32: 1>}
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[IF:%.*]]:4 = scf.if
      %12 = scf.if %11 -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[FULL]][[[FSTAGE]]], [[FTOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
        nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // Phase flip BEFORE acquire FULL
        // CHECK: [[GC1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
        // CHECK: [[GP_OUT:%.*]] = arith.xori [[FPF:%.*]], [[GC1]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[GC0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
        // CHECK: [[G_EQ:%.*]] = arith.cmpi eq, [[FSTAGE]], [[GC0]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[GP_SEL:%.*]] = arith.select [[G_EQ]], [[GP_OUT]], [[FPF]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[FSTAGE]], [[GP_SEL]]] {ttg.partition = array<i32: 0>}
        %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %15 = nvws.semaphore.buffer %full, %token_2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: [[BUF_LOAD:%.*]] = nvws.semaphore.buffer [[FULL]][[[FSTAGE]]], [[GTOK]] {ttg.partition = array<i32: 0>}
        // CHECK: ttng.tmem_load [[BUF_LOAD]][] {ttg.partition = array<i32: 0>}
        // CHECK: nvws.semaphore.release [[EMPTY]][[[FSTAGE]]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
        %result_3, %token_4 = ttng.tmem_load %15[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // Stage advance + phase flip BEFORE re-acquire EMPTY.
        // CHECK: [[PC1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
        // CHECK: [[NEXT_STAGE_RAW:%.*]] = arith.addi [[FSTAGE]], [[PC1]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
        // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT_STAGE_RAW]], [[DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
        // CHECK: [[NEXT_STAGE:%.*]] = arith.select [[WRAP]], [[ZERO]], [[NEXT_STAGE_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[PFLIP:%.*]] = arith.xori [[FPE]], {{%.*}} {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PEQ:%.*]] = arith.cmpi eq, [[NEXT_STAGE]], {{%.*}} {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PP_OUT:%.*]] = arith.select [[PEQ]], [[PFLIP]], [[FPE]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[NEXT_STAGE]], [[PP_OUT]]] {ttg.partition = array<i32: 1>}
        %token_6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[PTOK]], [[NEXT_STAGE]], [[GP_SEL]], [[PP_OUT]]
        scf.yield %token_6 : !ttg.async.token
      } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[FTOK]], [[FSTAGE]], [[FPF]], [[FPE]]
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>]}
      // CHECK: [[BUF_POST:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[IF]]#1], [[IF]]#0 {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tmem_store {{%.*}}, [[BUF_POST]][], {{%.*}} {ttg.partition = array<i32: 1>}
      %13 = nvws.semaphore.buffer %empty, %12 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %14 = ttng.tmem_store %cst, %13[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[IF]]#0, [[IF]]#1, [[IF]]#2, [[IF]]#3
      scf.yield %12 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32}
    // CHECK: nvws.semaphore.release [[FULL]][[[FOR]]#1], [[FOR]]#0 [#nvws.async_op<none>]
    nvws.semaphore.release %full, %3 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @matmul_tma_persistent_ws_kernel
  tt.func public @matmul_tma_persistent_ws_kernel(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg6, %arg8], [%0, %c1_i64] : <f8E4M3FN>, <128x128xf8E4M3FN, #shared>
    %2 = arith.extsi %arg4 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg1, [%arg7, %arg8], [%2, %c1_i64] : <f8E4M3FN>, <128x128xf8E4M3FN, #shared>
    %4 = arith.extsi %arg5 : i32 to i64
    %5 = tt.make_tensor_descriptor %arg2, [%arg6, %arg7], [%4, %c1_i64] : <f8E4M3FN>, <128x128xf8E4M3FN, #shared>
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg6, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg7, %c127_i32 : i32
    %10 = arith.divsi %9, %c128_i32 : i32
    %11 = arith.addi %arg8, %c127_i32 : i32
    %12 = arith.divsi %11, %c128_i32 : i32
    %13 = arith.muli %8, %10 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    %15 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>
    %ab_empty = nvws.semaphore.create %15, %16 true : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>
    %ab_full = nvws.semaphore.create %15, %16 false : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-DAG: [[AB_EMPTY:%.*]] = nvws.semaphore.create [[AB_BUF0:%.*]], [[AB_BUF1:%.*]] true
    // CHECK-DAG: [[AB_FULL:%.*]] = nvws.semaphore.create [[AB_BUF2:%.*]], [[AB_BUF3:%.*]] false
    // CHECK-DAG: [[ACC_EMPTY:%.*]] = nvws.semaphore.create [[ACC_BUF0:%.*]] true
    // CHECK-DAG: [[ACC_FULL:%.*]] = nvws.semaphore.create [[ACC_BUF1:%.*]] false
    %empty_acc = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full_acc = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[OUTER:%.*]]:6 = scf.for [[OUTER_IV:%.*]] = [[OUTER_LB:%.*]] to [[OUTER_UB:%.*]] step [[OUTER_STEP:%.*]] iter_args([[AB_S:%.*]] = [[AB_S_INIT:%.*]], [[AB_PF:%.*]] = [[AB_PF_INIT:%.*]], [[AB_PE:%.*]] = [[AB_PE_INIT:%.*]], [[ACC_S:%.*]] = [[ACC_S_INIT:%.*]], [[ACC_PE:%.*]] = [[ACC_PE_INIT:%.*]], [[ACC_PF:%.*]] = [[ACC_PF_INIT:%.*]]) -> (i32, i32, i32, i32, i32, i32)
    scf.for %arg9 = %6 to %13 step %c148_i32  : i32 {
      %20 = arith.divsi %arg9, %14 {ttg.partition = array<i32: 0, 2>} : i32
      %21 = arith.muli %20, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %22 = arith.subi %8, %21 {ttg.partition = array<i32: 0, 2>} : i32
      %23 = arith.minsi %22, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %24 = arith.remsi %arg9, %23 {ttg.partition = array<i32: 0, 2>} : i32
      %25 = arith.addi %21, %24 {ttg.partition = array<i32: 0, 2>} : i32
      %26 = arith.remsi %arg9, %14 {ttg.partition = array<i32: 0, 2>} : i32
      %27 = arith.divsi %26, %23 {ttg.partition = array<i32: 0, 2>} : i32
      %28 = arith.muli %25, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %29 = arith.muli %27, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      // ACC stage advance: addi/cmpi/select wrapping at depth=2
      // CHECK: [[ACC_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[ACC_S_NEXT:%.*]] = arith.addi [[ACC_S]], [[ACC_C1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ACC_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
      // CHECK: [[ACC_S_WRAP:%.*]] = arith.cmpi eq, [[ACC_S_NEXT]], [[ACC_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ACC_C0:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[ASTAGE0:%.*]] = arith.select [[ACC_S_WRAP]], [[ACC_C0]], [[ACC_S_NEXT]] {ttg.partition = array<i32: 0, 1>} : i32
      // Phase flip BEFORE acquire ACC_EMPTY (partition 0)
      // CHECK: [[ACC_PE_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[ACC_PE_SHIFT0:%.*]] = arith.shli [[ACC_PE_C1]], [[ASTAGE0]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE0:%.*]] = arith.xori [[ACC_PE]], [[ACC_PE_SHIFT0]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE0_SHR:%.*]] = arith.shrui [[APHASE0]], [[ASTAGE0]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE0_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[APHASE0_BIT:%.*]] = arith.andi [[APHASE0_SHR]], [[APHASE0_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ATOK0:%.*]] = nvws.semaphore.acquire [[ACC_EMPTY]][[[ASTAGE0]], [[APHASE0_BIT]]] {ttg.partition = array<i32: 0>}
      %token = nvws.semaphore.acquire %empty_acc {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %30 = nvws.semaphore.buffer %empty_acc, %token {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[BUF_ACC:%.*]] = nvws.semaphore.buffer [[ACC_EMPTY]][[[ASTAGE0]]], [[ATOK0]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store {{.*}}, [[BUF_ACC]][], {{.*}} {ttg.partition = array<i32: 0>}
      %31 = ttng.tmem_store %cst, %30[], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: nvws.semaphore.release [[ACC_FULL]][[[ASTAGE0]]], [[ATOK0]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %full_acc, %token [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Phase flip BEFORE acquire ACC_FULL (partition 1)
      // CHECK: [[ACC_C1_NEXT:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[ACC_S_NEXT1:%.*]] = arith.addi [[ASTAGE0]], [[ACC_C1_NEXT]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ACC_DEPTH_NEXT:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
      // CHECK: [[ACC_S_WRAP1:%.*]] = arith.cmpi eq, [[ACC_S_NEXT1]], [[ACC_DEPTH_NEXT]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ACC_C0_NEXT:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[ASTAGE1:%.*]] = arith.select [[ACC_S_WRAP1]], [[ACC_C0_NEXT]], [[ACC_S_NEXT1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ACC_PF_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[ACC_PF_SHIFT0:%.*]] = arith.shli [[ACC_PF_C1]], [[ASTAGE1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE1:%.*]] = arith.xori [[ACC_PF]], [[ACC_PF_SHIFT0]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE1_SHR:%.*]] = arith.shrui [[APHASE1]], [[ASTAGE1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[APHASE1_BIT:%.*]] = arith.andi [[APHASE1_SHR]], [[APHASE1_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK1:%.*]] = nvws.semaphore.acquire [[ACC_FULL]][[[ASTAGE1]], [[APHASE1_BIT]]] {ttg.partition = array<i32: 1>}
      %token_1 = nvws.semaphore.acquire %full_acc {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[INNER:%.*]]:4 = scf.for [[INNER_IV:%.*]] = [[INNER_LB:%.*]] to [[INNER_UB:%.*]] step [[INNER_STEP:%.*]] iter_args({{%.*}} = [[INNER_USED:%.*]], [[AB_S_I:%.*]] = [[AB_S]], [[AB_PF_I:%.*]] = [[AB_PF]], [[AB_PE_I:%.*]] = [[AB_PE]]) -> (i1, i32, i32, i32)
      %32 = scf.for %arg10 = %c0_i32 to %12 step %c1_i32 iter_args(%arg11 = %false) -> (i1)  : i32 {
        %36 = arith.muli %arg10, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // AB stage advance + phase flip BEFORE acquire AB_EMPTY
        // CHECK: [[AB_C1:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} 1 : i32
        // CHECK: [[AB_S_NEXT:%.*]] = arith.addi [[AB_S_I]], [[AB_C1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
        // CHECK: [[AB_DEPTH:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} 3 : i32
        // CHECK: [[AB_S_WRAP:%.*]] = arith.cmpi eq, [[AB_S_NEXT]], [[AB_DEPTH]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
        // CHECK: [[AB_C0:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} 0 : i32
        // CHECK: [[ABSTAGE:%.*]] = arith.select [[AB_S_WRAP]], [[AB_C0]], [[AB_S_NEXT]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
        // CHECK: [[AB_PE_C1:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 1 : i32
        // CHECK: [[AB_PE_FLIP:%.*]] = arith.xori [[AB_PE_I]], [[AB_PE_C1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // CHECK: [[AB_PE_CMP:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
        // CHECK: [[AB_PE_EQ:%.*]] = arith.cmpi eq, [[ABSTAGE]], [[AB_PE_CMP]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // CHECK: [[AB_PE_NEW:%.*]] = arith.select [[AB_PE_EQ]], [[AB_PE_FLIP]], [[AB_PE_I]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // CHECK: [[ABTOK_P:%.*]] = nvws.semaphore.acquire [[AB_EMPTY]][[[ABSTAGE]], [[AB_PE_NEW]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        %token_9 = nvws.semaphore.acquire %ab_empty {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[BUFS_AB:%.*]]:2 = nvws.semaphore.buffer [[AB_EMPTY]][[[ABSTAGE]]], [[ABTOK_P]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        %buffers_8:2 = nvws.semaphore.buffer %ab_empty, %token_9 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        // CHECK: nvws.descriptor_load {{.*}} [[BUFS_AB]]#0 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        nvws.descriptor_load %1[%28, %36] 16384 %buffers_8#0 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf8E4M3FN, #shared>, i32, i32, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        // CHECK: nvws.descriptor_load {{.*}} [[BUFS_AB]]#1 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        nvws.descriptor_load %3[%29, %36] 16384 %buffers_8#1 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf8E4M3FN, #shared>, i32, i32, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        // CHECK: nvws.semaphore.release [[AB_FULL]][[[ABSTAGE]]], [[ABTOK_P]] [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        nvws.semaphore.release %ab_full, %token_9 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token

        // Phase flip BEFORE acquire AB_FULL
        // CHECK: [[AB_PF_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
        // CHECK: [[AB_PF_FLIP:%.*]] = arith.xori [[AB_PF_I]], [[AB_PF_C1]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[AB_PF_CMP:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
        // CHECK: [[AB_PF_EQ:%.*]] = arith.cmpi eq, [[ABSTAGE]], [[AB_PF_CMP]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[AB_PF_NEW:%.*]] = arith.select [[AB_PF_EQ]], [[AB_PF_FLIP]], [[AB_PF_I]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[ABTOK_C:%.*]] = nvws.semaphore.acquire [[AB_FULL]][[[ABSTAGE]], [[AB_PF_NEW]]] {ttg.partition = array<i32: 1>}
        %token_11 = nvws.semaphore.acquire %ab_full {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[BUFS_ABF:%.*]]:2 = nvws.semaphore.buffer [[AB_FULL]][[[ABSTAGE]]], [[ABTOK_C]] {ttg.partition = array<i32: 1>}
        %buffers_10:2 = nvws.semaphore.buffer %ab_full, %token_11 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        %37 = ttg.memdesc_trans %buffers_10#1 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable, 1x128x128>
        // CHECK: [[BUF_ACCF_INNER:%.*]] = nvws.semaphore.buffer [[ACC_FULL]][[[ASTAGE1]]], [[ATOK1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
        %38 = nvws.semaphore.buffer %full_acc, %token_1 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: ttng.tc_gen5_mma [[BUFS_ABF]]#0, {{.*}}, [[BUF_ACCF_INNER]][], {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>}
        %39 = ttng.tc_gen5_mma %buffers_10#0, %37, %38[], %arg11, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: nvws.semaphore.release [[AB_EMPTY]][[[ABSTAGE]]], [[ABTOK_C]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
        nvws.semaphore.release %ab_empty, %token_11 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} {{%.*}}, [[ABSTAGE]], [[AB_PF_NEW]], [[AB_PE_NEW]]
        scf.yield %true : i1
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: nvws.semaphore.release [[ACC_EMPTY]][[[ASTAGE1]]], [[ATOK1]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty_acc, %token_1 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %token_3 = nvws.semaphore.acquire %empty_acc {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %33 = nvws.semaphore.buffer %empty_acc, %token_3 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %result_4, %token_5 = ttng.tmem_load %33[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      nvws.semaphore.release %full_acc, %token_3 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Phase flip BEFORE acquire ACC_EMPTY
      // CHECK: [[ACC_PE_SHIFT2:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[ACC_PE_SHIFT2_V:%.*]] = arith.shli [[ACC_PE_SHIFT2]], [[ASTAGE1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE2:%.*]] = arith.xori [[APHASE0]], [[ACC_PE_SHIFT2_V]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE2_SHR:%.*]] = arith.shrui [[APHASE2]], [[ASTAGE1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE2_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[APHASE2_BIT:%.*]] = arith.andi [[APHASE2_SHR]], [[APHASE2_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ATOK2:%.*]] = nvws.semaphore.acquire [[ACC_EMPTY]][[[ASTAGE1]], [[APHASE2_BIT]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[BUF_ACCE2:%.*]] = nvws.semaphore.buffer [[ACC_EMPTY]][[[ASTAGE1]]], [[ATOK2]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_load [[BUF_ACCE2]][] {ttg.partition = array<i32: 0>}
      %token_7 = nvws.semaphore.acquire %full_acc {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[ACC_FULL]][[[ASTAGE1]]], [[ATOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // Phase flip BEFORE acquire ACC_FULL
      // CHECK: [[ACC_PF_SHIFT2:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[ACC_PF_SHIFT2_V:%.*]] = arith.shli [[ACC_PF_SHIFT2]], [[ASTAGE1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE3:%.*]] = arith.xori [[APHASE1]], [[ACC_PF_SHIFT2_V]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE3_SHR:%.*]] = arith.shrui [[APHASE3]], [[ASTAGE1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE3_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[APHASE3_BIT:%.*]] = arith.andi [[APHASE3_SHR]], [[APHASE3_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK3:%.*]] = nvws.semaphore.acquire [[ACC_FULL]][[[ASTAGE1]], [[APHASE3_BIT]]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[ACC_EMPTY]][[[ASTAGE1]]], [[ATOK3]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty_acc, %token_7 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %34 = tt.fp_to_fp %result_4 {ttg.partition = array<i32: 0>}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %35 = ttg.convert_layout %34 {ttg.partition = array<i32: 0>} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %5[%28, %29], %35 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x128xf8E4M3FN, #shared>, tensor<128x128xf8E4M3FN, #blocked1>
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[INNER]]#1, [[INNER]]#2, [[INNER]]#3, [[ASTAGE1]], [[APHASE2]], [[APHASE3]]
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1>, array<i32: 2>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @for_loop_control_operand_ppg
  tt.func @for_loop_control_operand_ppg(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>) {
    %true = arith.constant true
    %semBuf = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create {{.*}} true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create {{.*}} false
    // CHECK: [[S0:%.*]] = arith.constant 0 : i32
    // CHECK: [[PE_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant 1 : i32
    // Pre-loop: single-phase flips for both sems, BEFORE acquire EMPTY
    // CHECK: [[PF1_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF1_FLIP:%.*]] = arith.xori [[PE_INIT]], [[PF1_C1]] : i32
    // CHECK: [[PF1_CMP:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF1_EQ:%.*]] = arith.cmpi eq, [[S0]], [[PF1_CMP]] : i32
    // CHECK: [[PF1_OUT:%.*]] = arith.select [[PF1_EQ]], [[PF1_FLIP]], [[PE_INIT]] : i32
    // CHECK: [[PE1_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE1_FLIP:%.*]] = arith.xori [[PE_INIT]], [[PE1_C1]] : i32
    // CHECK: [[PE1_CMP:%.*]] = arith.constant 0 : i32
    // CHECK: [[PE1_EQ:%.*]] = arith.cmpi eq, [[S0]], [[PE1_CMP]] : i32
    // CHECK: [[P0_OUT:%.*]] = arith.select [[PE1_EQ]], [[PE1_FLIP]], [[PE_INIT]] : i32
    // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[S0]], [[P0_OUT]]]
    %empty = nvws.semaphore.create %semBuf true : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %semBuf false : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %tok = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[FOR0:%.*]]:3 = scf.for {{.*}} iter_args([[F0TOK:%.*]] = [[TOK0]], [[F0PF:%.*]] = %{{.*}}, [[F0PE:%.*]] = %{{.*}}) -> (!ttg.async.token, i32, i32)
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
      %ptrub = tt.addptr %ptr0, %iv0 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>, i32
      %ub1 = tt.load %ptrub {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>
      %lb1 = "lb1"(%iv0) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      %step1 = "step1"(%iv0) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      // CHECK: scf.for {{.*}} : i32 {
      %tok5 = scf.for %iv = %lb1 to %ub1 step %step1 iter_args(%tok2 = %tok1) -> (!ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[BUF_INNER:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[S0]]], [[F0TOK]] {ttg.partition = array<i32: 2>}
        %buf = nvws.semaphore.buffer %empty, %tok2 {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_INNER]], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>}
        ttng.tc_gen5_mma %sA, %sB, %buf, %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %tok2 : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1, 2>}
      // CHECK: nvws.semaphore.release [[FULL]][[[S0]]], [[F0TOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 2>}
      nvws.semaphore.release %full, %tok5 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Phase flip BEFORE acquire FULL
      // CHECK: [[GC1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[GC1_FLIP:%.*]] = arith.xori [[F0PF:%.*]], [[GC1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[GC1_CMP:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[GC1_EQ:%.*]] = arith.cmpi eq, [[S0]], [[GC1_CMP]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[P1_OUT:%.*]] = arith.select [[GC1_EQ]], [[GC1_FLIP]], [[F0PF]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[TOK1:%.*]] = nvws.semaphore.acquire [[FULL]][[[S0]], [[P1_OUT]]] {ttg.partition = array<i32: 1>}
      %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]][[[S0]]], [[TOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Phase flip BEFORE re-acquire EMPTY
      // CHECK: [[PC1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // CHECK: [[PC1_FLIP:%.*]] = arith.xori [[F0PE:%.*]], [[PC1]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[PC1_CMP:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[PC1_EQ:%.*]] = arith.cmpi eq, [[S0]], [[PC1_CMP]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[P2_OUT:%.*]] = arith.select [[PC1_EQ]], [[PC1_FLIP]], [[F0PE]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TOK2:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[S0]], [[P2_OUT]]] {ttg.partition = array<i32: 2>}
      %tok6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[TOK2]], [[P1_OUT]], [[P2_OUT]]
      scf.yield {ttg.partition = array<i32: 1, 2>} %tok6 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 0, 1>, array<i32: 2>]}
    // CHECK: nvws.semaphore.release [[FULL]][[[S0]]], [[FOR0]]#0 [#nvws.async_op<tc5mma>]
    // Post-loop: phase flips for both sems, BEFORE acquire FULL
    // CHECK: [[POST_C1A:%.*]] = arith.constant 1 : i32
    // CHECK: [[POST_FLIPA:%.*]] = arith.xori [[POST_OLD_A:%.*]], [[POST_C1A]] : i32
    // CHECK: [[POST_CMPA:%.*]] = arith.constant 0 : i32
    // CHECK: [[POST_EQA:%.*]] = arith.cmpi eq, [[S0]], [[POST_CMPA]] : i32
    // CHECK: arith.select [[POST_EQA]], [[POST_FLIPA]], [[POST_OLD_A]] : i32
    // CHECK: [[POST_C1B:%.*]] = arith.constant 1 : i32
    // CHECK: [[POST_FLIPB:%.*]] = arith.xori [[POST_OLD_B:%.*]], [[POST_C1B]] : i32
    // CHECK: [[POST_CMPB:%.*]] = arith.constant 0 : i32
    // CHECK: [[POST_EQB:%.*]] = arith.cmpi eq, [[S0]], [[POST_CMPB]] : i32
    // CHECK: [[P_END:%.*]] = arith.select [[POST_EQB]], [[POST_FLIPB]], [[POST_OLD_B]] : i32
    // CHECK: [[TOK_END:%.*]] = nvws.semaphore.acquire [[FULL]][[[S0]], [[P_END]]]
    // CHECK: nvws.semaphore.release [[EMPTY]][[[S0]]], [[TOK_END]] [#nvws.async_op<none>]
    nvws.semaphore.release %full, %tok0 [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %token_end = nvws.semaphore.acquire %full : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    nvws.semaphore.release %empty, %token_end [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}
