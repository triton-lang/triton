// RUN: triton-opt %s -split-input-file --triton-nvidia-aref-lowering | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blockedB = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#mshared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {
    "ttg.num-ctas"  = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32,
    ttg.target = "cuda:100", "ttg.warp-specialized" = true
  } {
  tt.func public @aref_put_tmaldg_get_lds(
    %desc0: !tt.tensordesc<tensor<128x64xf16, #shared>>,
    %arg0: !tt.tensordesc<tensor<1x128xf16, #nvmma_128>>,
    %arg1: tensor<32xi32, #blockedA>,
    %arg2: i32,
    %ptr: tensor<128x64x!tt.ptr<f16>, #blocked>,
    %K : i32
  ) attributes {noinline = false} {
    // CHECK-DAG: %[[INIT:.*]] = arith.constant 0
    // CHECK-DAG: %[[C3:.*]] = arith.constant 3
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %x = tt.get_program_id x : i32

    // CHECK-DAG: %[[AREF_BUF:.*]] = ttg.local_alloc : () ->   !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %arefBuf = ttg.local_alloc : () ->   !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    // CHECK-DAG: %[[EMPTY_MBAR_ALLOC:.*]] = ttg.local_alloc {aref_empty_mbarriers}
    // CHECK-DAG: %[[FULL_MBAR_ALLOC:.*]] = ttg.local_alloc {aref_full_mbarriers}
    %aref = ttng.aref_create %arefBuf : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>


    ttng.warp_group start_warp(0) num_warps(1) : {{

        // CHECK: %[[RET:.*]]:2 = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[AREF_IDX:.*]] = {{.*}}, %[[PHASE:.*]] = %[[INIT]])
        %idx = scf.for %i = %c0 to %K step %c4 iter_args(%arefIdx = %c0) -> i32 : i32 {
          %y = arith.muli %i, %c16 : i32
          // CHECK-DAG: %[[AREF_ENTER_MBAR_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]]
          // CHECK: %[[EMPTY_MBAR:.*]] = ttg.memdesc_subview %[[EMPTY_MBAR_ALLOC]][%[[AREF_ENTER_MBAR_IDX]]]

          // CHECK-DAG: %[[PHASE_DIV:.*]] = arith.divsi %[[PHASE]], %[[C3]] {put_phase}
          // CHECK: %[[PHASE_MOD:.*]] = arith.andi %[[PHASE_DIV]], %[[C1]] {put_phase}
          // CHECK: %[[PUT_PHASE:.*]] = arith.xori %[[PHASE_MOD]], %[[C1]] {put_phase}
          // CHECK: ttng.wait_barrier %[[EMPTY_MBAR]], %[[PUT_PHASE]]

          // CHECK-DAG: %[[AREF_SUBVIEW_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {put_view}
          // CHECK: %[[SMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_BUF]][%[[AREF_SUBVIEW_IDX]], {{.*}}, {{.*}}]

          %buf = ttng.aref_put.enter %aref[%arefIdx] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

          // CHECK-DAG: %[[AREF_PUT_MBAR:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {full_mbar}
          // CHECK: %[[FULL_MBAR:.*]] = ttg.memdesc_subview %[[FULL_MBAR_ALLOC]][%[[AREF_PUT_MBAR]]]

          // CHECK: ttng.barrier_expect %[[FULL_MBAR]], 16384
          // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[SMEM_VIEW]], %[[FULL_MBAR]]
          %a = tt.descriptor_load %desc0[%x, %y] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
          ttg.local_store %a, %buf : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.aref_put.exit %aref[%arefIdx], producers = [#ttng.aref_producer<tmaldg>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32

          %arefIdx1 = arith.addi %arefIdx, %c1 : i32
          // CHECK-DAG: %[[AREF_PHASE:.*]] = arith.addi %[[PHASE]], %[[C1]] {next_phase}
          // CHECK: scf.yield {{.*}}, %[[AREF_PHASE]] : i32
          scf.yield %arefIdx1 : i32
        }
      ttng.warp_group_return
    } {barId = 0 : i32}}


    ttng.warp_group start_warp(4) num_warps(4) : {{
        %bar = ttg.local_alloc : () ->   !ttg.memdesc<1xi64, #mshared, #smem, mutable>

        // CHECK: %[[RET:.*]]:2 = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[AREF_IDX:.*]] = {{.*}}, %[[PHASE:.*]] = %[[INIT]])
        %idx = scf.for %i = %c0 to %K step %c4 iter_args(%arefIdx = %c0) -> i32 : i32 {
          %y = arith.muli %i, %c16 : i32

          // CHECK-DAG: %[[AREF_ENTER_MBAR_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]]
          // CHECK: %[[FULL_MBAR:.*]] = ttg.memdesc_subview %[[FULL_MBAR_ALLOC]][%[[AREF_ENTER_MBAR_IDX]]]

          // CHECK: %[[PHASE_DIV:.*]] = arith.divsi %[[PHASE]], %[[C3]] {get_phase}
          // CHECK: %[[PHASE_MOD:.*]] = arith.andi %[[PHASE_DIV]], %[[C1]] {get_phase}
          // CHECK: ttng.wait_barrier %[[FULL_MBAR]], %[[PHASE_MOD]]

          // CHECK-DAG: %[[AREF_SUBVIEW_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {get_view}
          // CHECK: %[[SMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_BUF]][%[[AREF_SUBVIEW_IDX]], {{.*}}, {{.*}}]
          %buf = ttng.aref_get.enter %aref[%arefIdx] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          // CHECK: %[[AREF_PHASE:.*]] = arith.addi %[[PHASE]], %[[C1]] {next_phase}

          // CHECK: {{.*}} = ttg.local_load %[[SMEM_VIEW]]
          %val = ttg.local_load %buf : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>


          // CHECK-DAG: %[[AREF_EXIT_MBAR_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {empty_mbar}
          // CHECK: %[[EXIT_MBAR:.*]] = ttg.memdesc_subview %[[EMPTY_MBAR_ALLOC]][%[[AREF_EXIT_MBAR_IDX]]]
          // CHECK: nvws.arrive_barrier %[[EXIT_MBAR]], tracked_async_op = <none>
          ttng.aref_get.exit %aref[%arefIdx], consumers = [#ttng.aref_consumer<lds>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32

          // need store for %val = ttg.local_load %buf not be DCEd
          tt.store %ptr, %val : tensor<128x64x!tt.ptr<f16>, #blocked>

          %arefIdx1 = arith.addi %arefIdx, %c1 : i32
          // CHECK: scf.yield {{.*}}, %[[AREF_PHASE]] : i32
          scf.yield %arefIdx1 : i32
        }
      ttng.warp_group_return
    } {barId = 1 : i32}}

    ttng.warp_group start_warp(8) num_warps(1) : {{
      %arefGatherBuf = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #nvmma_128, #smem, mutable>
      %arefGather = ttng.aref_create %arefGatherBuf : <[!ttg.memdesc<1x32x128xf16, #nvmma_128, #smem, mutable>]>

      %buf2 = ttng.aref_get.enter %arefGather[%c0] <%c0>: <[!ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
      ttng.aref_get.exit %arefGather[%c0], consumers = [#ttng.aref_consumer<lds>] : <[!ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>]>, i32

      %buf1 = ttng.aref_put.enter %arefGather[%c0] <%c0>: <[!ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
      // CHECK: %[[PTR:.*]] = ttng.tensor_desc_to_tma_ptr {{.*}}
      // CHECK: ttng.async_tma_gather %[[PTR]][{{.*}}2, {{.*}}]
      // CHECK-NEXT: ttng.warp_group_return
      %b = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xf16, #nvmma_128>>, tensor<32xi32, #blockedA>, i32) -> tensor<32x128xf16, #blockedB>
      ttg.local_store %b, %buf1 : tensor<32x128xf16, #blockedB> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
      ttng.aref_put.exit %arefGather[%c0], producers = [#ttng.aref_producer<tmaldg>] : <[!ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>]>, i32

      ttng.warp_group_return
    } {barId = 2 : i32}}

    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#mshared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
module attributes {
    "ttg.num-ctas"  = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32,
    ttg.target = "cuda:100", "ttg.warp-specialized" = true
  } {
  tt.func public @aref_put_tmaldg_sttm_sts_get_lds_ldtm(
    %desc0: !tt.tensordesc<tensor<128x256xf16, #shared>>,
    %K : i32
  ) attributes {noinline = false} {
    %bar = ttg.local_alloc : () ->   !ttg.memdesc<1xi64, #mshared, #smem, mutable>

    // CHECK-DAG: %[[INIT:.*]] = arith.constant 0
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2
    // CHECK-DAG: %[[C3:.*]] = arith.constant 3
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32

    %x = tt.get_program_id x : i32

    // CHECK-DAG: %[[AREF_SMEM_BUF:.*]] = ttg.local_alloc : () ->   !ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>
    %arefSmemBuf = ttg.local_alloc : () ->   !ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>
    // CHECK-DAG: %[[E_SMEM_MBAR_ALLOC:.*]] = ttg.local_alloc {aref_empty_mbarriers} : () -> !ttg.memdesc<3xi64,
    // CHECK-DAG: %[[F_SMEM_MBAR_ALLOC:.*]] = ttg.local_alloc {aref_full_mbarriers} : () -> !ttg.memdesc<3xi64,
    %arefSmem = ttng.aref_create %arefSmemBuf : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>
    // CHECK-DAG: %[[E_SMEM1_MBAR_ALLOC:.*]] = ttg.local_alloc {aref_empty_mbarriers} : () -> !ttg.memdesc<3xi64,
    // CHECK-DAG: %[[F_SMEM1_MBAR_ALLOC:.*]] = ttg.local_alloc {aref_full_mbarriers} : () -> !ttg.memdesc<3xi64,
    %arefSmem1 = ttng.aref_create %arefSmemBuf : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>

    // CHECK-DAG: %[[AREF_TMEM_BUF:.*]] = ttng.tmem_alloc : () ->   !ttg.memdesc<2x128x256xf32
    %arefTmemBuf = ttng.tmem_alloc : () ->    !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-DAG: %[[E_TMEM_MBAR_ALLOC:.*]] = ttg.local_alloc {aref_empty_mbarriers} : () -> !ttg.memdesc<2xi64,
    // CHECK-DAG: %[[F_TMEM_MBAR_ALLOC:.*]] = ttg.local_alloc {aref_full_mbarriers} : () -> !ttg.memdesc<2xi64,
    %arefTmem = ttng.aref_create %arefTmemBuf : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>

    ttng.warp_group start_warp(0) num_warps(4) : {{
        // CHECK: %[[RET:.*]]:4 = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[AREF_IDX:.*]] = {{.*}}, %[[SPHASE:.*]] = %[[INIT]], %[[TPHASE:.*]] = %[[INIT]], %[[S1PHASE:.*]] = %[[INIT]])
        %idx = scf.for %i = %c0 to %K step %c4 iter_args(%arefIdx = %c0) -> i32 : i32 {
          %y = arith.muli %i, %c16 : i32

          // CHECK: %[[SMEM_PUT_ENTER_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {empty_mbar}
          // CHECK: %[[E_SMEM_MBAR:.*]] = ttg.memdesc_subview %[[E_SMEM_MBAR_ALLOC]][%[[SMEM_PUT_ENTER_IDX]]]
          %smemLdg = ttng.aref_put.enter %arefSmem[%arefIdx] : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
          // CHECK: %[[SPHASE1:.*]] = arith.addi %[[SPHASE]], %[[C1]] {next_phase}

          %a = tt.descriptor_load %desc0[%x, %y] : !tt.tensordesc<tensor<128x256xf16, #shared>> -> tensor<128x256xf16, #blocked>
          ttg.local_store %a, %smemLdg : tensor<128x256xf16, #blocked> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
          ttng.aref_put.exit %arefSmem[%arefIdx], producers = [#ttng.aref_producer<tmaldg>] : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>, i32

          // CHECK: %[[TMEM_GET_ENTER_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C2]] {full_mbar}
          // CHECK: %[[F_TMEM_MBAR:.*]] = ttg.memdesc_subview %[[F_TMEM_MBAR_ALLOC]][%[[TMEM_GET_ENTER_IDX]]]

          // CHECK: %[[TPHASE_DIV:.*]] = arith.divsi %[[TPHASE]], %[[C2]] {get_phase}
          // CHECK: %[[TPHASE_MOD:.*]] = arith.andi %[[TPHASE_DIV]], %[[C1]] {get_phase}
          // CHECK: ttng.wait_barrier %[[F_TMEM_MBAR]], %[[TPHASE_MOD]]

          // CHECK: %[[TMEM_GET_ENTER_IDX1:.*]] = arith.remsi %[[AREF_IDX]], %[[C2]] {get_view}
          // CHECK: %[[TMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_TMEM_BUF]][%[[TMEM_GET_ENTER_IDX1]], {{.*}}, {{.*}}]
          %tmemLdttm = ttng.aref_get.enter %arefTmem[%arefIdx] : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32 -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

          // CHECK: {{.*}} = ttng.tmem_load %[[TMEM_VIEW]]
          %val32 = ttng.tmem_load  %tmemLdttm : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>

          // CHECK: %[[TMEM_GET_EXIT_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C2]]
          // CHECK: %[[E_TMEM_MBAR:.*]] = ttg.memdesc_subview %[[E_TMEM_MBAR_ALLOC]][%[[TMEM_GET_EXIT_IDX]]]
          // CHECK: nvws.arrive_barrier %[[E_TMEM_MBAR]], tracked_async_op = <none>
          ttng.aref_get.exit %arefTmem[%arefIdx], consumers = [#ttng.aref_consumer<ldtm>] : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32

          %val16  = arith.truncf %val32 : tensor<128x256xf32, #blocked> to tensor<128x256xf16, #blocked>

          // CHECK: %[[SMEM_PUT_ENTER_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {empty_mbar}
          // CHECK: %[[E_SMEM_MBAR:.*]] = ttg.memdesc_subview %[[E_SMEM1_MBAR_ALLOC]][%[[SMEM_PUT_ENTER_IDX]]]

          // CHECK: %[[SPHASE_DIV:.*]] = arith.divsi %[[S1PHASE]], %[[C3]] {put_phase}
          // CHECK: %[[SPHASE_MOD:.*]] = arith.andi %[[SPHASE_DIV]], %[[C1]] {put_phase}
          // CHECK: %[[SPHASE_PUT:.*]] = arith.xori %[[SPHASE_MOD]], %[[C1]] {put_phase}
          // CHECK: ttng.wait_barrier %[[E_SMEM_MBAR]], %[[SPHASE_PUT]]

          // CHECK: %[[SMEM_GET_ENTER_IDX1:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]]
          // CHECK: %[[SMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_SMEM_BUF]][%[[SMEM_GET_ENTER_IDX1]], {{.*}}, {{.*}}]
          %smemSts = ttng.aref_put.enter %arefSmem1[%arefIdx] : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>

          // CHECK: ttg.local_store %{{.*}}, %[[SMEM_VIEW]]
          ttg.local_store %val16, %smemSts : tensor<128x256xf16, #blocked> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>

          // CHECK: %[[SMEM_PUT_EXIT_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {full_mbar}
          // CHECK: %[[F_SMEM_MBAR:.*]] = ttg.memdesc_subview %[[F_SMEM1_MBAR_ALLOC]][%[[SMEM_PUT_EXIT_IDX]]]
          // CHECK: nvws.arrive_barrier %[[F_SMEM_MBAR]], tracked_async_op = <none>
          ttng.aref_put.exit %arefSmem1[%arefIdx], producers = [#ttng.aref_producer<sts>] : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>, i32

          %smemSts1 = ttng.aref_get.enter %arefSmem1[%arefIdx] <%arefIdx> : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
          ttng.aref_get.exit %arefSmem1[%arefIdx], consumers = [#ttng.aref_consumer<lds>] : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>, i32

          %arefIdx1 = arith.addi %arefIdx, %c1 : i32
          scf.yield %arefIdx1 : i32
        }
      ttng.warp_group_return
    } {barId = 0 : i32}}

    ttng.warp_group start_warp(4) num_warps(4) : {{
        // CHECK: %[[RET:.*]]:3 = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[AREF_IDX:.*]] = {{.*}}, %[[SPHASE:.*]] = %[[INIT]], %[[TPHASE:.*]] = %[[INIT]])
        %idx = scf.for %i = %c0 to %K step %c4 iter_args(%arefIdx = %c0) -> i32 : i32 {

          // CHECK: %[[SMEM_GET_ENTER_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {get_view}
          // CHECK-NEXT: %[[SMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_SMEM_BUF]][%[[SMEM_GET_ENTER_IDX]], {{.*}}, {{.*}}]
          %smemLds = ttng.aref_get.enter %arefSmem[%arefIdx] : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
          // CHECK: {{.*}} = ttg.local_load %[[SMEM_VIEW]]
          %val = ttg.local_load %smemLds : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> tensor<128x256xf16, #blocked>

          // CHECK: %[[SMEM_GET_EXIT_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {empty_mbar}
          // CHECK: %[[E_SMEM_MBAR:.*]] = ttg.memdesc_subview %[[E_SMEM_MBAR_ALLOC]][%[[SMEM_GET_EXIT_IDX]]]
          // CHECK: nvws.arrive_barrier %[[E_SMEM_MBAR]], tracked_async_op = <none>
          ttng.aref_get.exit %arefSmem[%arefIdx], consumers = [#ttng.aref_consumer<lds>] : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>, i32

          %val32 = arith.extf %val : tensor<128x256xf16, #blocked> to tensor<128x256xf32, #blocked>

          // CHECK: %[[TMEM_PUT_ENTER_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C2]] {put_view}
          // CHECK-NEXT: %[[TMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_TMEM_BUF]][%[[TMEM_PUT_ENTER_IDX]], {{.*}}, {{.*}}]
          %tmemStm = ttng.aref_put.enter %arefTmem[%arefIdx] : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32 -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: ttng.tmem_store {{.*}}, %[[TMEM_VIEW]]
          ttng.tmem_store %val32, %tmemStm, %true : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

          // CHECK: %[[TMEM_PUT_EXIT_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C2]] {full_mbar}
          // CHECK: %[[F_TMEM_MBAR:.*]] = ttg.memdesc_subview %[[F_TMEM_MBAR_ALLOC]][%[[TMEM_PUT_EXIT_IDX]]]
          // CHECK: nvws.arrive_barrier %[[F_TMEM_MBAR]], tracked_async_op = <none>
          ttng.aref_put.exit %arefTmem[%arefIdx], producers = [#ttng.aref_producer<sttm>] : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32


          %arefIdx1 = arith.addi %arefIdx, %c1 : i32
          scf.yield %arefIdx1 : i32
        }

      ttng.warp_group_return
    } {barId = 1 : i32}}

    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {
    "ttg.num-ctas"  = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32,
    ttg.target = "cuda:100", "ttg.warp-specialized" = true
  } {
  tt.func public @aref_phase_propagation(
    %desc0: !tt.tensordesc<tensor<128x64xf16, #shared>>,
    %K : i32
  ) attributes {noinline = false} {
    %bar = ttg.local_alloc : () ->   !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    %c0 = arith.constant 0 : i32
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %c128 = arith.constant 16 : i32
    %x = tt.get_program_id x : i32

    %arefBuf = ttg.local_alloc : () ->   !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %aref = ttng.aref_create %arefBuf : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>

    ttng.warp_group start_warp(0) num_warps(1) : {{
        // CHECK: %[[RET_FOR1:.*]]:2 = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args({{.*}} = {{.*}}, %[[PHASE1:.*]] = {{.*}})
        %idx = scf.for %i = %c0 to %K step %c4 iter_args(%arefIdx = %c0) -> i32 : i32 {
          %y = arith.muli %i, %c16 : i32
          %mod4 = arith.remsi %i, %c4 : i32
          %cond = arith.cmpi eq, %mod4, %c0 : i32
          // CHECK: %[[RET_IF1:.*]]:2 = scf.if
          %idx1 = scf.if %cond -> i32 {
            // CHECK: %[[RET_FOR2:.*]]:2 = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args({{.*}} = {{.*}}, %[[PHASE2:.*]] = %[[PHASE1]])
            %idx2 = scf.for %ii = %c0 to %c4 step %c1 iter_args(%arefIdx1 = %arefIdx) -> i32 : i32 {
              %y1 = arith.muli %ii, %c16 : i32

              %buf = ttng.aref_put.enter %aref[%arefIdx1] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
              %a = tt.descriptor_load %desc0[%x, %y1] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
              ttg.local_store %a, %buf : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
              ttng.aref_put.exit %aref[%arefIdx1], producers = [#ttng.aref_producer<tmaldg>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32
              // CHECK: %[[PHASE2a:.*]] = arith.addi %[[PHASE2]], %[[C1]] {next_phase}

              %arefIdx2 = arith.addi %arefIdx1, %c1 : i32
              // CHECK: scf.yield {{.*}}, %[[PHASE2a]]
              scf.yield %arefIdx2 : i32
            }
            // CHECK: scf.yield {{.*}}, %[[RET_FOR2]]#1
            scf.yield %idx2 : i32
          } else {
            // CHECK: scf.yield {{.*}}, %[[PHASE1]]
            scf.yield %arefIdx : i32
          }

            // CHECK: scf.yield {{.*}}, %[[RET_IF1]]#1
          scf.yield %idx1 : i32
        }

        %buf = ttng.aref_put.enter %aref[%c0] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        // CHECK: %[[PHASE3:.*]] = arith.addi %[[RET_FOR1]]#1, %[[C1]] {next_phase}
        ttng.aref_put.exit %aref[%c0], producers = [#ttng.aref_producer<tmaldg>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32

        %buf1 = ttng.aref_put.enter %aref[%c0] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttng.aref_put.exit %aref[%c0], producers = [#ttng.aref_producer<tmaldg>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32

      ttng.warp_group_return
    } {barId = 0 : i32}}

    ttng.warp_group start_warp(1) num_warps(1) : {{
        // CHECK: %[[RET_FOR1:.*]] = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[PHASE1:.*]] = {{.*}})
        scf.for %i = %c0 to %K step %c1 : i32 {
          %mod4 = arith.remsi %i, %c4 : i32
          %cond = arith.cmpi eq, %mod4, %c0 : i32
          // CHECK: %[[RET_IF1:.*]] = scf.if
          scf.if %cond {
            // CHECK: %[[RET_FOR2:.*]] = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[PHASE2:.*]] = %[[PHASE1]])
            scf.for %ii = %c0 to %c4 step %c1 : i32 {
              %arefIdx = arith.addi %i, %ii : i32
              %y = arith.muli %arefIdx, %c16 : i32

              %buf = ttng.aref_get.enter %aref[%arefIdx] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
              // CHECK: %[[PHASE2a:.*]] = arith.addi %[[PHASE2]], %[[C1]] {next_phase}
              %a = ttg.local_load %buf : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
              ttng.aref_get.exit %aref[%arefIdx], consumers = [#ttng.aref_consumer<lds>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32
              // CHECK: scf.yield %[[PHASE2a]]
            }
            // CHECK: } else {
            // CHECK-NEXT: scf.yield %[[PHASE1]]
          }
        }

        %cond1 = arith.cmpi eq, %K, %c128 : i32
        // CHECK: %[[RET_IF2:.*]] = scf.if
        scf.if %cond1 {
          // CHECK: %[[RET_FOR2:.*]] = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[PHASE2:.*]] = %[[RET_FOR1]])
          scf.for %ii = %c0 to %c128 step %c1 : i32 {
            %mod2 = arith.remsi %ii, %c2 : i32
            %cond = arith.cmpi eq, %mod2, %c0 : i32
            scf.if %cond {
              %arefIdx = arith.divsi %ii, %c2 : i32
              %y = arith.muli %arefIdx, %c16 : i32

              %buf = ttng.aref_get.enter %aref[%arefIdx] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
              // CHECK: %[[PHASE2a:.*]] = arith.addi %[[PHASE2]], %[[C1]] {next_phase}
              %a = ttg.local_load %buf : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
              ttng.aref_get.exit %aref[%arefIdx], consumers = [#ttng.aref_consumer<lds>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32
              // CHECK: scf.yield %[[PHASE2a]]
            }
            // CHECK: } else {
            // CHECK-NEXT: scf.yield %[[PHASE2]]
          }
        }
        // CHECK: } else {
        // CHECK-NEXT: scf.yield %[[RET_FOR1]]

        %buf = ttng.aref_get.enter %aref[%c0] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        // CHECK: %[[PHASE3:.*]] = arith.addi %[[RET_IF2]], %[[C1]] {next_phase}
        ttng.aref_get.exit %aref[%c0], consumers = [#ttng.aref_consumer<lds>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32

        %buf1 = ttng.aref_get.enter %aref[%c0] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttng.aref_get.exit %aref[%c0], consumers = [#ttng.aref_consumer<lds>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32

      ttng.warp_group_return
    } {barId = 1 : i32}}

    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {
    "ttg.num-ctas"  = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32,
    ttg.target = "cuda:100", "ttg.warp-specialized" = true
  } {
  tt.func public @aref_manual_phase(
    %desc0: !tt.tensordesc<tensor<128x64xf16, #shared>>,
    %K : i32
  ) attributes {noinline = false} {
    %bar = ttg.local_alloc : () ->   !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %c128 = arith.constant 128 : i32
    %x = tt.get_program_id x : i32

    %arefBuf = ttg.local_alloc : () ->   !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %aref = ttng.aref_create %arefBuf : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    %aref1 = ttng.aref_create %arefBuf : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>

    ttng.warp_group start_warp(0) num_warps(1) : {{
        // CHECK: {{.*}} = scf.for %[[I:.*]] = %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[AREF_IDX0:.*]] = %[[C0]])
        %idx = scf.for %i = %c0 to %K step %c4 iter_args(%arefIdx = %c0) -> i32 : i32 {
          %y = arith.muli %i, %c16 : i32
          %mod4 = arith.remsi %i, %c4 : i32
          %cond = arith.cmpi eq, %mod4, %c0 : i32
          %idx1 = scf.if %cond -> i32 {
            // CHECK: {{.*}} = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[AREF_IDX1:.*]] = %[[AREF_IDX0]])
            %idx2 = scf.for %ii = %c0 to %c4 step %c1 iter_args(%arefIdx1 = %arefIdx) -> i32 : i32 {
              %y1 = arith.muli %ii, %c16 : i32

              %buf = ttng.aref_put.enter %aref[%arefIdx1] <%arefIdx1>: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
              // CHECK: %[[PHASE_DIV:.*]] = arith.divsi %[[AREF_IDX1]], %[[C3]] {put_phase}
              // CHECK: %[[PHASE_AND:.*]] = arith.andi %[[PHASE_DIV]], %[[C1]] {put_phase}
              // CHECK: %[[PHASE_PUT:.*]] = arith.xori %[[PHASE_AND]], %[[C1]] {put_phase}
              // CHECK: ttng.wait_barrier {{.*}}, %[[PHASE_PUT]]
              %a = tt.descriptor_load %desc0[%x, %y1] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
              ttg.local_store %a, %buf : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
              ttng.aref_put.exit %aref[%arefIdx1], producers = [#ttng.aref_producer<tmaldg>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32

              %arefIdx2 = arith.addi %arefIdx1, %c1 : i32
              scf.yield %arefIdx2 : i32
            }
            scf.yield %idx2 : i32
          } else {
            scf.yield %arefIdx : i32
          }
          %buf = ttng.aref_get.enter %aref[%i] <%i>: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          // CHECK: %[[PHASE_DIV:.*]] = arith.divsi %[[I]], %[[C3]] {get_phase}
          // CHECK: %[[PHASE_GET:.*]] = arith.andi %[[PHASE_DIV]], %[[C1]] {get_phase}
          // CHECK: ttng.wait_barrier {{.*}}, %[[PHASE_GET]]
          ttng.aref_get.exit %aref[%i], consumers=[#ttng.aref_consumer<lds>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32

          scf.yield %idx1 : i32
        }
      ttng.warp_group_return
    } {barId = 0 : i32}}

    ttng.warp_group start_warp(1) num_warps(1) : {{
        // CHECK: scf.for %[[I:.*]] = %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]]
        scf.for %i = %c0 to %K step %c1 : i32 {
          %mod4 = arith.remsi %i, %c4 : i32
          %cond = arith.cmpi eq, %mod4, %c0 : i32
          scf.if %cond {
            scf.for %ii = %c0 to %c4 step %c1 : i32 {
              %arefIdx = arith.addi %i, %ii {tag1} : i32
              // CHECK: %[[AREF_IDX:.*]] = arith.addi {{.*}}, {{.*}} {tag1} : i32
              %y = arith.muli %arefIdx, %c16 : i32

              %buf = ttng.aref_get.enter %aref[%arefIdx] <%arefIdx>: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
              // CHECK: %[[PHASE_DIV:.*]] = arith.divsi %[[AREF_IDX]], %[[C3]] {get_phase}
              // CHECK: %[[PHASE_GET:.*]] = arith.andi %[[PHASE_DIV]], %[[C1]] {get_phase}
              // CHECK: ttng.wait_barrier {{.*}}, %[[PHASE_GET]]
              %a = ttg.local_load %buf : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
              ttng.aref_get.exit %aref[%arefIdx], consumers = [#ttng.aref_consumer<lds>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32
            }
          }
          %buf = ttng.aref_put.enter %aref1[%i] <%i>: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          // CHECK: %[[PHASE_DIV:.*]] = arith.divsi %[[I]], %[[C3]] {put_phase}
          // CHECK: %[[PHASE_AND:.*]] = arith.andi %[[PHASE_DIV]], %[[C1]] {put_phase}
          // CHECK: %[[PHASE_PUT:.*]] = arith.xori %[[PHASE_AND]], %[[C1]] {put_phase}
          // CHECK: ttng.wait_barrier {{.*}}, %[[PHASE_PUT]]
          ttng.aref_put.exit %aref1[%i], producers=[#ttng.aref_producer<sts>]: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32

        }

        %cond1 = arith.cmpi eq, %K, %c128 : i32
        scf.if %cond1 {
          scf.for %ii = %c0 to %c128 step %c1 : i32 {
            %mod2 = arith.remsi %ii, %c2 : i32
            %cond = arith.cmpi eq, %mod2, %c0 : i32
            scf.if %cond {
              %arefIdx = arith.divsi %ii, %c2 {tag2} : i32
              // CHECK: %[[AREF_IDX:.*]] = arith.divsi {{.*}}, {{.*}} {tag2} : i32
              %y = arith.muli %arefIdx, %c16 : i32

              %buf = ttng.aref_get.enter %aref[%arefIdx] <%arefIdx>: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
              // CHECK: %[[PHASE_DIV:.*]] = arith.divsi %[[AREF_IDX]], %[[C3]] {get_phase}
              // CHECK: %[[PHASE_GET:.*]] = arith.andi %[[PHASE_DIV]], %[[C1]] {get_phase}
              // CHECK: ttng.wait_barrier {{.*}}, %[[PHASE_GET]]
              %a = ttg.local_load %buf : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
              ttng.aref_get.exit %aref[%arefIdx], consumers = [#ttng.aref_consumer<lds>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32
            }
          }
        }

        %buf1 = ttng.aref_get.enter %aref1[%c0] <%c0>: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttng.aref_get.exit %aref1[%c0], consumers=[#ttng.aref_consumer<lds>]: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, i32

      ttng.warp_group_return
    } {barId = 1 : i32}}

    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
#mshared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {
    "ttg.num-ctas"  = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32,
    ttg.target = "cuda:100", "ttg.warp-specialized" = true
  } {
  tt.func public @aref_put_utcmma_get_utcmma(
    %desc0: !tt.tensordesc<tensor<128x64xf16, #shared>>,
    %ptr: tensor<128x256x!tt.ptr<f32>, #blocked>,
    %tile0 : i32,
    %tile1 : i32,
    %K : i32
  ) attributes {noinline = false} {
    %bar = ttg.local_alloc : () ->   !ttg.memdesc<1xi64, #mshared, #smem, mutable>

    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %7 = ttg.local_alloc {aref_buffer, groups = [@nvws.tma_load]} : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %8 = ttg.local_alloc {aref_buffer, groups = [@nvws.tma_load]} : () -> !ttg.memdesc<3x64x256xf16, #shared, #smem, mutable>
    %arefS = ttng.aref_create %7, %8 {groups = [@nvws.tma_load]} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x64x256xf16, #shared, #smem, mutable>]>
    %10 = ttng.tmem_alloc {aref_buffer, groups = [@nvws.mma]} : () -> !ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %arefT = ttng.aref_create %10 {groups = [@nvws.mma]} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>

    ttng.warp_group start_warp(0) num_warps(1) : {{
      scf.for %tile = %tile0 to %tile1 step %c1 : i32 {
        scf.for %k = %c0 to %K step %c1 : i32 {
          %a = arith.muli %tile, %K : i32
          %idx = arith.addi %a, %k {tag1} : i32
          %abuf, %bbuf = ttng.aref_put.enter %arefS[%idx] <%idx> : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x64x256xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
          ttng.aref_put.exit %arefS[%idx], producers=[#ttng.aref_producer<sts>, #ttng.aref_producer<sts>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x64x256xf16, #shared, #smem, mutable>]>, i32
        }
      }
      ttng.warp_group_return
    } {barId = 0}}


    ttng.warp_group start_warp(1) num_warps(1) : {{
      // CHECK: scf.for %[[TILE:.*]] =
      scf.for %tile = %tile0 to %tile1 step %c1 : i32 {
        // CHECK: %[[PHASE_DIV:.*]] = arith.divsi %[[TILE]], {{.*}} {put_phase}
        // CHECK: %[[PHASE_AND:.*]] = arith.andi %[[PHASE_DIV]], {{.*}} {put_phase}
        // CHECK: %[[PHASE_PUT:.*]] = arith.xori %[[PHASE_AND]], {{.*}} {put_phase}
        // CHECK: ttng.wait_barrier {{.*}}, %[[PHASE_PUT]]

        // CHECK: %[[AREF_IDX:.*]] = arith.remsi %[[TILE]], {{.*}} {put_view} : i32
        // CHECK: %[[ACC:.*]] = ttg.memdesc_subview {{.*}}[%[[AREF_IDX]], {{.*}}, {{.*}}]
        %acc = ttng.aref_put.enter %arefT[%tile] <%tile> : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32 -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.for %k = %c0 to %K step %c1 : i32{
          %a = arith.muli %tile, %K : i32
          %idx = arith.addi %a, %k {tag2} : i32
          // CHECK: %[[AREF_IDX:.*]] = arith.addi {{.*}}, {{.*}} {tag2} : i32
          // CHECK: %[[PHASE_DIV:.*]] = arith.divsi %[[AREF_IDX]], {{.*}} {get_phase}
          // CHECK: %[[PHASE_GET:.*]] = arith.andi %[[PHASE_DIV]], {{.*}} {get_phase}
          // CHECK: ttng.wait_barrier {{.*}}, %[[PHASE_GET]]
          %abuf, %bbuf = ttng.aref_get.enter %arefS[%idx] <%idx> : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x64x256xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
          // CHECK: ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC]]
          ttng.tc_gen5_mma %abuf, %bbuf, %acc, %true, %true {is_async} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: %[[MBAR_IDX:.*]] = arith.remsi %[[AREF_IDX]], {{.*}} {empty_mbar}
          // CHECK: %[[MBAR_VIEW:.*]] = ttg.memdesc_subview {{.*}}[%[[MBAR_IDX]]]
          // CHECK: nvws.arrive_barrier %[[MBAR_VIEW]], tracked_async_op = <umma>
          ttng.aref_get.exit %arefS[%idx], consumers=[#ttng.aref_consumer<umma>, #ttng.aref_consumer<umma>] : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x64x256xf16, #shared, #smem, mutable>]>, i32

        }
        // CHECK: %[[MBAR_IDX:.*]] = arith.remsi %[[TILE]], {{.*}} {full_mbar}
        // CHECK: %[[MBAR_VIEW:.*]] = ttg.memdesc_subview {{.*}}[%[[MBAR_IDX]]]
        // CHECK: nvws.arrive_barrier %[[MBAR_VIEW]], tracked_async_op = <umma>
        ttng.aref_put.exit %arefT[%tile], producers=[#ttng.aref_producer<umma>] : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32
      }
      ttng.warp_group_return
    } {barId = 1}}

    ttng.warp_group start_warp(4) num_warps(4) : {{
      // CHECK: scf.for %[[TILE:.*]]
      scf.for %tile = %tile0 to %tile1 step %c1 : i32 {
        %acc = ttng.aref_get.enter %arefT[%tile] <%tile> : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32 -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        %val = ttng.tmem_load %acc : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked2>
        ttng.aref_get.exit %arefT[%tile], consumers = [#ttng.aref_consumer<ldtm>] : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32
      }
      ttng.warp_group_return
    } {barId = 2}}


    tt.return
  }
}
