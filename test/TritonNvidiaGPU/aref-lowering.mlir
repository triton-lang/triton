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

        // CHECK: %[[RET:.*]] = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[AREF_IDX:.*]] = {{.*}})
        %idx = scf.for %i = %c0 to %K step %c4 iter_args(%arefIdx = %c0) -> i32 : i32 {
          %y = arith.muli %i, %c16 : i32
          // CHECK-DAG: %[[AREF_ENTER_MBAR_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]]
          // CHECK: %[[EMPTY_MBAR:.*]] = ttg.memdesc_subview %[[EMPTY_MBAR_ALLOC]][%[[AREF_ENTER_MBAR_IDX]]]

          // CHECK-DAG: %[[PHASE_DIV:.*]] = arith.divsi %[[AREF_IDX]], %[[C3]] {put_phase}
          // CHECK: %[[PHASE_MOD:.*]] = arith.andi %[[PHASE_DIV]], %[[C1]] {put_phase}
          // CHECK: %[[PUT_PHASE:.*]] = arith.xori %[[PHASE_MOD]], %[[C1]] {put_phase}
          // CHECK: ttng.wait_barrier %[[EMPTY_MBAR]], %[[PUT_PHASE]]

          // CHECK-DAG: %[[AREF_SUBVIEW_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {put_view}
          // CHECK: %[[SMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_BUF]][%[[AREF_SUBVIEW_IDX]], {{.*}}, {{.*}}]

          %buf, %tok = ttng.aref_put.enter %aref[%arefIdx] {aref_tag = "0"}: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> none

          // CHECK-DAG: %[[AREF_PUT_MBAR:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {full_mbar}
          // CHECK: %[[FULL_MBAR:.*]] = ttg.memdesc_subview %[[FULL_MBAR_ALLOC]][%[[AREF_PUT_MBAR]]]

          // CHECK: ttng.barrier_expect %[[FULL_MBAR]], 16384
          // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[SMEM_VIEW]], %[[FULL_MBAR]]
          %a = tt.descriptor_load %desc0[%x, %y] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
          ttg.local_store %a, %buf : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.aref_put.exit %aref[%arefIdx] , producers = [#ttng.aref_producer<tmaldg>] {aref_tag = "0"}: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>

          %arefIdx1 = arith.addi %arefIdx, %c1 : i32
          // CHECK: scf.yield {{.*}} : i32
          scf.yield %arefIdx1 : i32
        }
      ttng.warp_group_return
    } {barId = 0 : i32}}


    ttng.warp_group start_warp(4) num_warps(4) : {{
        %bar = ttg.local_alloc : () ->   !ttg.memdesc<1xi64, #mshared, #smem, mutable>

        // CHECK: %[[RET:.*]] = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[AREF_IDX:.*]] = {{.*}})
        %idx = scf.for %i = %c0 to %K step %c4 iter_args(%arefIdx = %c0) -> i32 : i32 {
          %y = arith.muli %i, %c16 : i32

          // CHECK-DAG: %[[AREF_ENTER_MBAR_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]]
          // CHECK: %[[FULL_MBAR:.*]] = ttg.memdesc_subview %[[FULL_MBAR_ALLOC]][%[[AREF_ENTER_MBAR_IDX]]]

          // CHECK: %[[PHASE_DIV:.*]] = arith.divsi %[[AREF_IDX]], %[[C3]] {get_phase}
          // CHECK: %[[PHASE_MOD:.*]] = arith.andi %[[PHASE_DIV]], %[[C1]] {get_phase}
          // CHECK: ttng.wait_barrier %[[FULL_MBAR]], %[[PHASE_MOD]]

          // CHECK-DAG: %[[AREF_SUBVIEW_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {get_view}
          // CHECK: %[[SMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_BUF]][%[[AREF_SUBVIEW_IDX]], {{.*}}, {{.*}}]
          %buf, %tok = ttng.aref_get.enter %aref[%arefIdx] {aref_tag = "1"}: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> none

          // CHECK: {{.*}} = ttg.local_load %[[SMEM_VIEW]]
          %val = ttg.local_load %buf : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>


          // CHECK-DAG: %[[AREF_EXIT_MBAR_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {empty_mbar}
          // CHECK: %[[EXIT_MBAR:.*]] = ttg.memdesc_subview %[[EMPTY_MBAR_ALLOC]][%[[AREF_EXIT_MBAR_IDX]]]
          // CHECK: nvws.arrive_barrier %[[EXIT_MBAR]], tracked_async_op = <none>
          ttng.aref_get.exit %aref[%arefIdx], consumers = [#ttng.aref_consumer<lds>] {aref_tag = "1"}: <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>

          // need store for %val = ttg.local_load %buf not be DCEd
          tt.store %ptr, %val : tensor<128x64x!tt.ptr<f16>, #blocked>

          %arefIdx1 = arith.addi %arefIdx, %c1 : i32
          // CHECK: scf.yield {{.*}} : i32
          scf.yield %arefIdx1 : i32
        }
      ttng.warp_group_return
    } {barId = 1 : i32}}

    ttng.warp_group start_warp(8) num_warps(1) : {{
      %arefGatherBuf = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #nvmma_128, #smem, mutable>
      %arefGather = ttng.aref_create %arefGatherBuf : <[!ttg.memdesc<1x32x128xf16, #nvmma_128, #smem, mutable>]>

      %buf2, %tok2 = ttng.aref_get.enter %arefGather[%c0] {aref_tag = "2"}: <[!ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable> none
      ttng.aref_get.exit %arefGather[%c0], consumers = [#ttng.aref_consumer<lds>] {aref_tag = "2"}: <[!ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>]>

      %buf1, %tok1 = ttng.aref_put.enter %arefGather[%c0] {aref_tag = "3"}: <[!ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable> none
      // CHECK: %[[PTR:.*]] = ttng.tensor_desc_to_tma_ptr {{.*}}
      // CHECK: ttng.async_tma_gather %[[PTR]][{{.*}}2, {{.*}}]
      // CHECK-NEXT: ttng.warp_group_return
      %b = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xf16, #nvmma_128>>, tensor<32xi32, #blockedA>, i32) -> tensor<32x128xf16, #blockedB>
      ttg.local_store %b, %buf1 : tensor<32x128xf16, #blockedB> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
      ttng.aref_put.exit %arefGather[%c0], producers = [#ttng.aref_producer<tmaldg>] {aref_tag = "3"}: <[!ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>]>

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
        // CHECK: %[[RET:.*]] = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[AREF_IDX:.*]] = {{.*}})
        %idx = scf.for %i = %c0 to %K step %c4 iter_args(%arefIdx = %c0) -> i32 : i32 {
          %y = arith.muli %i, %c16 : i32

          // CHECK: %[[SMEM_PUT_ENTER_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {empty_mbar}
          // CHECK: %[[E_SMEM_MBAR:.*]] = ttg.memdesc_subview %[[E_SMEM_MBAR_ALLOC]][%[[SMEM_PUT_ENTER_IDX]]]
          %smemLdg, %s = ttng.aref_put.enter %arefSmem[%arefIdx] {aref_tag = "0"}: <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable> none

          %a = tt.descriptor_load %desc0[%x, %y] : !tt.tensordesc<tensor<128x256xf16, #shared>> -> tensor<128x256xf16, #blocked>
          ttg.local_store %a, %smemLdg : tensor<128x256xf16, #blocked> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
          ttng.aref_put.exit %arefSmem[%arefIdx], producers = [#ttng.aref_producer<tmaldg>] {aref_tag = "0"}: <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>

          // CHECK: %[[TMEM_GET_ENTER_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C2]] {full_mbar}
          // CHECK: %[[F_TMEM_MBAR:.*]] = ttg.memdesc_subview %[[F_TMEM_MBAR_ALLOC]][%[[TMEM_GET_ENTER_IDX]]]

          // CHECK: %[[TPHASE_DIV:.*]] = arith.divsi %[[AREF_IDX]], %[[C2]] {get_phase}
          // CHECK: %[[TPHASE_MOD:.*]] = arith.andi %[[TPHASE_DIV]], %[[C1]] {get_phase}
          // CHECK: ttng.wait_barrier %[[F_TMEM_MBAR]], %[[TPHASE_MOD]]

          // CHECK: %[[TMEM_GET_ENTER_IDX1:.*]] = arith.remsi %[[AREF_IDX]], %[[C2]] {get_view}
          // CHECK: %[[TMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_TMEM_BUF]][%[[TMEM_GET_ENTER_IDX1]], {{.*}}, {{.*}}]
          %tmemLdttm, %t = ttng.aref_get.enter %arefTmem[%arefIdx] {aref_tag = "1"}: <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> none

          // CHECK: {{.*}} = ttng.tmem_load %[[TMEM_VIEW]]
          %val32 = ttng.tmem_load  %tmemLdttm : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>

          // CHECK: %[[TMEM_GET_EXIT_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C2]]
          // CHECK: %[[E_TMEM_MBAR:.*]] = ttg.memdesc_subview %[[E_TMEM_MBAR_ALLOC]][%[[TMEM_GET_EXIT_IDX]]]
          // CHECK: nvws.arrive_barrier %[[E_TMEM_MBAR]], tracked_async_op = <none>
          ttng.aref_get.exit %arefTmem[%arefIdx], consumers = [#ttng.aref_consumer<ldtm>] {aref_tag = "1"} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>

          %val16  = arith.truncf %val32 : tensor<128x256xf32, #blocked> to tensor<128x256xf16, #blocked>

          // CHECK: %[[SMEM_PUT_ENTER_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {empty_mbar}
          // CHECK: %[[E_SMEM_MBAR:.*]] = ttg.memdesc_subview %[[E_SMEM1_MBAR_ALLOC]][%[[SMEM_PUT_ENTER_IDX]]]

          // CHECK: %[[SPHASE_DIV:.*]] = arith.divsi %[[AREF_IDX]], %[[C3]] {put_phase}
          // CHECK: %[[SPHASE_MOD:.*]] = arith.andi %[[SPHASE_DIV]], %[[C1]] {put_phase}
          // CHECK: %[[SPHASE_PUT:.*]] = arith.xori %[[SPHASE_MOD]], %[[C1]] {put_phase}
          // CHECK: ttng.wait_barrier %[[E_SMEM_MBAR]], %[[SPHASE_PUT]]

          // CHECK: %[[SMEM_GET_ENTER_IDX1:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]]
          // CHECK: %[[SMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_SMEM_BUF]][%[[SMEM_GET_ENTER_IDX1]], {{.*}}, {{.*}}]
          %smemSts, %s1 = ttng.aref_put.enter %arefSmem1[%arefIdx] {aref_tag = "2"}: <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable> none

          // CHECK: ttg.local_store %{{.*}}, %[[SMEM_VIEW]]
          ttg.local_store %val16, %smemSts : tensor<128x256xf16, #blocked> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>

          // CHECK: %[[SMEM_PUT_EXIT_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {full_mbar}
          // CHECK: %[[F_SMEM_MBAR:.*]] = ttg.memdesc_subview %[[F_SMEM1_MBAR_ALLOC]][%[[SMEM_PUT_EXIT_IDX]]]
          // CHECK: nvws.arrive_barrier %[[F_SMEM_MBAR]], tracked_async_op = <none>
          ttng.aref_put.exit %arefSmem1[%arefIdx], producers = [#ttng.aref_producer<sts>] {aref_tag = "2"}: <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>

          %smemSts1, %s2 = ttng.aref_get.enter %arefSmem1[%arefIdx] {aref_tag = "2"} : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable> none
          ttng.aref_get.exit %arefSmem1[%arefIdx], consumers = [#ttng.aref_consumer<lds>] {aref_tag = "2"} : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>

          %arefIdx1 = arith.addi %arefIdx, %c1 : i32
          scf.yield %arefIdx1 : i32
        }
      ttng.warp_group_return
    } {barId = 0 : i32}}

    ttng.warp_group start_warp(4) num_warps(4) : {{
        // CHECK: %[[RET:.*]] = scf.for %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args(%[[AREF_IDX:.*]] = {{.*}})
        %idx = scf.for %i = %c0 to %K step %c4 iter_args(%arefIdx = %c0) -> i32 : i32 {

          // CHECK: %[[SMEM_GET_ENTER_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {get_view}
          // CHECK-NEXT: %[[SMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_SMEM_BUF]][%[[SMEM_GET_ENTER_IDX]], {{.*}}, {{.*}}]
          %smemLds, %s = ttng.aref_get.enter %arefSmem[%arefIdx] {aref_tag = "3"}: <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable> none
          // CHECK: {{.*}} = ttg.local_load %[[SMEM_VIEW]]
          %val = ttg.local_load %smemLds : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> tensor<128x256xf16, #blocked>

          // CHECK: %[[SMEM_GET_EXIT_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C3]] {empty_mbar}
          // CHECK: %[[E_SMEM_MBAR:.*]] = ttg.memdesc_subview %[[E_SMEM_MBAR_ALLOC]][%[[SMEM_GET_EXIT_IDX]]]
          // CHECK: nvws.arrive_barrier %[[E_SMEM_MBAR]], tracked_async_op = <none>
          ttng.aref_get.exit %arefSmem[%arefIdx], consumers = [#ttng.aref_consumer<lds>] {aref_tag = "3"} : <[!ttg.memdesc<3x128x256xf16, #shared, #smem, mutable>]>

          %val32 = arith.extf %val : tensor<128x256xf16, #blocked> to tensor<128x256xf32, #blocked>

          // CHECK: %[[TMEM_PUT_ENTER_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C2]] {put_view}
          // CHECK-NEXT: %[[TMEM_VIEW:.*]] = ttg.memdesc_subview %[[AREF_TMEM_BUF]][%[[TMEM_PUT_ENTER_IDX]], {{.*}}, {{.*}}]
          %tmemStm, %t = ttng.aref_put.enter %arefTmem[%arefIdx] {aref_tag = "4"}: <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> none
          // CHECK: ttng.tmem_store {{.*}}, %[[TMEM_VIEW]]
          ttng.tmem_store %val32, %tmemStm, %true : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

          // CHECK: %[[TMEM_PUT_EXIT_IDX:.*]] = arith.remsi %[[AREF_IDX]], %[[C2]] {full_mbar}
          // CHECK: %[[F_TMEM_MBAR:.*]] = ttg.memdesc_subview %[[F_TMEM_MBAR_ALLOC]][%[[TMEM_PUT_EXIT_IDX]]]
          // CHECK: nvws.arrive_barrier %[[F_TMEM_MBAR]], tracked_async_op = <none>
          ttng.aref_put.exit %arefTmem[%arefIdx], producers = [#ttng.aref_producer<sttm>] {aref_tag = "4"} : <[!ttg.memdesc<2x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>


          %arefIdx1 = arith.addi %arefIdx, %c1 : i32
          scf.yield %arefIdx1 : i32
        }

      ttng.warp_group_return
    } {barId = 1 : i32}}

    tt.return
  }
}
