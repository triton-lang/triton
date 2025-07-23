// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-aref | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  //CHECK-LABEL: @aref_lowering
  tt.func @aref_lowering(%d : !ttg.memdesc<3x64x16xf16, #shared0, #tmem>,
                         %e : !ttg.memdesc<3x16x32xf16, #shared0, #smem>,
                         %cond : i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %lb = arith.constant 0 : i32
    // CHECK:   [[C3:%.*]] = arith.constant 3 : i32
    // CHECK:   [[C0:%.*]] = arith.constant 0 : i32
    // CHECK:   [[C1:%.*]] = arith.constant 1 : i32
    %ub = arith.constant 4 : i32

    // CHECK:        [[EMPTY0:%.*]] = ttg.local_alloc
    // CHECK-NEXT:   [[FULL0:%.*]] = ttg.local_alloc
    // CHECK-NEXT:   scf.for
    // CHECK-NEXT:     [[EMPTYSLICE:%.*]] = ttg.memdesc_subview [[EMPTY0]]
    // CHECK-NEXT:     ttng.init_barrier [[EMPTYSLICE]], 1
    // CHECK-NEXT:     [[FULLSLICE:%.*]] = ttg.memdesc_subview [[FULL0]]
    // CHECK-NEXT:     ttng.init_barrier [[FULLSLICE]], 129
    // CHECK-NEXT:   }
    %aref0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

    // CHECK:        [[EMPTY1:%.*]] = ttg.local_alloc
    // CHECK-NEXT:   [[FULL1:%.*]] = ttg.local_alloc
    // CHECK-NEXT:   scf.for
    // CHECK-NEXT:     [[EMPTYSLICE:%.*]] = ttg.memdesc_subview [[EMPTY1]]
    // CHECK-NEXT:     ttng.init_barrier [[EMPTYSLICE]], 256
    // CHECK-NEXT:     [[FULLSLICE:%.*]] = ttg.memdesc_subview [[FULL1]]
    // CHECK-NEXT:     ttng.init_barrier [[FULLSLICE]], 128
    // CHECK-NEXT:   }
    %aref1 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

    nvws.warp_group
    partition0  num_warps(4) {
      // CHECK: [[IDX:%.*]]:4 = scf.for [[I:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[C1:%.*]] iter_args([[IDX0:%.*]] = [[C0]], [[IDX1:%.*]] = [[C0]], [[IDX2:%.*]] = [[C0]], [[IDX3:%.*]] = [[C0]])
      scf.for %i = %lb to %ub step %c1_i32 : i32{

        // CHECK-NEXT: [[EMPTYIDX:%.*]] = arith.remsi [[IDX0]], [[C3]]
        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_subview [[EMPTY0]][[[EMPTYIDX]]]
        // CHECK-NEXT: [[PHASE_DIV:%.*]] = arith.divsi [[IDX0]], [[C3]]
        // CHECK-NEXT: [[PHASE_AND:%.*]] = arith.andi [[PHASE_DIV]], [[C1]]
        // CHECK-NEXT: [[PHASE_XOR:%.*]] = arith.xori [[PHASE_AND]], [[C1]]
        // CHECK-NEXT: ttng.wait_barrier [[EMPTYMBAR]], [[PHASE_XOR]]
        %1:2 = nvws.aref.put.enter %aref0[%c0_i32] {aref_tag = "put0"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>

        // CHECK-NEXT: [[STAGE:%.*]] = arith.remsi [[IDX0]], [[C3]]
        // CHECK-NEXT: [[BUFA:%.*]] = ttg.memdesc_subview %arg0[[[STAGE]],{{.*}},{{.*}}]
        // CHECK-NEXT: [[BUFB:%.*]] = ttg.memdesc_subview %arg1[[[STAGE]],{{.*}},{{.*}}]
        // CHECK-NEXT: [[FULLIDX:%.*]] = arith.remsi [[IDX2]], [[C3]]
        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_subview [[FULL0]][[[FULLIDX]]]
        // CHECK-NEXT: ttng.barrier_expect [[FULLMBAR]], 0
        // CHECK-NEXT: [[IDX0a:%.*]] = arith.addi [[IDX0]], [[C1]]
        // CHECK-NEXT: "tma_load"([[BUFA]])
        // CHECK-NEXT: "cp_async"([[BUFB]])
        "tma_load"(%1#0) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>) -> ()
        "cp_async"(%1#1) : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        // CHECK-NEXT: [[FULLIDX:%.*]] = arith.remsi [[IDX2]], [[C3]]
        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_subview [[FULL0]][[[FULLIDX]]]
        // CHECK-NEXT: nvws.async_complete [[FULLMBAR]], async_op = <tma_load>
        // CHECK-NEXT: nvws.async_complete [[FULLMBAR]], async_op = <cp_async>
        // CHECK-NEXT: [[IDX2a:%.*]] = arith.addi [[IDX2]], [[C1]]
        nvws.aref.put.exit %aref0[%c0_i32] [#nvws.async_op<tma_load>, #nvws.async_op<cp_async>] {aref_tag = "put0"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

        // CHECK-NEXT: [[IDX13:%.*]]:2 = scf.if
        scf.if %cond {

          // CHECK: arith.remsi [[IDX1]], [[C3]]
          // CHECK: arith.divsi [[IDX1]], [[C3]]
          // CHECK-NEXT: arith.andi {{.*}}, [[C1]]
          // CHECK-NEXT: arith.xori
          // CHECK-NEXT: ttng.wait_barrier
          // CHECK: [[IDX1a:%.*]] = arith.addi [[IDX1]], [[C1]]
          %2:2 = nvws.aref.put.enter %aref1[%c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
          "tmem_store"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

          // CHECK: arith.remsi [[IDX3]], [[C3]]
          // CHECK: [[IDX3a:%.*]] = arith.addi [[IDX3]], [[C1]]
          nvws.aref.put.exit %aref1[%c0_i32] [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

          // CHECK: scf.yield [[IDX1a]], [[IDX3a]]
        }
        // CHECK-NEXT: } else {
        // CHECK-NEXT:   scf.yield [[IDX1]], [[IDX3]]
        // CHECK-NEXT: }

        // CHECK: scf.yield [[IDX0a]], [[IDX13]]#0, [[IDX2a]], [[IDX13]]#1
      }

      // CHECK: [[IDX1:%.*]]:2 = scf.if
      scf.if %cond {

        // CHECK: arith.remsi [[IDX]]#0, [[C3]]
        // CHECK: arith.divsi [[IDX]]#0, [[C3]]
        // CHECK: [[IDX0a:%.*]] = arith.addi [[IDX]]#0, [[C1]]
        %1:2 = nvws.aref.put.enter %aref0[%c0_i32] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
        "tma_load"(%1#0) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>) -> ()
        "cp_async"(%1#1) : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        // CHECK: arith.remsi [[IDX]]#2, [[C3]]
        // CHECK: [[IDX2a:%.*]] = arith.addi [[IDX]]#2, [[C1]]
        nvws.aref.put.exit %aref0[%c0_i32] [#nvws.async_op<tma_load>, #nvws.async_op<cp_async>] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
      }

      // CHECK: arith.remsi [[IDX]]#1, [[C3]]
      // CHECK: arith.divsi [[IDX]]#1, [[C3]]
      %1:2 = nvws.aref.put.enter %aref1[%c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
      "tmem_store"(%1#0, %1#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      // CHECK: arith.remsi [[IDX]]#3, [[C3]]
      nvws.aref.put.exit %aref1[%c0_i32] [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
      nvws.warp_group.return
    }
    partition1 num_warps(8) {
      // CHECK: [[IDX:%.*]]:4 = scf.for [[I:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[C1:%.*]] iter_args([[IDX0:%.*]] = [[C0]], [[IDX1:%.*]] = [[C0]], [[IDX2:%.*]] = [[C0]], [[IDX3:%.*]] = [[C0]])
      scf.for %i = %lb to %ub step %c1_i32 : i32{

        // CHECK-NEXT: [[FULLIDX:%.*]] = arith.remsi [[IDX0]], [[C3]]
        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_subview [[FULL0]][[[FULLIDX]]]
        // CHECK-NEXT: [[PHASE_DIV:%.*]] = arith.divsi [[IDX0]], [[C3]]
        // CHECK-NEXT: [[PHASE_AND:%.*]] = arith.andi [[PHASE_DIV]], [[C1]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]], [[PHASE_AND]]
        %2:2 = nvws.aref.get.enter %aref0[%c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>

        // CHECK-NEXT: [[STAGE:%.*]] = arith.remsi [[IDX0]], [[C3]]
        // CHECK-NEXT: [[BUFA:%.*]] = ttg.memdesc_subview %arg0[[[STAGE]],{{.*}},{{.*}}]
        // CHECK-NEXT: [[BUFB:%.*]] = ttg.memdesc_subview %arg1[[[STAGE]],{{.*}},{{.*}}]
        // CHECK-NEXT: arith.addi
        // CHECK-NEXT: "tc5mma"([[BUFA]], [[BUFB]])
        "tc5mma"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        // CHECK-NEXT: [[EMPTYIDX:%.*]] = arith.remsi [[IDX2]], [[C3]]
        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_subview [[EMPTY0]][[[EMPTYIDX]]]
        // CHECK-NEXT: nvws.async_complete [[EMPTYMBAR]], async_op = <tc5mma>
        // CHECK-NEXT: arith.addi
        nvws.aref.get.exit %aref0[%c0_i32] [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

        // CHECK: [[IDX13:%.*]]:2 = scf.if
        scf.if %cond {
          // CHECK: arith.remsi [[IDX1]], [[C3]]
          // CHECK: arith.divsi [[IDX1]], [[C3]]
          // CHECK-NEXT: arith.andi {{.*}}, [[C1]]
          // CHECK-NEXT: ttng.wait_barrier
          %3:2 = nvws.aref.get.enter %aref1[%c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
          "tmem_load"(%3#0, %3#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

          // CHECK: arith.remsi [[IDX3]], [[C3]]
          // CHECK-NEXT: ttg.memdesc_subview
          // CHECK-NEXT: nvws.async_complete {{.*}}, async_op = <none>
          nvws.aref.get.exit %aref1[%c0_i32] [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
        }
        // CHECK: } else {
        // CHECK-NEXT:   scf.yield [[IDX1]], [[IDX3]]
        // CHECK-NEXT: }

        // CHECK: scf.yield {{.*}}, [[IDX13]]#0, {{.*}}, [[IDX13]]#1
      }
      scf.if %cond {
        // CHECK: arith.remsi [[IDX]]#0, [[C3]]
        // CHECK: arith.divsi [[IDX]]#0, [[C3]]
        // CHECK-NEXT: arith.andi {{.*}}, [[C1]]
        // CHECK-NEXT: ttng.wait_barrier
        %2:2 = nvws.aref.get.enter %aref0[%c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
        "tc5mma"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        // CHECK: arith.remsi [[IDX]]#2, [[C3]]
        nvws.aref.get.exit %aref0[%c0_i32] [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
      }
      // CHECK: } else {
      // CHECK-NEXT:   scf.yield [[IDX]]#0, [[IDX]]#2
      // CHECK-NEXT: }

      %2:2 = nvws.aref.get.enter %aref1[%c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
      "tmem_load"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      nvws.aref.get.exit %aref1[%c0_i32] [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
      nvws.warp_group.return
    }
    tt.return
  }
}
