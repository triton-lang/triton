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
    // CHECK-NEXT:     ttng.init_barrier [[FULLSLICE]], 2
    // CHECK-NEXT:   }
    %aref0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

    // CHECK:        [[EMPTY1:%.*]] = ttg.local_alloc
    // CHECK-NEXT:   [[FULL1:%.*]] = ttg.local_alloc
    // CHECK-NEXT:   scf.for
    // CHECK-NEXT:     [[EMPTYSLICE:%.*]] = ttg.memdesc_subview [[EMPTY1]]
    // CHECK-NEXT:     ttng.init_barrier [[EMPTYSLICE]], 1
    // CHECK-NEXT:     [[FULLSLICE:%.*]] = ttg.memdesc_subview [[FULL1]]
    // CHECK-NEXT:     ttng.init_barrier [[FULLSLICE]], 1
    // CHECK-NEXT:   }
    %aref1 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

    nvws.warp_group
    partition0  num_warps(4) {
      // CHECK: [[IDX:%.*]]:6 = scf.for [[I:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[C1:%.*]] iter_args([[S0:%.*]] = [[C0]], [[P0:%.*]] = [[C1]], [[S1:%.*]] = [[C0]], [[P1:%.*]] = [[C1]], [[S2:%.*]] = [[C0]],  [[S3:%.*]] = [[C0]])
      scf.for %i = %lb to %ub step %c1_i32 : i32{

        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_subview [[EMPTY0]][[[S0]]]
        // CHECK-NEXT: ttng.wait_barrier [[EMPTYMBAR]], [[P0]]
        %1:2 = nvws.aref.put.enter %aref0[%c0_i32, %c0_i32] {aref_tag = "put0"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>

        // CHECK-NEXT: [[BUFA:%.*]] = ttg.memdesc_subview %arg0[[[S0]],{{.*}},{{.*}}]
        // CHECK-NEXT: [[BUFB:%.*]] = ttg.memdesc_subview %arg1[[[S0]],{{.*}},{{.*}}]
        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_subview [[FULL0]][[[S2]]]
        // CHECK-NEXT: ttng.barrier_expect [[FULLMBAR]], 0
        // CHECK-NEXT: [[S0a:%.*]] = arith.addi [[S0]], [[C1]]
        // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S0a]], [[C3]]
        // CHECK-NEXT: [[S0b:%.*]] = arith.select [[CMP]], [[C0]], [[S0a]]
        // CHECK-NEXT: [[P0a:%.*]] = arith.xori [[P0]], [[C1]]
        // CHECK-NEXT: [[P0b:%.*]] = arith.select [[CMP]], [[P0a]], [[P0]]
        // CHECK-NEXT: "tma_load"([[BUFA]])
        // CHECK-NEXT: "sts"([[BUFB]])
        "tma_load"(%1#0) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>) -> ()
        "sts"(%1#1) : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_subview [[FULL0]][[[S2]]]
        // CHECK-NEXT: ttng.arrive_barrier [[FULLMBAR]], 1
        // CHECK-NEXT: [[S2a:%.*]] = arith.addi [[S2]], [[C1]]
        // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S2a]], [[C3]]
        // CHECK-NEXT: [[S2b:%.*]] = arith.select [[CMP]], [[C0]], [[S2a]]
        nvws.aref.put.exit %aref0[%c0_i32] [#nvws.async_op<tma_load>, #nvws.async_op<none>] {aref_tag = "put0"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

        // CHECK-NEXT: [[SP1S3:%.*]]:3 = scf.if
        scf.if %cond {

          // CHECK-NEXT: [[BAR:%.*]] = ttg.memdesc_subview {{.*}}[[[S1]]]
          // CHECK-NEXT: ttng.wait_barrier [[BAR]], [[P1]]
          // CHECK: [[S1a:%.*]] = arith.addi [[S1]], [[C1]]
          // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S1a]], [[C3]]
          // CHECK-NEXT: [[S1b:%.*]] = arith.select [[CMP]], [[C0]], [[S1a]]
          // CHECK-NEXT: [[P1a:%.*]] = arith.xori [[P1]], [[C1]]
          // CHECK-NEXT: [[P1b:%.*]] = arith.select [[CMP]], [[P1a]], [[P1]]

          %2:2 = nvws.aref.put.enter %aref1[%c0_i32, %c0_i32] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
          "tmem_store"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

          // CHECK: [[BAR:%.*]] = ttg.memdesc_subview {{.*}}[[[S3]]]
          // CHECK-NEXT: ttng.arrive_barrier [[BAR]], 1
          // CHECK: [[S3a:%.*]] = arith.addi [[S3]], [[C1]]
          // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S3a]], [[C3]]
          // CHECK-NEXT: [[S3b:%.*]] = arith.select [[CMP]], [[C0]], [[S3a]]
          nvws.aref.put.exit %aref1[%c0_i32] [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

          // CHECK: scf.yield [[S1b]], [[P1b]], [[S3b]]
        }
        // CHECK-NEXT: } else {
        // CHECK-NEXT:   scf.yield [[S1]], [[P1]], [[S3]]
        // CHECK-NEXT: }

        // CHECK: scf.yield [[S0b]], [[P0b]], [[SP1S3]]#0, [[SP1S3]]#1, [[S2b]], [[SP1S3]]#2
        // CHECK-NEXT: }
      }

      // CHECK-NEXT: [[IDX1:%.*]]:3 = scf.if
      scf.if %cond {

        // CHECK-NEXT: [[BAR:%.*]] = ttg.memdesc_subview {{.*}}[[[IDX]]#0]
        // CHECK-NEXT: ttng.wait_barrier [[BAR]], [[IDX]]#1
        %1:2 = nvws.aref.put.enter %aref0[%c0_i32, %c0_i32] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
        "tma_load"(%1#0) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>) -> ()
        "sts"(%1#1) : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
        //CHECK: sts

        // CHECK: [[BAR:%.*]] = ttg.memdesc_subview {{.*}}[[[IDX]]#4]
        // CHECK-NEXT: ttng.arrive_barrier [[BAR]]
        nvws.aref.put.exit %aref0[%c0_i32] [#nvws.async_op<tma_load>, #nvws.async_op<none>] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
      }

      // CHECK: [[BAR:%.*]] = ttg.memdesc_subview {{.*}}[[[IDX]]#2]
      // CHECK-NEXT: ttng.wait_barrier [[BAR]], [[IDX]]#3
      %1:2 = nvws.aref.put.enter %aref1[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
      "tmem_store"(%1#0, %1#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      // CHECK: [[BAR:%.*]] = ttg.memdesc_subview {{.*}}[[[IDX]]#5]
      // CHECK-NEXT: ttng.arrive_barrier [[BAR]], 1
      nvws.aref.put.exit %aref1[%c0_i32] [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
      nvws.warp_group.return
    }
    partition1 num_warps(8) {
      // CHECK: [[IDX:%.*]]:6 = scf.for [[I:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[C1:%.*]] iter_args([[S0:%.*]] = [[C0]], [[P0:%.*]] = [[C0]], [[S1:%.*]] = [[C0]], [[P1:%.*]] = [[C0]], [[S2:%.*]] = [[C0]], [[S3:%.*]] = [[C0]])
      scf.for %i = %lb to %ub step %c1_i32 : i32{

        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_subview [[FULL0]][[[S0]]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]], [[P0]]
        %2:2 = nvws.aref.get.enter %aref0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>

        // CHECK-NEXT: [[BUFA:%.*]] = ttg.memdesc_subview %arg0[[[S0]],{{.*}},{{.*}}]
        // CHECK-NEXT: [[BUFB:%.*]] = ttg.memdesc_subview %arg1[[[S0]],{{.*}},{{.*}}]
        // CHECK-NEXT: arith.addi
        // CHECK-NEXT: arith.cmpi
        // CHECK-NEXT: arith.select
        // CHECK-NEXT: arith.xori
        // CHECK-NEXT: arith.select
        // CHECK-NEXT: "tc5mma"([[BUFA]], [[BUFB]])
        "tc5mma"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_subview [[EMPTY0]][[[S2]]]
        // CHECK-NEXT: ttng.tc_gen5_commit [[EMPTYMBAR]]
        // CHECK-NEXT: arith.addi
        // CHECK-NEXT: arith.cmpi
        // CHECK-NEXT: arith.select
        // CHECK-NOT: arith.xori
        // CHECK-NOT: arith.select
        nvws.aref.get.exit %aref0[%c0_i32] [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

        // CHECK: [[IDX13:%.*]]:3 = scf.if
        scf.if %cond {
          // CHECK: [[BAR:%.*]] = ttg.memdesc_subview {{.*}}[[[S1]]]
          // CHECK-NEXT: ttng.wait_barrier  [[BAR]], [[P1]]
          %3:2 = nvws.aref.get.enter %aref1[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
          "tmem_load"(%3#0, %3#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
          // CHECK: tmem_load

          // CHECK-NEXT: [[BAR:%.*]] = ttg.memdesc_subview {{.*}}[[[S3]]]
          // CHECK-NEXT: ttng.arrive_barrier [[BAR]], 1
          nvws.aref.get.exit %aref1[%c0_i32] [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
        }
        // CHECK: } else {
        // CHECK-NEXT:   scf.yield [[S1]], [[P1]], [[S3]]
        // CHECK-NEXT: }

        // CHECK: scf.yield {{.*}}, {{.*}}, [[IDX13]]#0, [[IDX13]]#1, {{.*}}, [[IDX13]]#2
      }
      scf.if %cond {
        // CHECK: [[BAR:%.*]] = ttg.memdesc_subview {{.*}}[[[IDX]]#0]
        // CHECK-NEXT: ttng.wait_barrier  [[BAR]], [[IDX]]#1
        %2:2 = nvws.aref.get.enter %aref0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
        "tc5mma"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        // CHECK: [[BAR:%.*]] = ttg.memdesc_subview {{.*}}[[[IDX]]#4]
        // CHECK-NEXT: ttng.tc_gen5_commit  [[BAR]]
        nvws.aref.get.exit %aref0[%c0_i32] [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
      }
      // CHECK: } else {
      // CHECK-NEXT:   scf.yield [[IDX]]#0, [[IDX]]#1, [[IDX]]#4
      // CHECK-NEXT: }

      %2:2 = nvws.aref.get.enter %aref1[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
      "tmem_load"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      nvws.aref.get.exit %aref1[%c0_i32] [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #tmem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
      nvws.warp_group.return
    }
    tt.return
  }
}
