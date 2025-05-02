// RUN: triton-opt %s -split-input-file --triton-nvidia-ttng-wg-to-aref-if | FileCheck %s

module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @ttg_wg_2warps(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    nvvm.barrier0
    // CHECK: %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK-DAG: %[[CNST32:.*]] = arith.constant  32
    // CHECK-DAG: %[[WID:.*]] = arith.divsi %[[TID]], %[[CNST32]]
    // CHECK-DAG: %[[CNST4:.*]] = arith.constant 4
    // CHECK-DAG: %[[P1:.*]] = arith.cmpi sge, %[[WID]], %[[CNST4]]
    // CHECK-DAG: %[[CNST8:.*]] = arith.constant 8
    // CHECK-DAG: %[[P2:.*]] = arith.cmpi slt, %[[WID]], %[[CNST8]]
    // CHECK-DAG: %[[P3:.*]] = arith.andi %[[P1]], %[[P2]]
    // CHECK-DAG: scf.if %[[P3]]
    ttng.warp_group start_warp(4) num_warps(4) :  {{
      ttng.warp_group_return
    } {barId = 1 : i32, groups = [@nvws.tma_load]}}

    // CHECK: %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK-DAG: %[[CNST32:.*]] = arith.constant  32
    // CHECK-DAG: %[[WID:.*]] = arith.divsi %[[TID]], %[[CNST32]]
    // CHECK-DAG: %[[CNST0:.*]] = arith.constant 0
    // CHECK-DAG: %[[P1:.*]] = arith.cmpi sge, %[[WID]], %[[CNST0]]
    // CHECK-DAG: %[[CNST4:.*]] = arith.constant 4
    // CHECK-DAG: %[[P2:.*]] = arith.cmpi slt, %[[WID]], %[[CNST4]]
    // CHECK-DAG: %[[P3:.*]] = arith.andi %[[P1]], %[[P2]]
    // CHECK-DAG: scf.if %[[P3]]
    ttng.warp_group start_warp(0) num_warps(4) : {{
      ttng.warp_group_return
    } {barId = 2 : i32, groups = [@nvws.mma]}}
    tt.return
  }
}

// -----

module attributes {nvwm.epilogue = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.mma = {num_warps = 1 : i32, start_warp = 4 : i32}, nvws.tma_load = {num_warps = 1 : i32, start_warp = 5 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @ttg_wg_3warps(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    nvvm.barrier0
    // CHECK: %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK-DAG: %[[CNST32:.*]] = arith.constant  32
    // CHECK-DAG: %[[WID:.*]] = arith.divsi %[[TID]], %[[CNST32]]
    // CHECK-DAG: %[[CNST5:.*]] = arith.constant 5
    // CHECK-DAG: %[[P1:.*]] = arith.cmpi sge, %[[WID]], %[[CNST5]]
    // CHECK-DAG: %[[CNST6:.*]] = arith.constant 6
    // CHECK-DAG: %[[P2:.*]] = arith.cmpi slt, %[[WID]], %[[CNST6]]
    // CHECK-DAG: %[[P3:.*]] = arith.andi %[[P1]], %[[P2]]
    // CHECK-DAG: scf.if %[[P3]]
    ttng.warp_group start_warp(5) num_warps(1) :  {{
      ttng.warp_group_return
    } {barId = 1 : i32, groups = [@nvws.tma_load]}}

    // CHECK: %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK-DAG: %[[CNST32:.*]] = arith.constant  32
    // CHECK-DAG: %[[WID:.*]] = arith.divsi %[[TID]], %[[CNST32]]
    // CHECK-DAG: %[[CNST0:.*]] = arith.constant 0
    // CHECK-DAG: %[[P1:.*]] = arith.cmpi sge, %[[WID]], %[[CNST0]]
    // CHECK-DAG: %[[CNST4:.*]] = arith.constant 4
    // CHECK-DAG: %[[P2:.*]] = arith.cmpi slt, %[[WID]], %[[CNST4]]
    // CHECK-DAG: %[[P3:.*]] = arith.andi %[[P1]], %[[P2]]
    // CHECK-DAG: scf.if %[[P3]]
    ttng.warp_group start_warp(0) num_warps(4) : {{
      ttng.warp_group_return
    } {barId = 2 : i32, groups = [@nvws.epilogue]}}

    // CHECK: %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK-DAG: %[[CNST32:.*]] = arith.constant  32
    // CHECK-DAG: %[[WID:.*]] = arith.divsi %[[TID]], %[[CNST32]]
    // CHECK-DAG: %[[CNST4:.*]] = arith.constant 4
    // CHECK-DAG: %[[P1:.*]] = arith.cmpi sge, %[[WID]], %[[CNST4]]
    // CHECK-DAG: %[[CNST5:.*]] = arith.constant 5
    // CHECK-DAG: %[[P2:.*]] = arith.cmpi slt, %[[WID]], %[[CNST5]]
    // CHECK-DAG: %[[P3:.*]] = arith.andi %[[P1]], %[[P2]]
    // CHECK-DAG: scf.if %[[P3]]
    ttng.warp_group start_warp(4) num_warps(1) : {{
      ttng.warp_group_return
    } {barId = 3 : i32, groups = [@nvws.mma]}}
    tt.return
  }
}