// RUN: triton-opt %s -split-input-file -mlir-print-local-scope -allow-unregistered-dialect -convert-warp-specialize-to-llvm -canonicalize=region-simplify=disabled | FileCheck %s --check-prefixes=COMMON,CHECK
// RUN: triton-opt %s -split-input-file -mlir-print-local-scope -allow-unregistered-dialect -triton-amdgpu-convert-warp-specialize-to-llvm=arch=gfx1250 -canonicalize=region-simplify=disabled | FileCheck %s --check-prefixes=COMMON,AMD

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 11 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @rewrite_barriers
llvm.func @rewrite_barriers() attributes {allocation.offset = 32 : i32} {
  // CHECK-DAG: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: [[C2:%.*]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: [[C3:%.*]] = llvm.mlir.constant(3 : i32)
  // CHECK-DAG: [[C64:%.*]] = llvm.mlir.constant(64 : i32)
  // CHECK-DAG: [[C128:%.*]] = llvm.mlir.constant(128 : i32)

  // CHECK: nvvm.barrier id = [[C2]] number_of_threads = [[C128]]
  // CHECK: nvvm.barrier id = [[C3]] number_of_threads = [[C64]]
  // CHECK: bar.warp.sync

  // CHECK: bb{{[0-9]+}}:
  // CHECK-NEXT: nvvm.barrier id = [[C0]] number_of_threads = [[C128]]
  nvvm.barrier0
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 10>}
  default {
    // CHECK: nvvm.barrier id = [[C0]] number_of_threads = [[C128]]
    nvvm.barrier0
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    nvvm.barrier0
    ttg.warp_return
  }
  partition1() num_warps(2) {
    nvvm.barrier0
    ttg.warp_return
  }
  partition2() num_warps(1) {
    nvvm.barrier0
    ttg.warp_return
  } : () -> ()
  // CHECK: nvvm.barrier id = [[C0]] number_of_threads = [[C128]]
  nvvm.barrier0
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 11 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "hip:gfx1250"} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// AMD-LABEL: @rewrite_barriers
// AMD-DAG: llvm.mlir.global internal @nbar1
// AMD-DAG: llvm.mlir.global internal @nbar2
// AMD-DAG: llvm.mlir.global internal @nbar3
// AMD-DAG: llvm.mlir.global internal @nbar4

llvm.func @rewrite_barriers() attributes {allocation.offset = 32 : i32} {
  // AMD: bb{{[0-9]+}}:
  // AMD-NEXT: rocdl.barrier

  // Check that named barriers are used and that we have the correct counts:
  // AMD-DAG-COUNT-6: rocdl.s.barrier.join
  // AMD-DAG-COUNT-4: rocdl.s.barrier.signal.var {{.*}}, 4
  // AMD-DAG-COUNT-1: rocdl.s.barrier.signal.var {{.*}}, 2
  // AMD-DAG-COUNT-1: rocdl.s.barrier.signal.var {{.*}}, 1
  // AMD-DAG-COUNT-6: rocdl.s.barrier.wait 1

  rocdl.barrier
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 10>}
  default {
    rocdl.barrier
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    rocdl.barrier
    ttg.warp_return
  }
  partition1() num_warps(2) {
    rocdl.barrier
    ttg.warp_return
  }
  partition2() num_warps(1) {
    rocdl.barrier
    ttg.warp_return
  } : () -> ()
  rocdl.barrier
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 11 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @generate_switch_loop
llvm.func @generate_switch_loop() attributes {allocation.offset = 32 : i32} {
  // CHECK-DAG: [[CNEG1:%.*]] = llvm.mlir.constant(-1 : i32)
  // CHECK-DAG: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // COMMON-DAG: [[C4:%.*]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: [[C31:%.*]] = llvm.mlir.constant(31 : i32)
  // CHECK-DAG: [[C32:%.*]] = llvm.mlir.constant(32 : i32)

  // COMMON-DAG: [[C0_i8:%.*]] = llvm.mlir.constant(0 : i8)
  // COMMON-DAG: [[C1_i8:%.*]] = llvm.mlir.constant(1 : i8)
  // COMMON-DAG: [[C2_i8:%.*]] = llvm.mlir.constant(2 : i8)
  // COMMON-DAG: [[C3_i8:%.*]] = llvm.mlir.constant(3 : i8)

  // COMMON-DAG: [[SMEM_ADDR:%.*]] = llvm.mlir.addressof @global_smem

  // CHECK-NEXT: [[TIDX:%.*]] = nvvm.read.ptx.sreg.tid.x
  // CHECK-NEXT: [[WID:%.*]] = llvm.udiv [[TIDX]], [[C32]]
  // CHECK-NEXT: [[WARP_ID:%.*]] = nvvm.shfl.sync idx [[CNEG1]], [[WID]], [[C0]], [[C31]]
  // CHECK-NEXT: [[IS_DEFAULT:%.*]] = llvm.icmp "ult" [[WARP_ID]], [[C4]]
  // CHECK-NEXT: llvm.cond_br [[IS_DEFAULT]], [[BODY:\^.*]], [[SWITCH_LOOP:\^.*]]

  // CHECK: [[SWITCH_LOOP]]:
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
  // CHECK-NEXT: [[REL_WID:%.*]] = llvm.sub [[WARP_ID]], [[C4]]

  // CHECK-NEXT: [[STATE_PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][[[REL_WID]]]
  // CHECK-NEXT: [[STATE:%.*]] = llvm.load [[STATE_PTR]]
  // CHECK-NEXT: llvm.switch [[STATE]] : i8, [[DEFAULT:\^.*]] [
  // CHECK-NEXT: 0: [[PARTITION0:\^.*]],
  // CHECK-NEXT: 1: [[PARTITION1:\^.*]],
  // CHECK-NEXT: 2: [[PARTITION2:\^.*]],
  // CHECK-NEXT: 3: [[EXIT:\^.*]]

  // CHECK: [[DEFAULT]]:
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]] {loop_annotation = #llvm.loop_annotation<licm = <disable = true>>}

  // CHECK: [[EXIT]]:
  // CHECK-NEXT: llvm.return

  // CHECK: [[PARTITION0]]:
  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "partition0"
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]

  // CHECK: [[PARTITION1]]:
  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "partition1"
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]

  // CHECK: [[PARTITION2]]:
  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "partition2"
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]

  // CHECK: [[BODY]]:
  // CHECK-NEXT: "before"
  // CHECK-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32]

  // CHECK-NEXT: llvm.store [[C0_i8]], [[SMEM_BASE]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][2]
  // CHECK-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][3]
  // CHECK-NEXT: llvm.store [[C0_i8]], [[PTR]]

  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][4]
  // CHECK-NEXT: llvm.store [[C1_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][5]
  // CHECK-NEXT: llvm.store [[C1_i8]], [[PTR]]

  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][6]
  // CHECK-NEXT: llvm.store [[C2_i8]], [[PTR]]

  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.br [[DEFAULT_PARTITION:\^.*]]
  // CHECK: [[DEFAULT_PARTITION]]:
  // CHECK-NEXT: "default"
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.br [[AFTER:\^.*]]

  // AMD: [[WID:%.*]] = llvm.call_intrinsic "llvm.amdgcn.wave.id"
  // AMD-NEXT: [[IS_DEFAULT:%.*]] = llvm.icmp "ult" [[WID]], [[C4]]
  // AMD-NEXT: llvm.cond_br [[IS_DEFAULT]], [[BODY:\^bb[0-9]+]], [[SWITCH_LOOP:\^bb[0-9]+]]

  // AMD: [[SWITCH_LOOP]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
  // AMD-NEXT: [[REL_WID:%.*]] = llvm.sub [[WID]], [[C4]]

  // AMD-NEXT: [[STATE_PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][[[REL_WID]]]
  // AMD-NEXT: [[STATE:%.*]] = llvm.load [[STATE_PTR]]
  // AMD-NEXT: llvm.switch [[STATE]] : i8, [[DEFAULT:\^bb[0-9]+]] [
  // AMD-NEXT: 0: [[PARTITION0:\^bb[0-9]+]],
  // AMD-NEXT: 1: [[PARTITION1:\^bb[0-9]+]],
  // AMD-NEXT: 2: [[PARTITION2:\^bb[0-9]+]],
  // AMD-NEXT: 3: [[EXIT:\^bb[0-9]+]]

  // AMD: [[DEFAULT]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[SWITCH_LOOP]] {loop_annotation = #llvm.loop_annotation<licm = <disable = true>>}

  // AMD: [[EXIT]]:
  // AMD-NEXT: llvm.return

  // AMD: [[PARTITION0]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: "partition0"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[SWITCH_LOOP]]

  // AMD: [[PARTITION1]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: "partition1"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[SWITCH_LOOP]]

  // AMD: [[PARTITION2]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: "partition2"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[SWITCH_LOOP]]

  // AMD: [[BODY]]:
  // AMD-NEXT: "before"
  // AMD-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32]

  // AMD-NEXT: llvm.store [[C0_i8]], [[SMEM_BASE]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // AMD-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][2]
  // AMD-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][3]
  // AMD-NEXT: llvm.store [[C0_i8]], [[PTR]]

  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][4]
  // AMD-NEXT: llvm.store [[C1_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][5]
  // AMD-NEXT: llvm.store [[C1_i8]], [[PTR]]

  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][6]
  // AMD-NEXT: llvm.store [[C2_i8]], [[PTR]]

  // AMD: rocdl.barrier
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[DEFAULT_PARTITION:\^bb[0-9]+]]
  // AMD: [[DEFAULT_PARTITION]]:
  // AMD-NEXT: "default"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[AFTER:\^bb[0-9]+]]

  "before"() : () -> ()
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 10>}
  default {
    "default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "partition0"() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(2) {
    "partition1"() : () -> ()
    ttg.warp_return
  }
  partition2() num_warps(1) {
    "partition2"() : () -> ()
    ttg.warp_return
  } : () -> ()
  // CHECK: [[AFTER]]:
  // CHECK-NEXT: "after"

  // CHECK-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32]

  // CHECK-NEXT: llvm.store [[C3_i8]], [[SMEM_BASE]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][2]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][3]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][4]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][5]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][6]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.return

  // AMD: [[AFTER:\^bb[0-9]+]]:
  // AMD-NEXT: "after"

  // AMD-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32]

  // AMD-NEXT: llvm.store [[C3_i8]], [[SMEM_BASE]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // AMD-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][2]
  // AMD-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][3]
  // AMD-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][4]
  // AMD-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][5]
  // AMD-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][6]
  // AMD-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.return

  "after"() : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @pass_captures
llvm.func @pass_captures() attributes {allocation.offset = 32 : i32} {
  // CHECK-DAG: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // COMMON-DAG: [[SMEM_ADDR:%.*]] = llvm.mlir.addressof @global_smem

  // CHECK: ^bb4:
  // CHECK-NEXT: [[ARG0_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: [[ARG0:%.*]] = llvm.load [[ARG0_PTR]] {alignment = 1 : i64}
  // CHECK-NEXT: [[ARG1_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: [[ARG1:%.*]] = llvm.load [[ARG1_PTR]] {alignment = 1 : i64}
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "use"([[ARG0]], [[ARG1]])
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])

  // CHECK: ^bb5:
  // CHECK: [[INS:%.*]]:2 = "produce"()
  // CHECK: [[ARG0_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: llvm.store [[INS]]#0, [[ARG0_PTR]] {alignment = 1 : i64}
  // CHECK-NEXT: [[ARG1_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: llvm.store [[INS]]#1, [[ARG1_PTR]] {alignment = 1 : i64}
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])

  // AMD: ^bb4:
  // AMD-NEXT: [[ARG0_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // AMD-NEXT: [[ARG0:%.*]] = llvm.load [[ARG0_PTR]] {alignment = 1 : i64}
  // AMD-NEXT: [[ARG1_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // AMD-NEXT: [[ARG1:%.*]] = llvm.load [[ARG1_PTR]] {alignment = 1 : i64}
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: "use"([[ARG0]], [[ARG1]])
  // AMD-NEXT: rocdl.barrier

  // AMD: ^bb5:
  // AMD: [[INS:%.*]]:2 = "produce"()
  // AMD: [[ARG0_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // AMD-NEXT: llvm.store [[INS]]#0, [[ARG0_PTR]] {alignment = 1 : i64}
  // AMD-NEXT: [[ARG1_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // AMD-NEXT: llvm.store [[INS]]#1, [[ARG1_PTR]] {alignment = 1 : i64}
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: rocdl.barrier

  %ins:2 = "produce"() : () -> (i32, i64)
  ttg.warp_specialize(%ins#0, %ins#1) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg2: i32, %arg3: i64) num_warps(4) {
    "use"(%arg2, %arg3) : (i32, i64) -> ()
    ttg.warp_return
  } : (i32, i64) -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 18 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @partition_warpid_order
llvm.func @partition_warpid_order() attributes {allocation.offset = 32 : i32} {
  // COMMON-DAG: [[SMEM_ADDR:%.*]] = llvm.mlir.addressof @global_smem
  // COMMON-DAG: [[C0_i8:%.*]] = llvm.mlir.constant(0 : i8)
  // COMMON-DAG: [[C1_i8:%.*]] = llvm.mlir.constant(1 : i8)
  // COMMON-DAG: [[C2_i8:%.*]] = llvm.mlir.constant(2 : i8)

  // COMMON: llvm.switch
  // COMMON-NEXT: 0: [[PARTITION0:\^.*]],
  // COMMON-NEXT: 1: [[PARTITION1:\^.*]],
  // COMMON-NEXT: 2: [[PARTITION2:\^.*]],
  // COMMON-NEXT: 3: [[EXIT:\^.*]]

  // COMMON: [[PARTITION0]]:
  // COMMON: "ws0_partition0"
  // COMMON: [[PARTITION1]]:
  // COMMON: "ws0_partition1"
  // COMMON: [[PARTITION2]]:
  // COMMON: "ws0_partition2"

  // COMMON: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]]

  // COMMON-NEXT: llvm.store [[C1_i8]], [[SMEM_BASE]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[1]
  // COMMON-NEXT: llvm.store [[C1_i8]], [[PTR]]

  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // COMMON-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // COMMON-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // COMMON-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // COMMON-NEXT: llvm.store [[C0_i8]], [[PTR]]

  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // COMMON-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // COMMON-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[8]
  // COMMON-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[9]
  // COMMON-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[10]
  // COMMON-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[11]
  // COMMON-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[12]
  // COMMON-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[13]
  // COMMON-NEXT: llvm.store [[C2_i8]], [[PTR]]
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 6, 4, 10>}
  default {
    "ws0_default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "ws0_partition0"() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(2) {
    "ws0_partition1"() : () -> ()
    ttg.warp_return
  }
  partition2() num_warps(8) {
    "ws0_partition2"() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @multiple_specialize
llvm.func @multiple_specialize() attributes {allocation.offset = 32 : i32} {
  // COMMON-DAG: llvm.mlir.addressof @global_smem
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // COMMON-DAG: [[C0_i8:%.*]] = llvm.mlir.constant(0 : i8)
  // COMMON-DAG: [[C1_i8:%.*]] = llvm.mlir.constant(1 : i8)
  // COMMON-DAG: [[C2_i8:%.*]] = llvm.mlir.constant(2 : i8)
  // COMMON-DAG: [[C3_i8:%.*]] = llvm.mlir.constant(3 : i8)
  // COMMON-DAG: [[C4_i8:%.*]] = llvm.mlir.constant(4 : i8)
  // COMMON-DAG: [[C5_i8:%.*]] = llvm.mlir.constant(5 : i8)
  // COMMON-DAG: [[Cn1_i8:%.*]] = llvm.mlir.constant(-1 : i8)

  // CHECK: llvm.switch
  // CHECK-NEXT: 0: [[WS0_PARTITION0:\^.*]],
  // CHECK-NEXT: 1: [[WS0_PARTITION1:\^.*]],
  // CHECK-NEXT: 2: [[WS0_PARTITION2:\^.*]],
  // CHECK-NEXT: 3: [[WS1_PARTITION0:\^.*]],
  // CHECK-NEXT: 4: [[WS1_PARTITION1:\^.*]],
  // CHECK-NEXT: 5: [[WS3_PARTITION0:\^.*]],
  // CHECK-NEXT: 6: [[EXIT:\^.*]]

  // CHECK: [[WS0_PARTITION0]]:
  // CHECK: "ws0_partition0"
  // CHECK: [[WS0_PARTITION1]]:
  // CHECK: "ws0_partition1"
  // CHECK: [[WS0_PARTITION2]]:
  // CHECK: "ws0_partition2"
  // CHECK: [[WS1_PARTITION0]]:
  // CHECK: "ws1_partition0"
  // CHECK: [[WS1_PARTITION1]]:
  // CHECK: "ws1_partition1"
  // CHECK: [[WS3_PARTITION0]]:
  // CHECK: "ws3_partition0"

  // CHECK: getelementptr
  // CHECK-NEXT: llvm.store [[C0_i8]], [[SMEM_BASE:%[0-9]+]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.store [[C1_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.store [[C1_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK: "ws0_default"

  // AMD: llvm.switch
  // AMD-NEXT: 0: [[WS0_PARTITION0:\^bb[0-9]+]],
  // AMD-NEXT: 1: [[WS0_PARTITION1:\^bb[0-9]+]],
  // AMD-NEXT: 2: [[WS0_PARTITION2:\^bb[0-9]+]],
  // AMD-NEXT: 3: [[WS1_PARTITION0:\^bb[0-9]+]],
  // AMD-NEXT: 4: [[WS1_PARTITION1:\^bb[0-9]+]],
  // AMD-NEXT: 5: [[WS3_PARTITION0:\^bb[0-9]+]],
  // AMD-NEXT: 6: [[EXIT:\^bb[0-9]+]]

  // AMD: [[WS0_PARTITION0]]:
  // AMD: "ws0_partition0"
  // AMD: [[WS0_PARTITION1]]:
  // AMD: "ws0_partition1"
  // AMD: [[WS0_PARTITION2]]:
  // AMD: "ws0_partition2"
  // AMD: [[WS1_PARTITION0]]:
  // AMD: "ws1_partition0"
  // AMD: [[WS1_PARTITION1]]:
  // AMD: "ws1_partition1"
  // AMD: [[WS3_PARTITION0]]:
  // AMD: "ws3_partition0"

  // AMD: getelementptr
  // AMD-NEXT: llvm.store [[C0_i8]], [[SMEM_BASE:%[0-9]+]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // AMD-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // AMD-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // AMD-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // AMD-NEXT: llvm.store [[C1_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // AMD-NEXT: llvm.store [[C1_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // AMD-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // AMD-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // AMD: rocdl.barrier
  // AMD: "ws0_default"

  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 10>}
  default {
    "ws0_default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "ws0_partition0"() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(2) {
    "ws0_partition1"() : () -> ()
    ttg.warp_return
  }
  partition2() num_warps(1) {
    "ws0_partition2"() : () -> ()
    ttg.warp_return
  } : () -> ()

  // CHECK: getelementptr
  // CHECK-NEXT: llvm.store [[C4_i8]], [[SMEM_BASE:%[0-9]+]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[C4_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.store [[C4_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.store [[C4_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK: "ws1_default"

  // AMD: getelementptr
  // AMD-NEXT: llvm.store [[C4_i8]], [[SMEM_BASE:%[0-9]+]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // AMD-NEXT: llvm.store [[C4_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // AMD-NEXT: llvm.store [[C4_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // AMD-NEXT: llvm.store [[C4_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // AMD-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // AMD-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // AMD-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // AMD-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // AMD: rocdl.barrier
  // AMD: "ws1_default"

  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 8, 4>}
  default {
    "ws1_default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "ws1_partition0"() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(4) {
    "ws1_partition1"() : () -> ()
    ttg.warp_return
  } : () -> ()

  // CHECK: getelementptr
  // CHECK-NEXT: llvm.store [[Cn1_i8]], [[SMEM_BASE:%[0-9]+]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[1]
  // CHECK-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK: "ws2_default"

  // AMD: getelementptr
  // AMD-NEXT: llvm.store [[Cn1_i8]], [[SMEM_BASE:%[0-9]+]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[1]
  // AMD-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // AMD-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // AMD-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // AMD-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // AMD-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // AMD-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // AMD-NEXT: llvm.store [[Cn1_i8]], [[PTR]]
  // AMD: rocdl.barrier
  // AMD: "ws2_default"

  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32>}
  default {
    "ws2_default"() : () -> ()
    ttg.warp_yield
  } : () -> ()

  // CHECK: getelementptr
  // CHECK-NEXT: llvm.store [[C5_i8]], [[SMEM_BASE:%[0-9]+]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK: "ws3_default"

  // AMD: getelementptr
  // AMD-NEXT: llvm.store [[C5_i8]], [[SMEM_BASE:%[0-9]+]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // AMD-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // AMD-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // AMD-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // AMD-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // AMD-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // AMD-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // AMD-NEXT: llvm.store [[C5_i8]], [[PTR]]
  // AMD: rocdl.barrier
  // AMD: "ws3_default"

  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    "ws3_default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(8) {
    "ws3_partition0"() : () -> ()
    ttg.warp_return
  }: () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @cfg
llvm.func @cfg() attributes {allocation.offset = 32 : i32} {
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)

  // COMMON: [[SWITCH_LOOP:\^bb1]]:
  // COMMON: llvm.switch
  // COMMON-NEXT: 0: [[PARTITION:\^.*]],
  // COMMON-NEXT: 1: [[EXIT:\^.*]]

  // CHECK: [[PARTITION]]:
  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "something"()[[[A:\^.*]], [[B:\^.*]]]
  // CHECK: [[A]]:
  // CHECK-NEXT: "A"
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]
  // CHECK: [[B]]:
  // CHECK-NEXT: "B"
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]

  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK: llvm.br [[DEFAULT:\^.*]]
  // CHECK: [[DEFAULT]]:
  // CHECK-NEXT: "something"()[[[A:\^.*]], [[B:\^.*]]]
  // CHECK: [[A]]:
  // CHECK-NEXT: "A"
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.br [[AFTER:\^.*]]
  // CHECK: [[B]]:
  // CHECK-NEXT: "B"
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: llvm.br [[AFTER]]

  // AMD: [[PARTITION]]:
  // AMD: rocdl.barrier
  // AMD-NEXT: "something"()[[[A:\^bb[0-9]+]], [[B:\^bb[0-9]+]]]
  // AMD: [[A]]:
  // AMD-NEXT: "A"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[SWITCH_LOOP]]
  // AMD: [[B]]:
  // AMD-NEXT: "B"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[SWITCH_LOOP]]

  // AMD: rocdl.barrier
  // AMD-NEXT: rocdl.barrier
  // AMD: llvm.br [[DEFAULT:\^bb[0-9]+]]
  // AMD: [[DEFAULT]]:
  // AMD-NEXT: "something"()[[[A:\^bb[0-9]+]], [[B:\^bb[0-9]+]]]
  // AMD: [[A]]:
  // AMD-NEXT: "A"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[AFTER:\^bb[0-9]+]]
  // AMD: [[B]]:
  // AMD-NEXT: "B"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[AFTER]]

  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    "something"()[^A, ^B] : () -> ()
  ^A:
   "A"() : () -> ()
    ttg.warp_yield
  ^B:
   "B"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "something"()[^A, ^B] : () -> ()
  ^A:
   "A"() : () -> ()
    ttg.warp_return
  ^B:
   "B"() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @no_captures
llvm.func @no_captures() attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.total-num-warps" = 6 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @type_conversion_results
// COMMON-NOT: !tt.ptr<i32>
// COMMON-NOT: unrealized_conversion_cast
llvm.func @type_conversion_results() attributes {allocation.offset = 0 : i32} {
  // COMMON: [[CAP:%.*]] = "produce"
  %cap = "produce"() : () -> !llvm.ptr<1>
  %0 = builtin.unrealized_conversion_cast %cap : !llvm.ptr<1> to !tt.ptr<i32>
  %1 = ttg.warp_specialize(%0) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    // COMMON: llvm.br [[AFTER:\^.*]]([[CAP]] : !llvm.ptr<1>)
    ttg.warp_yield %0 : !tt.ptr<i32>
  }
  partition0(%arg1: !tt.ptr<i32>) num_warps(2) {
    %3 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<i32> to !llvm.ptr<1>
    %4 = llvm.load %3 : !llvm.ptr<1> -> i32
    ttg.warp_return
  } : (!tt.ptr<i32>) -> !tt.ptr<i32>
  // COMMON: [[AFTER]]([[OUT:%.*]]: !llvm.ptr<1>):
  %2 = builtin.unrealized_conversion_cast %1 : !tt.ptr<i32> to !llvm.ptr<1>
  // COMMON-NEXT: "use"([[OUT]])
  "use"(%2) : (!llvm.ptr<1>) -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.total-num-warps" = 6 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @capture_function_arg
llvm.func @capture_function_arg(%arg0: i32) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%arg0) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg1: i32) num_warps(1) {
    // COMMON: "use"(%arg0)
    "use"(%arg1) : (i32) -> ()
    ttg.warp_return
  } : (i32) -> ()
  llvm.return
}

// COMMON-LABEL: @type_conversion_func_arg
llvm.func @type_conversion_func_arg(%arg0: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<1> to !tt.ptr<i32>
  ttg.warp_specialize(%0) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg1: !tt.ptr<i32>) num_warps(1) {
    %1 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<i32> to !llvm.ptr<1>
    // COMMON: "use"(%arg0)
    "use"(%1) : (!llvm.ptr<1>) -> ()
    ttg.warp_return
  } : (!tt.ptr<i32>) -> ()
  llvm.return
}

// COMMON-LABEL: @trivial_remat
llvm.func @trivial_remat() attributes {allocation.offset = 0 : i32} {
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // COMMON-DAG: [[CAP0:%.*]] = llvm.mlir.constant(0 : i32)
  // COMMON-DAG: [[CAP1:%.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>

  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  ttg.warp_specialize(%0, %1) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: i32, %arg1: !llvm.ptr<3>) num_warps(1) {
  // CHECK: ^bb4:
    // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
    // CHECK-NEXT: "use"([[CAP0]], [[CAP1]])
  // AMD: ^bb4:
    // AMD-NEXT: rocdl.barrier
    // AMD-NEXT: "use"([[CAP0]], [[CAP1]])
    "use"(%arg0, %arg1) : (i32, !llvm.ptr<3>) -> ()
    // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
    // AMD-NEXT: rocdl.barrier
    ttg.warp_return
  } : (i32, !llvm.ptr<3>) -> ()
  llvm.return
}

// COMMON-LABEL: @remat_subgraph
llvm.func @remat_subgraph(%arg0: i32, %arg1: i32) attributes {allocation.offset = 0 : i32} {
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // COMMON-DAG: [[ADDR:%.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>

  %0 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  %1 = llvm.getelementptr %0[%arg0] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
  %2 = llvm.add %arg0, %arg1 : i32
  %3 = llvm.mul %2, %arg1 : i32
  %4 = llvm.urem %2, %3 : i32
  ttg.warp_specialize(%1, %4) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg2: !llvm.ptr<3>, %arg3: i32) num_warps(1) {
  // CHECK: ^bb4:
    // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
    // CHECK-NEXT: [[ADD:%.*]] = llvm.add %arg0, %arg1 : i32
    // CHECK-NEXT: [[MUL:%.*]] = llvm.mul [[ADD]], %arg1 : i32
    // CHECK-NEXT: [[UREM:%.*]] = llvm.urem [[ADD]], [[MUL]] : i32
    // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[ADDR]][%arg0]
    // CHECK-NEXT: "use"([[PTR]], [[UREM]])
  // AMD: ^bb4:
    // AMD-NEXT: rocdl.barrier
    // AMD-NEXT: [[ADD:%.*]] = llvm.add %arg0, %arg1 : i32
    // AMD-NEXT: [[MUL:%.*]] = llvm.mul [[ADD]], %arg1 : i32
    // AMD-NEXT: [[UREM:%.*]] = llvm.urem [[ADD]], [[MUL]] : i32
    // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[ADDR]][%arg0]
    // AMD-NEXT: "use"([[PTR]], [[UREM]])
    "use"(%arg2, %arg3) : (!llvm.ptr<3>, i32) -> ()
    // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
    // AMD-NEXT: rocdl.barrier
    ttg.warp_return
  } : (!llvm.ptr<3>, i32) -> ()
  llvm.return
}

}

// -----

module attributes {ttg.maxnreg = 80 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.total-num-warps" = 16 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @dynamic_register_reallocation
llvm.func @dynamic_register_reallocation() attributes {allocation.offset = 0 : i32} {
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)

  // CHECK: cond_br %{{.*}}, [[ENTRY:\^.*]], [[SWITCH_LOOP:\^.*]]

  // CHECK: [[SWITCH_LOOP]]:
  // CHECK-NEXT: nvvm.setmaxregister decrease 24
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK: llvm.switch
  // CHECK-NEXT: 0: [[PARTITION0:\^.*]],
  // CHECK-NEXT: 1: [[PARTITION1:\^.*]],
  // CHECK-NEXT: 2: [[PARTITION2:\^.*]],
  // CHECK-NEXT: 3: [[EXIT:\^.*]]

  // CHECK: [[PARTITION0]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 80
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "partition0"()
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: nvvm.setmaxregister decrease 24

  // CHECK: [[PARTITION1]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 48
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "partition1"()
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: nvvm.setmaxregister decrease 24

  // CHECK: [[PARTITION2]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 128
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "partition2"()
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: nvvm.setmaxregister decrease 24

  // CHECK: [[ENTRY]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 248

  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: setmaxregister decrease 152
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK: "default"
  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: setmaxregister increase 248

  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 12>, actualRegisters = array<i32: 152, 80, 48, 128>}
  default {
    "default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "partition0"() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(4) {
    "partition1"() : () -> ()
    ttg.warp_return
  }
  partition2() num_warps(4) {
    "partition2"() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {ttg.maxnreg = 128 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.total-num-warps" = 16 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @dynamic_register_reallocation
llvm.func @dynamic_register_reallocation_overalloc() attributes {allocation.offset = 0 : i32} {
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)

  // CHECK: cond_br %{{.*}}, [[ENTRY:\^.*]], [[SWITCH_LOOP:\^.*]]

  // CHECK: [[SWITCH_LOOP]]:
  // CHECK-NEXT: nvvm.setmaxregister decrease 80
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK: llvm.switch
  // CHECK-NEXT: 0: [[PARTITION0:\^.*]],
  // CHECK-NEXT: 1: [[PARTITION1:\^.*]],
  // CHECK-NEXT: 2: [[PARTITION2:\^.*]],
  // CHECK-NEXT: 3: [[EXIT:\^.*]]

  // CHECK: [[PARTITION0]]:
  // CHECK-NEXT: nvvm.setmaxregister decrease 24
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "partition0"()
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: nvvm.setmaxregister increase 80

  // CHECK: [[PARTITION1]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 192
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "partition1"()
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: nvvm.setmaxregister decrease 80

  // CHECK: [[PARTITION2]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 192
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: "partition2"()
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: nvvm.setmaxregister decrease 80

  // CHECK: [[ENTRY]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 256

  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: setmaxregister decrease 104
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK: "default"
  // CHECK: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NEXT: setmaxregister increase 256

  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 12>, actualRegisters = array<i32: 104, 24, 192, 192>}
  default {
    "default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "partition0"() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(4) {
    "partition1"() : () -> ()
    ttg.warp_return
  }
  partition2() num_warps(4) {
    "partition2"() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}
