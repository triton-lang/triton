// RUN: triton-opt %s -split-input-file -mlir-print-local-scope -allow-unregistered-dialect -convert-warp-specialize-to-llvm -canonicalize=region-simplify=disabled | FileCheck %s --check-prefixes=COMMON,CHECK
// RUN: triton-opt %s -split-input-file -mlir-print-local-scope -allow-unregistered-dialect -triton-amdgpu-convert-warp-specialize-to-llvm=gfx-arch=gfx1250 -canonicalize=region-simplify=disabled | FileCheck %s --check-prefixes=COMMON,AMD

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

  // CHECK: llvm.cond_br {{.*}}, [[DEFAULT:\^bb[0-9]+]], {{\^bb[0-9]+}}
  // CHECK: nvvm.barrier id = [[C2]] number_of_threads = [[C128]] aligned = false
  // CHECK: nvvm.barrier id = [[C3]] number_of_threads = [[C64]] aligned = false
  // CHECK: bar.warp.sync

  // CHECK: [[DEFAULT]]:
  // CHECK-NEXT: nvvm.barrier id = [[C0]] number_of_threads = [[C128]] aligned = false
  nvvm.barrier
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 10>}
  default {
    // CHECK: nvvm.barrier id = [[C0]] number_of_threads = [[C128]] aligned = false
    nvvm.barrier
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    nvvm.barrier
    ttg.warp_return
  }
  partition1() num_warps(2) {
    nvvm.barrier
    ttg.warp_return
  }
  partition2() num_warps(1) {
    nvvm.barrier
    ttg.warp_return
  } : () -> ()
  // CHECK: nvvm.barrier id = [[C0]] number_of_threads = [[C128]] aligned = false
  nvvm.barrier
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 11 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "hip:gfx1250"} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// AMD-LABEL: @rewrite_barriers
// AMD-DAG: llvm.mlir.global internal @nbar1() {addr_space = 15 : i32}
// AMD-DAG: llvm.mlir.global internal @nbar2() {addr_space = 15 : i32}
// AMD-DAG: llvm.mlir.global internal @nbar3() {addr_space = 15 : i32}
// AMD-DAG: llvm.mlir.global internal @nbar4() {addr_space = 15 : i32}

llvm.func @rewrite_barriers() attributes {allocation.offset = 32 : i32} {
  // AMD-DAG: [[NBAR1:%.*]] = llvm.mlir.addressof @nbar1 : !llvm.ptr<15>
  // AMD-DAG: [[NBAR2:%.*]] = llvm.mlir.addressof @nbar2 : !llvm.ptr<15>
  // AMD-DAG: [[NBAR3:%.*]] = llvm.mlir.addressof @nbar3 : !llvm.ptr<15>

  // AMD: llvm.cond_br %{{[0-9]+}}, [[DEFAULT:\^bb[0-9]+]], [[WORKERS:\^bb[0-9]+]]

  // AMD: [[WORKERS]]:
  // AMD-NEXT: rocdl.barrier
  // AMD: llvm.switch {{.*}} : i32, [[PARTITION0:\^bb[0-9]+]] [
  // AMD-NEXT: 8: [[PARTITION1:\^bb[0-9]+]],
  // AMD: 10: [[PARTITION2:\^bb[0-9]+]]

  // AMD: [[JOIN:\^bb[0-9]+]]:
  // AMD-NEXT: rocdl.barrier

  // partition0
  // AMD: [[PARTITION0]]:
  // AMD-NEXT: rocdl.s.barrier.join [[NBAR2]]
  // AMD-NEXT: rocdl.s.barrier.signal.var [[NBAR2]] member_cnt = 4
  // AMD-NEXT: rocdl.s.barrier.wait id = 1
  // AMD-NEXT: llvm.br [[JOIN]]

  // partition1
  // AMD: [[PARTITION1]]:
  // AMD-NEXT: rocdl.s.barrier.join [[NBAR3]]
  // AMD-NEXT: rocdl.s.barrier.signal.var [[NBAR3]] member_cnt = 2
  // AMD-NEXT: rocdl.s.barrier.wait id = 1
  // AMD-NEXT: llvm.br [[JOIN]]

  // partition2
  // AMD: [[PARTITION2]]:
  // AMD-NEXT: rocdl.wave.barrier
  // AMD-NEXT: llvm.br [[JOIN]]

  // default partition
  // AMD: [[DEFAULT]]:
  // AMD: rocdl.barrier
  // AMD-NEXT: rocdl.s.barrier.join [[NBAR1]]
  // AMD-NEXT: rocdl.s.barrier.signal.var [[NBAR1]] member_cnt = 4
  // AMD-NEXT: rocdl.s.barrier.wait id = 1

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

// CHECK: llvm.func internal @inner_func_nw4_ws(%arg0: i32)
llvm.func internal @inner_func_nw4() attributes {"ws_num_warps" = 4 : i32} {
  // CHECK: [[C128:%.*]] = llvm.mlir.constant(128 : i32)
  // CHECK: "use.barrier_id"(%arg0) : (i32) -> ()
  %barrier_id = nvg.warp_group_barrier_id
  "use.barrier_id"(%barrier_id) : (i32) -> ()
  // CHECK: nvvm.barrier id = %arg0 number_of_threads = [[C128]] aligned = false
  // CHECK: llvm.call @inner_func_nw4_ws(%arg0) : (i32) -> ()
  nvvm.barrier
  llvm.call @inner_func_nw4() : () -> ()
  llvm.return
}

// CHECK: llvm.func internal @inner_func_nw2_ws(%arg0: f32 {some.attr = "some_value"}, %arg1: i32)
llvm.func internal @inner_func_nw2(%arg0: f32 {some.attr = "some_value"}) attributes {"ws_num_warps" = 2 : i32} {
  // CHECK: [[C64:%.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: nvvm.barrier id = %arg1 number_of_threads = [[C64]] aligned = false
  nvvm.barrier
  llvm.return
}

// CHECK: llvm.func internal @inner_func_nw1_ws(%arg0: i32)
llvm.func internal @inner_func_nw1() attributes {"ws_num_warps" = 1 : i32} {
  // CHECK: nvvm.bar.warp.sync
  nvvm.barrier
  llvm.return
}

llvm.func @__libdevice_function()

// CHECK: llvm.func @rewrite_barriers()
llvm.func @rewrite_barriers() attributes {allocation.offset = 32 : i32} {
  // CHECK-DAG: [[CST:%.*]] = llvm.mlir.constant({{.*}}) : f32
  // CHECK-DAG: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: [[C2:%.*]] = llvm.mlir.constant(2 : i32)
  // CHECK-DAG: [[C3:%.*]] = llvm.mlir.constant(3 : i32)
  // CHECK-DAG: [[C4:%.*]] = llvm.mlir.constant(4 : i32)
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 10>}
  default {
    llvm.call @inner_func_nw4() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    // CHECK: call @inner_func_nw4_ws([[C2]]) : (i32) -> ()
    llvm.call @inner_func_nw4() : () -> ()
    // CHEC: call @__libdevice_function() : () -> ()
    llvm.call @__libdevice_function() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(2) {
    // CHECK: call @inner_func_nw2_ws([[CST]], [[C3]]) : (f32, i32) -> ()
    %cst = llvm.mlir.constant(4.2 : f32) : f32
    llvm.call @inner_func_nw2(%cst) : (f32) -> ()
    ttg.warp_return
  }
  partition2() num_warps(1) {
    // CHECK: call @inner_func_nw1_ws([[C4]]) : (i32) -> ()
    llvm.call @inner_func_nw1() : () -> ()
    ttg.warp_return
  } : () -> ()
  // CHECK: call @inner_func_nw4_ws([[C0]]) : (i32) -> ()

  // CHECK: call @inner_func_nw4_ws([[C0]]) : (i32) -> ()
  llvm.call @inner_func_nw4() : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 11 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @generate_switch_loop
llvm.func @generate_switch_loop(%again: i1) attributes {allocation.offset = 32 : i32} {
  // CHECK-DAG: [[CNEG1:%.*]] = llvm.mlir.constant(-1 : i32)
  // CHECK-DAG: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // COMMON-DAG: [[C4:%.*]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: [[C31:%.*]] = llvm.mlir.constant(31 : i32)
  // CHECK-DAG: [[C32:%.*]] = llvm.mlir.constant(32 : i32)

  // COMMON-DAG: [[C0_i8:%.*]] = llvm.mlir.constant(0 : i8)
  // COMMON-DAG: [[C1_i8:%.*]] = llvm.mlir.constant(1 : i8)
  // AMD-DAG: [[C2_i8:%.*]] = llvm.mlir.constant(2 : i8)
  // AMD-DAG: [[C3_i8:%.*]] = llvm.mlir.constant(3 : i8)

  // COMMON-DAG: [[SMEM_ADDR:%.*]] = llvm.mlir.addressof @global_smem

  // CHECK-NEXT: [[TIDX:%.*]] = nvvm.read.ptx.sreg.tid.x
  // CHECK-NEXT: [[WID:%.*]] = llvm.udiv [[TIDX]], [[C32]]
  // CHECK-NEXT: [[WARP_ID:%.*]] = nvvm.shfl.sync idx [[CNEG1]], [[WID]], [[C0]], [[C31]]
  // CHECK-NEXT: [[IS_DEFAULT:%.*]] = llvm.icmp "ult" [[WARP_ID]], [[C4]]
  // CHECK-NEXT: llvm.cond_br [[IS_DEFAULT]], [[BODY:\^.*]], [[SWITCH_LOOP:\^.*]]

  // CHECK: [[SWITCH_LOOP]]:
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
  // CHECK-NEXT: [[GROUP:%.*]] = llvm.udiv [[WARP_ID]], [[C4]]
  // CHECK-NEXT: [[SLOT:%.*]] = llvm.sub [[GROUP]], [[C1]]

  // CHECK-NEXT: [[STATE_PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][[[SLOT]]]
  // CHECK-NEXT: [[STATE:%.*]] = llvm.load [[STATE_PTR]]
  // CHECK-NEXT: llvm.switch [[STATE]] : i8, [[EXIT:\^.*]] [
  // CHECK-NEXT: 0: [[SPLIT:\^bb[0-9]+]]
  // CHECK-NEXT: ]

  // CHECK: [[EXIT]]:
  // CHECK-NEXT: llvm.return
  // CHECK: [[SPLIT]]:
  // CHECK-NEXT: llvm.switch [[WARP_ID]] : i32, [[PARTITION0:\^bb[0-9]+]] [
  // CHECK-NEXT: 8: [[PARTITION1:\^bb[0-9]+]],
  // CHECK-NEXT: 9: [[PARTITION1]],
  // CHECK-NEXT: 10: [[PARTITION2:\^bb[0-9]+]]
  // CHECK-NEXT: ]
  // CHECK: [[JOIN:\^bb[0-9]+]]:
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]] loop_annotation = <licm = <disable = true>>

  // CHECK: [[PARTITION0]]:
  // CHECK: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: "partition0"
  // CHECK-NEXT: llvm.br [[JOIN]]

  // CHECK: [[PARTITION1]]:
  // CHECK: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: "partition1"
  // CHECK-NEXT: llvm.br [[JOIN]]

  // CHECK: [[PARTITION2]]:
  // CHECK: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: "partition2"
  // CHECK-NEXT: llvm.br [[JOIN]]

  // CHECK: [[BODY]]:
  // CHECK-NEXT: "before"
  // CHECK: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32]

  // CHECK-NEXT: llvm.store [[C0_i8]], [[SMEM_BASE]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: "default"
  // CHECK-NEXT: llvm.br [[AFTER:\^.*]]

  // AMD: [[WID:%.*]] = rocdl.wave.id : i32
  // AMD-NEXT: [[IS_DEFAULT:%.*]] = llvm.icmp "ult" [[WID]], [[C4]]
  // AMD-NEXT: llvm.cond_br [[IS_DEFAULT]], [[BODY:\^bb[0-9]+]], [[SWITCH_LOOP:\^bb[0-9]+]]

  // AMD: [[SWITCH_LOOP]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
  // AMD-NEXT: [[REL_WID:%.*]] = llvm.sub [[WID]], [[C4]]

  // AMD-NEXT: [[STATE_PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][[[REL_WID]]]
  // AMD-NEXT: [[STATE:%.*]] = llvm.load [[STATE_PTR]]
  // AMD-NEXT: llvm.switch [[STATE]] : i8, [[EXIT:\^bb[0-9]+]] [
  // AMD-NEXT: 0: [[PARTITION0:\^bb[0-9]+]],
  // AMD-NEXT: 1: [[PARTITION1:\^bb[0-9]+]],
  // AMD-NEXT: 2: [[PARTITION2:\^bb[0-9]+]]
  // AMD-NEXT: ]

  // AMD: [[EXIT]]:
  // AMD-NEXT: llvm.return

  // AMD: [[PARTITION0]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: "partition0"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[SWITCH_LOOP]] loop_annotation = <licm = <disable = true>>

  // AMD: [[PARTITION1]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: "partition1"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[SWITCH_LOOP]] loop_annotation = <licm = <disable = true>>

  // AMD: [[PARTITION2]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: "partition2"
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[SWITCH_LOOP]] loop_annotation = <licm = <disable = true>>

  // AMD: [[BODY]]:
  // AMD-NEXT: "before"
  // AMD: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32]

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
  // AMD-NEXT: "default"
  // AMD-NEXT: llvm.br [[AFTER:\^bb[0-9]+]]

  "before"() : () -> ()
  %false = llvm.mlir.constant(false) : i1
  llvm.br ^loop(%again : i1)
^loop(%repeat: i1):
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
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: "after"

  // CHECK: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32]

  // CHECK-NEXT: llvm.store [[C1_i8]], [[SMEM_BASE]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[C1_i8]], [[PTR]]
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: llvm.return

  // AMD: [[AFTER:\^bb[0-9]+]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: "after"

  // AMD: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][32]

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
  llvm.cond_br %repeat, ^loop(%false : i1), ^done
^done:
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
  // COMMON: llvm.cond_br {{.*}}, [[DEFAULT:\^bb[0-9]+]], {{\^bb[0-9]+}}
  // COMMON: llvm.switch {{.*}} : i32, [[PAYLOAD:\^bb[0-9]+]] [
  // COMMON-NEXT: 6: [[PADDING:\^bb[0-9]+]]
  // COMMON-NEXT: 7: [[PADDING]]
  // COMMON: [[JOIN:\^bb[0-9]+]]:
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // AMD-NEXT: rocdl.barrier

  // CHECK: [[PAYLOAD]]:
  // CHECK-NEXT: [[ARG0_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: [[ARG0:%.*]] = llvm.load [[ARG0_PTR]] <alignment = 1>
  // CHECK-NEXT: [[ARG1_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: [[ARG1:%.*]] = llvm.load [[ARG1_PTR]] <alignment = 1>
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: "use"([[ARG0]], [[ARG1]])
  // CHECK-NEXT: llvm.br [[JOIN]]
  // CHECK: [[PADDING]]:
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: llvm.br [[JOIN]]

  // CHECK: [[DEFAULT]]:
  // CHECK: [[INS:%.*]]:2 = "produce"()
  // CHECK: [[ARG0_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: llvm.store [[INS]]#0, [[ARG0_PTR]] <alignment = 1>
  // CHECK-NEXT: [[ARG1_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: llvm.store [[INS]]#1, [[ARG1_PTR]] <alignment = 1>
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false

  // AMD: [[PAYLOAD]]:
  // AMD-NEXT: [[ARG0_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // AMD-NEXT: [[ARG0:%.*]] = llvm.load [[ARG0_PTR]] <alignment = 1>
  // AMD-NEXT: [[ARG1_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // AMD-NEXT: [[ARG1:%.*]] = llvm.load [[ARG1_PTR]] <alignment = 1>
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: "use"([[ARG0]], [[ARG1]])
  // AMD-NEXT: llvm.br [[JOIN]]
  // AMD: [[PADDING]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.br [[JOIN]]

  // AMD: [[DEFAULT]]:
  // AMD: [[INS:%.*]]:2 = "produce"()
  // AMD: [[ARG0_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // AMD-NEXT: llvm.store [[INS]]#0, [[ARG0_PTR]] <alignment = 1>
  // AMD-NEXT: [[ARG1_PTR:%.*]] = llvm.getelementptr [[SMEM_ADDR]][0, 1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // AMD-NEXT: llvm.store [[INS]]#1, [[ARG1_PTR]] <alignment = 1>
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: rocdl.barrier

  %ins:2 = "produce"() : () -> (i32, i64)
  ttg.warp_specialize(%ins#0, %ins#1) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 6>}
  default {
    ttg.warp_yield
  }
  partition0(%arg2: i32, %arg3: i64) num_warps(2) {
    "use"(%arg2, %arg3) : (i32, i64) -> ()
    ttg.warp_return
  }
  partition1(%arg2: i32, %arg3: i64) num_warps(2) {
    ttg.warp_return
  } : (i32, i64) -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 18 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @partition_warpid_order
llvm.func @partition_warpid_order(%enabled: i1) attributes {allocation.offset = 32 : i32} {
  // COMMON-DAG: [[SMEM_ADDR:%.*]] = llvm.mlir.addressof @global_smem
  // COMMON-DAG: [[C0_i8:%.*]] = llvm.mlir.constant(0 : i8)
  // AMD-DAG: [[C1_i8:%.*]] = llvm.mlir.constant(1 : i8)
  // AMD-DAG: [[C2_i8:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-DAG: [[EXIT_STATE:%.*]] = llvm.mlir.constant(1 : i8)
  // AMD-DAG: [[EXIT_STATE:%.*]] = llvm.mlir.constant(3 : i8)

  // COMMON: [[STATE:%.*]] = llvm.load {{.*}} : !llvm.ptr<3> -> i8
  // COMMON-NEXT: llvm.switch [[STATE]] : i8
  // CHECK-NEXT: 0: [[GROUP:\^bb[0-9]+]]
  // CHECK-NEXT: ]
  // Partitions crossing a quartet boundary share the same command.
  // CHECK: [[GROUP]]:
  // CHECK-NEXT: llvm.switch {{.*}} : i32, [[PARTITION1:\^bb[0-9]+]] [
  // CHECK-NEXT: 6: [[PARTITION0:\^bb[0-9]+]],
  // CHECK-COUNT-3: {{[7-9]}}: [[PARTITION0]]
  // CHECK-NEXT: 10: [[PARTITION2:\^bb[0-9]+]],
  // CHECK-COUNT-7: 1{{[1-7]}}: [[PARTITION2]]
  // CHECK-NEXT: ]
  // AMD-NEXT: 0: [[PARTITION0:\^.*]],
  // AMD-NEXT: 1: [[PARTITION1:\^.*]],
  // AMD-NEXT: 2: [[PARTITION2:\^.*]]

  // COMMON: [[PARTITION0]]:
  // COMMON: "ws0_partition0"
  // COMMON: [[PARTITION1]]:
  // COMMON: "ws0_partition1"
  // COMMON: [[PARTITION2]]:
  // COMMON: "ws0_partition2"

  // COMMON: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]]

  // CHECK-NEXT: llvm.store [[C0_i8]], [[SMEM_BASE]]
  // AMD-NEXT: llvm.store [[C1_i8]], [[SMEM_BASE]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[1]
  // CHECK-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // AMD-NEXT: llvm.store [[C1_i8]], [[PTR]]

  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // COMMON-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // COMMON-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // COMMON-NEXT: llvm.store [[C0_i8]], [[PTR]]
  // CHECK-NEXT: nvvm.barrier
  // AMD-COUNT-2: llvm.store [[C0_i8]],
  // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][6]
  // AMD-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // AMD-COUNT-7: llvm.store [[C2_i8]],
  // AMD-NEXT: rocdl.barrier
  llvm.cond_br %enabled, ^specialize, ^done
^specialize:
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
  llvm.br ^done
^done:
  // COMMON: llvm.store [[EXIT_STATE]],
  // CHECK: nvvm.barrier
  // AMD: rocdl.barrier
  // COMMON-NEXT: llvm.return
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 16 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @warpid_warp_specialize
llvm.func @warpid_warp_specialize(%setup: i1) attributes {allocation.offset = 32 : i32} {
  // CHECK-DAG: [[C4:%.*]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: [[C12:%.*]] = llvm.mlir.constant(12 : i32)

  // Both paths reach the site. The larger partition starts at warp 4.
  // CHECK: [[WID:%.*]] = nvvm.shfl.sync idx
  // AMD: [[WID:%.*]] = rocdl.wave.id
  // COMMON: llvm.switch [[WID]] : i32, [[P1:\^bb[0-9]+]] [
  // COMMON-NEXT: 12: [[P0:\^bb[0-9]+]],
  // COMMON: 15: [[P0]]

  // Partition warp IDs are rewritten to be relative in this pass, while
  // keeping ttg.warp_id for NVGPUToLLVM to lower later.
  // COMMON: [[P0]]:
  // CHECK: %{{.*}} = ttg.warp_id
  // CHECK-NEXT: [[REL0:%.*]] = llvm.sub %{{.*}}, [[C12]] : i32
  // CHECK-NEXT: "use"([[REL0]]) : (i32) -> ()

  // COMMON: [[P1]]:
  // CHECK: %{{.*}} = ttg.warp_id
  // CHECK-NEXT: [[REL1:%.*]] = llvm.sub %{{.*}}, [[C4]] : i32
  // CHECK-NEXT: "use"([[REL1]]) : (i32) -> ()

  llvm.cond_br %setup, ^setup, ^launch
^setup:
  %0 = ttg.warp_id
  "use"(%0) : (i32) -> ()
  llvm.br ^launch
^launch:
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 12, 4>}
  default {
    %1 = ttg.warp_id
    "use"(%1) : (i32) -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %1 = ttg.warp_id
    "use"(%1) : (i32) -> ()
    ttg.warp_return
  }
  partition1() num_warps(8) {
    %1 = ttg.warp_id
    "use"(%1) : (i32) -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {ttg.maxnreg = 80 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32} {

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
  // AMD-DAG: [[C5_i8:%.*]] = llvm.mlir.constant(5 : i8)
  // AMD-DAG: [[C6_i8:%.*]] = llvm.mlir.constant(6 : i8)

  // CHECK: llvm.switch [[COMMAND:%[0-9]+]] : i8
  // CHECK-NEXT: 0: [[SPLIT:\^bb[0-9]+]],
  // CHECK-NEXT: 1: [[ACQUIRE:\^bb[0-9]+]],
  // CHECK-NEXT: 2: [[ACQUIRE]],
  // CHECK-NEXT: 3: [[SPLIT]]
  // CHECK-NEXT: ]
  // CHECK: [[SPLIT]]:
  // CHECK: llvm.switch {{.*}} : i32, [[WS0_PARTITION0:\^bb[0-9]+]] [
  // CHECK-NEXT: 4: [[WS0_PARTITION1:\^bb[0-9]+]],
  // CHECK-NEXT: 5: [[WS0_PARTITION1]],
  // CHECK-NEXT: 6: [[WS0_PARTITION2:\^bb[0-9]+]],
  // CHECK-NEXT: 7: [[WS0_PARTITION2]]
  // CHECK-NEXT: 24: [[WS3_PARTITION0:\^bb[0-9]+]]

  // CHECK: [[WS0_PARTITION0]]:
  // CHECK: "ws0_partition0"
  // CHECK: [[WS0_PARTITION1]]:
  // CHECK: "ws0_partition1"
  // CHECK: [[WS0_PARTITION2]]:
  // CHECK: "ws0_partition2"
  // The complete groups acquire their common cap before dispatching to the bodies.
  // CHECK: [[ACQUIRE]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 80
  // CHECK-NEXT: llvm.switch [[COMMAND]] : i8, [[WS1_PARTITION1:\^bb[0-9]+]] [
  // CHECK-NEXT: 2: [[WS1_PARTITION0:\^bb[0-9]+]]
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
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK: "ws0_default"

  // AMD: llvm.switch
  // AMD-NEXT: 0: [[WS0_PARTITION0:\^bb[0-9]+]],
  // AMD-NEXT: 1: [[WS0_PARTITION1:\^bb[0-9]+]],
  // AMD-NEXT: 2: [[WS0_PARTITION2:\^bb[0-9]+]],
  // AMD-NEXT: 3: [[WS1_PARTITION0:\^bb[0-9]+]],
  // AMD-NEXT: 4: [[WS1_PARTITION1:\^bb[0-9]+]],
  // AMD-NEXT: 5: [[WS3_PARTITION0:\^bb[0-9]+]]
  // AMD-NEXT: ]

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
  // AMD-NEXT: llvm.store [[C2_i8]], [[PTR]]
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
  partition2() num_warps(2) {
    "ws0_partition2"() : () -> ()
    ttg.warp_return
  } : () -> ()

  // CHECK: getelementptr
  // CHECK-NEXT: llvm.store [[C1_i8]], [[SMEM_BASE:%[0-9]+]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[C2_i8]], [[PTR]]
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
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

  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 8, 4>, actualRegisters = array<i32: 80, 80, 80>}
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

  // CHECK: "ws2_default"
  // AMD: "ws2_default"
  "ws2_default"() : () -> ()

  // CHECK: getelementptr
  // CHECK-NEXT: llvm.store [[C3_i8]], [[SMEM_BASE:%[0-9]+]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[C3_i8]], [[PTR]]
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
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
  // CHECK: llvm.store [[C4_i8]],
  // AMD: llvm.store [[C6_i8]],
  // COMMON: llvm.return
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @cfg
llvm.func @cfg() attributes {allocation.offset = 32 : i32} {
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)

  // COMMON: llvm.cond_br {{.*}}, [[DEFAULT:\^bb[0-9]+]], {{\^bb[0-9]+}}

  // CHECK: nvvm.barrier id = [[C1]] aligned = false
  // CHECK: "something"()[[[A:\^.*]], [[B:\^.*]]]
  // CHECK: [[A]]:
  // CHECK-NEXT: "A"
  // CHECK-NEXT: llvm.br [[JOIN:\^.*]]
  // CHECK: [[B]]:
  // CHECK-NEXT: "B"
  // CHECK-NEXT: llvm.br [[JOIN]]

  // CHECK: [[DEFAULT]]:
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: "something"()[[[A:\^.*]], [[B:\^.*]]]
  // CHECK: [[A]]:
  // CHECK-NEXT: "A"
  // CHECK-NEXT: llvm.br [[AFTER:\^.*]]
  // CHECK: [[B]]:
  // CHECK-NEXT: "B"
  // CHECK-NEXT: llvm.br [[AFTER]]
  // CHECK: [[AFTER]]:
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: llvm.return

  // AMD: rocdl.barrier
  // AMD: "something"()[[[A:\^bb[0-9]+]], [[B:\^bb[0-9]+]]]
  // AMD: [[A]]:
  // AMD-NEXT: "A"
  // AMD-NEXT: llvm.br [[JOIN:\^bb[0-9]+]]
  // AMD: [[B]]:
  // AMD-NEXT: "B"
  // AMD-NEXT: llvm.br [[JOIN]]

  // AMD: [[DEFAULT]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: "something"()[[[A:\^bb[0-9]+]], [[B:\^bb[0-9]+]]]
  // AMD: [[A]]:
  // AMD-NEXT: "A"
  // AMD-NEXT: llvm.br [[AFTER:\^bb[0-9]+]]
  // AMD: [[B]]:
  // AMD-NEXT: "B"
  // AMD-NEXT: llvm.br [[AFTER]]
  // AMD: [[AFTER]]:
  // AMD-NEXT: rocdl.barrier
  // AMD-NEXT: llvm.return

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
// COMMON-NOT: llvm.load
// COMMON-NOT: llvm.store
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
  // COMMON: "use"([[OUT]])
  "use"(%2) : (!llvm.ptr<1>) -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// COMMON-LABEL: @capture_function_arg
llvm.func @capture_function_arg(%arg0: i32) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%arg0) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg1: i32) num_warps(4) {
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
  partition0(%arg1: !tt.ptr<i32>) num_warps(4) {
    %1 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<i32> to !llvm.ptr<1>
    // COMMON: "use"(%arg0)
    "use"(%1) : (!llvm.ptr<1>) -> ()
    ttg.warp_return
  } : (!tt.ptr<i32>) -> ()
  llvm.return
}

// COMMON-LABEL: @trivial_remat
llvm.func @trivial_remat() attributes {allocation.offset = 0 : i32} {
  // COMMON-DAG: [[CAP0:%.*]] = llvm.mlir.constant(0 : i32)
  // COMMON-DAG: [[CAP1:%.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>

  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  ttg.warp_specialize(%0, %1) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: i32, %arg1: !llvm.ptr<3>) num_warps(4) {
  // CHECK: ^bb4:
    // CHECK-NEXT: "use"([[CAP0]], [[CAP1]])
  // AMD: ^bb4:
    // AMD-NEXT: "use"([[CAP0]], [[CAP1]])
    "use"(%arg0, %arg1) : (i32, !llvm.ptr<3>) -> ()
    // COMMON-NEXT: llvm.br
    ttg.warp_return
  } : (i32, !llvm.ptr<3>) -> ()
  llvm.return
}

// COMMON-LABEL: @remat_subgraph
llvm.func @remat_subgraph(%arg0: i32, %arg1: i32) attributes {allocation.offset = 0 : i32} {
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
  partition0(%arg2: !llvm.ptr<3>, %arg3: i32) num_warps(4) {
  // CHECK: ^bb4:
    // CHECK-NEXT: [[ADD:%.*]] = llvm.add %arg0, %arg1 : i32
    // CHECK-NEXT: [[MUL:%.*]] = llvm.mul [[ADD]], %arg1 : i32
    // CHECK-NEXT: [[UREM:%.*]] = llvm.urem [[ADD]], [[MUL]] : i32
    // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[ADDR]][%arg0]
    // CHECK-NEXT: "use"([[PTR]], [[UREM]])
  // AMD: ^bb4:
    // AMD-NEXT: [[ADD:%.*]] = llvm.add %arg0, %arg1 : i32
    // AMD-NEXT: [[MUL:%.*]] = llvm.mul [[ADD]], %arg1 : i32
    // AMD-NEXT: [[UREM:%.*]] = llvm.urem [[ADD]], [[MUL]] : i32
    // AMD-NEXT: [[PTR:%.*]] = llvm.getelementptr [[ADDR]][%arg0]
    // AMD-NEXT: "use"([[PTR]], [[UREM]])
    "use"(%arg2, %arg3) : (!llvm.ptr<3>, i32) -> ()
    // COMMON-NEXT: llvm.br
    ttg.warp_return
  } : (!llvm.ptr<3>, i32) -> ()
  llvm.return
}

}

// -----

module attributes {ttg.maxnreg = 80 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.total-num-warps" = 16 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @dynamic_register_reallocation
llvm.func @dynamic_register_reallocation(%run: i1) attributes {allocation.offset = 0 : i32} {
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)

  // CHECK: cond_br %{{.*}}, [[ENTRY:\^.*]], [[WORKERS:\^.*]]

  // CHECK: [[WORKERS]]:
  // CHECK-NEXT: nvvm.setmaxregister decrease 24
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP:\^bb[0-9]+]]
  // CHECK: [[SWITCH_LOOP]]:
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK: llvm.switch {{.*}} : i8, [[INVALID:\^bb[0-9]+]] [
  // CHECK-NEXT: 0: [[PARTITION0:\^.*]],
  // CHECK-NEXT: 1: [[PARTITION1:\^.*]],
  // CHECK-NEXT: 2: [[PARTITION2:\^.*]],
  // CHECK-NEXT: 3: [[EXIT:\^bb[0-9]+]]
  // CHECK-NEXT: ]
  // CHECK: [[INVALID]]:
  // CHECK-NEXT: llvm.unreachable
  // CHECK: [[EXIT]]:
  // CHECK-NEXT: llvm.return

  // CHECK: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: nvvm.setmaxregister decrease 24
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]
  // CHECK-COUNT-2: nvvm.setmaxregister decrease 24

  // CHECK: [[PARTITION0]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 80
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: "partition0"()

  // CHECK: [[PARTITION1]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 48
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: "partition1"()

  // CHECK: [[PARTITION2]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 128
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: "partition2"()

  // CHECK: [[ENTRY]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 248

  // CHECK: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: setmaxregister decrease 64
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK: "default"
  // CHECK: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: setmaxregister increase 248

  llvm.cond_br %run, ^launch, ^exit
^launch:
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 12>, actualRegisters = array<i32: 64, 80, 48, 128>}
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
  llvm.br ^exit
^exit:
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
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: llvm.switch [[WID:%.*]] : i32, [[LOW:\^bb[0-9]+]] [
  // CHECK-NEXT: 8: [[HIGH:\^bb[0-9]+]]
  // CHECK: 15: [[HIGH]]

  // CHECK: [[LOW]]:
  // CHECK-NEXT: nvvm.setmaxregister decrease 24
  // CHECK-NEXT: llvm.br [[PARTITION0:\^bb[0-9]+]]
  // CHECK: [[LOW_DONE:\^bb[0-9]+]]:
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: nvvm.setmaxregister increase 80

  // The split quartet and the adjacent quartet use the same high cap.
  // CHECK: [[HIGH]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 192
  // CHECK-NEXT: llvm.switch [[WID]] : i32, [[PARTITION1:\^bb[0-9]+]] [
  // CHECK-NEXT: 12: [[PARTITION2:\^bb[0-9]+]],
  // CHECK-NEXT: 13: [[PARTITION2]],
  // CHECK-NEXT: 14: [[PARTITION3:\^bb[0-9]+]],
  // CHECK-NEXT: 15: [[PARTITION3]]
  // CHECK-NEXT: ]
  // CHECK: [[HIGH_DONE:\^bb[0-9]+]]:
  // CHECK-NEXT: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: nvvm.setmaxregister decrease 80

  // CHECK: [[PARTITION0]]:
  // CHECK-NEXT: "partition0"()
  // CHECK-NEXT: llvm.br [[LOW_DONE]]
  // CHECK: [[PARTITION1]]:
  // CHECK-NEXT: "partition1"()
  // CHECK-NEXT: llvm.br [[HIGH_DONE]]
  // CHECK: [[PARTITION2]]:
  // CHECK-NEXT: "partition2"()
  // CHECK-NEXT: llvm.br [[HIGH_DONE]]
  // CHECK: [[PARTITION3]]:
  // CHECK-NEXT: "partition3"()
  // CHECK-NEXT: llvm.br [[HIGH_DONE]]

  // CHECK: [[ENTRY]]:
  // CHECK-NEXT: nvvm.setmaxregister increase 256

  // CHECK: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: setmaxregister decrease 104
  // CHECK: "default"
  // CHECK: nvvm.barrier id = [[C1]] aligned = false
  // CHECK-NEXT: setmaxregister increase 256

  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 12, 14>, actualRegisters = array<i32: 104, 24, 192, 192, 192>}
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
  partition2() num_warps(2) {
    "partition2"() : () -> ()
    ttg.warp_return
  }
  partition3() num_warps(2) {
    "partition3"() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {ttg.maxnreg = 80 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @gsan_dynamic_register_reallocation_disabled
llvm.func @gsan_dynamic_register_reallocation_disabled() attributes {allocation.offset = 0 : i32} {
  // CHECK-NOT: nvvm.setmaxregister
  // CHECK: llvm.return
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>, actualRegisters = array<i32: 152, 80>, "tti.disable_setmaxregister"}
  default {
    "default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "partition0"() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}
