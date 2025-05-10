// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-partition-loops -verify-diagnostics -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @no_partitions
tt.func @no_partitions(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step : i32 {
    // CHECK-NEXT: op_a
    "op_a"() : () -> ()
  } {ttg.partition.stages = []}
  tt.return
}

// CHECK-LABEL: @one_partition
tt.func @one_partition(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step : i32 {
    // CHECK-NEXT: op_a
    "op_a"() {ttg.partition = 0} : () -> ()
  } {ttg.partition.stages = [0]}
  tt.return
}

// CHECK-LABEL: @two_empty_partitions
tt.func @two_empty_partitions(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-NEXT: ttg.warp_specialize(%arg0, %arg1, %arg2)
  // CHECK-NEXT: default
  // CHECK-NEXT:   scf.for [[I:%.*]] = %arg0 to %arg1 step %arg2
  // CHECK-NEXT:     "op_a"([[I]])
  // CHECK-NEXT:   }
  // CHECK-NEXT:   warp_yield
  // CHECK-NEXT: }
  // CHECK-NEXT: partition0(%arg3: i32, %arg4: i32, %arg5: i32) num_warps(4)
  // CHECK-NEXT:   scf.for [[I:%.*]] = %arg3 to %arg4 step %arg5
  // CHECK-NEXT:     "op_a"([[I]])
  // CHECK-NEXT:   }
  // CHECK-NEXT:   warp_return
  scf.for %i = %lb to %ub step %step : i32 {
    "op_a"(%i) : (i32) -> ()
  } {ttg.partition.stages = [0, 0]}
  tt.return
}

// CHECK-LABEL: @empty_partition_fwd_root
tt.func @empty_partition_fwd_root(%lb: i32, %ub: i32, %step: i32) {
  %c0_i32 = arith.constant 0 : i32
  // CHECK: partition0
  // CHECK-NEXT: [[C0:%.*]] = arith.constant 0
  // CHECK-NEXT: scf.for [[I:%.*]] = {{.*}} iter_args([[K:%.*]] = [[C0]])
  // CHECK-NEXT:   "op_a"([[I]], [[K]])
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0_i32) -> i32 : i32 {
    %0 = "op_a"(%i, %k) : (i32, i32) -> i32
    scf.yield %0 : i32
  } {ttg.partition.stages = [0, 0]}
  tt.return
}

// CHECK-LABEL: @multiple_partitions
tt.func @multiple_partitions(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: default
  // CHECK-NEXT: scf.for
  // CHECK-NEXT:   [[X:%.*]] = "op_a"
  // CHECK-NEXT:   "op_b"([[X]])
  // CHECK-NEXT:   "op_b"([[X]])
  // CHECK-NEXT: }

  // CHECK: partition0
  // CHECK-NEXT: scf.for [[I:%arg[0-9]+]]
  // CHECK-NEXT:   [[Y:%.*]] = arith.addi [[I]], [[I]]
  // CHECK-NEXT:   [[X:%.*]] = "op_a"([[Y]])
  // CHECK-NEXT:   "op_b"([[X]])
  // CHECK-NEXT:   "op_b"([[X]])
  // CHECK-NEXT: }

  // CHECK: partition1
  // CHECK-NEXT: scf.for [[I:%arg[0-9]+]]
  // CHECK-NEXT:   [[Y:%.*]] = arith.addi [[I]], [[I]]
  // CHECK-NEXT:   [[Z:%.*]] = arith.addi [[I]], [[Y]]
  // CHECK-NEXT:   [[X:%.*]] = "op_a"([[Z]])
  // CHECK-NEXT:   "op_b"([[X]])
  // CHECK-NEXT:   "op_b"([[X]])
  // CHECK-NEXT: }

  scf.for %i = %lb to %ub step %step : i32 {
    %a = arith.addi %i, %i : i32
    %b = arith.addi %i, %a : i32

    %0 = "op_a"(%i) {ttg.partition = 0} : (i32) -> i32
    "op_b"(%0) {ttg.partition = 0} : (i32) -> ()
    "op_b"(%0) {ttg.partition = 0} : (i32) -> ()

    %1 = "op_a"(%a) {ttg.partition = 1} : (i32) -> i32
    "op_b"(%1) {ttg.partition = 1} : (i32) -> ()
    "op_b"(%1) {ttg.partition = 1} : (i32) -> ()

    %2 = "op_a"(%b) {ttg.partition = 2} : (i32) -> i32
    "op_b"(%2) {ttg.partition = 2} : (i32) -> ()
    "op_b"(%2) {ttg.partition = 2} : (i32) -> ()
  } {ttg.partition.stages = [0, 0, 0]}
  tt.return
}

// CHECK-LABEL: @split_block_arguments
tt.func @split_block_arguments(%lb: i32, %ub: i32, %step: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK:      default
  // CHECK-NEXT:   scf.for {{.*}} iter_args([[A:%.*]] = %c0_i32)
  // CHECK-NEXT:     [[X:%.*]] = "op_a"([[A]])
  // CHECK-NEXT:     yield [[X]] : i32

  // CHECK:      partition0
  // CHECK-NEXT:   %c1_i32 = arith.constant 1
  // CHECK-NEXT:   scf.for {{.*}} iter_args([[B:%.*]] = %c1_i32)
  // CHECK-NEXT:     [[X:%.*]] = "op_b"([[B]])
  // CHECK-NEXT:     yield [[X]] : i32
  scf.for %i = %lb to %ub step %step iter_args(%a = %c0_i32, %b = %c1_i32) -> (i32, i32) : i32 {
    %0 = "op_a"(%a) {ttg.partition = 0} : (i32) -> i32
    %1 = "op_b"(%b) {ttg.partition = 1} : (i32) -> i32
    scf.yield %0, %1 : i32, i32
  } {ttg.partition.stages = [0, 0]}
  tt.return
}

// CHECK-LABEL: @partition_outputs
tt.func @partition_outputs(%lb: i32, %ub: i32, %step: i32) -> (!ty, !ty, !ty) {
  // CHECK-NEXT: [[CST0:%.*]] = arith.constant dense<0>
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty
  %cst2 = arith.constant dense<2> : !ty

  // CHECK-NEXT: [[B_BUF:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[C_BUF:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[A_OUT:%.*]] = ttg.warp_specialize(%arg0, %arg1, %arg2, [[B_BUF]], [[C_BUF]])

  // CHECK-NEXT: default
  // CHECK-NEXT: [[OUT:%.*]] = scf.for [[I:%arg[0-9]+]] {{.*}} iter_args([[A:%.*]] = [[CST0]])
  // CHECK-NEXT:   [[X:%.*]] = "op_a"([[I]], [[A]])
  // CHECK-NEXT:   yield [[X]]
  // CHECK-NEXT: }
  // CHECK-NEXT: warp_yield [[OUT]]

  // CHECK:      partition0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: !ttg.memdesc{{.*}}, %arg7: !ttg.memdesc{{.*}}) num_warps(4)
  // CHECK-NEXT: [[CST1:%.*]] = arith.constant dense<1>
  // CHECK-NEXT: [[OUT:%.*]] = scf.for [[I:%arg[0-9]+]] {{.*}} iter_args([[B:%.*]] = [[CST1]])
  // CHECK-NEXT:   [[X:%.*]] = "op_b"([[I]], [[B]])
  // CHECK-NEXT:   yield [[X]]
  // CHECK-NEXT: }
  // CHECK-NEXT: local_store [[OUT]], %arg6

  // CHECK:      partition1(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: !ttg.memdesc{{.*}}, %arg7: !ttg.memdesc{{.*}}) num_warps(4)
  // CHECK-NEXT: [[CST2:%.*]] = arith.constant dense<2>
  // CHECK-NEXT: [[OUT:%.*]] = scf.for [[I:%arg[0-9]+]] {{.*}} iter_args([[C:%.*]] = [[CST2]])
  // CHECK-NEXT:   [[X:%.*]] = "op_c"([[I]], [[C]])
  // CHECK-NEXT:   yield [[X]]
  // CHECK-NEXT: }
  // CHECK-NEXT: local_store [[OUT]], %arg7

  %outs:3 = scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
    %0 = "op_a"(%i, %a) {ttg.partition = 0} : (i32, !ty) -> !ty
    %1 = "op_b"(%i, %b) {ttg.partition = 1} : (i32, !ty) -> !ty
    %2 = "op_c"(%i, %c) {ttg.partition = 2} : (i32, !ty) -> !ty
    scf.yield %0, %1, %2 : !ty, !ty, !ty
  } {ttg.partition.stages = [0, 0, 0]}

  // CHECK:      [[C_OUT:%.*]] = ttg.local_load [[C_BUF]]
  // CHECK-NEXT: local_dealloc [[C_BUF]]
  // CHECK-NEXT: [[B_OUT:%.*]] = ttg.local_load [[B_BUF]]
  // CHECK-NEXT: local_dealloc [[B_BUF]]

  // CHECK-NEXT: tt.return [[A_OUT]], [[B_OUT]], [[C_OUT]]
  tt.return %outs#0, %outs#1, %outs#2 : !ty, !ty, !ty
}

// CHECK-LABEL: @future_conditional_self_use
tt.func @future_conditional_self_use(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0_i32) -> i32 : i32 {
    %0 = "op_a"() {ttg.partition = 0 : i32} : () -> i32
    scf.if %cond {
      "use"(%k) : (i32) -> ()
    } {ttg.partition = 0 : i32}
    scf.yield %0 : i32
  } {ttg.partition.stages = [0]}
  tt.return
}

// CHECK-LABEL: @trivial_tensor_captures
tt.func @trivial_tensor_captures(%arg0: f16, %lb: i32, %ub: i32, %step: i32) {
  %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %1 = tt.splat %arg0 : f16 -> tensor<32xf16>
  // CHECK: ttg.warp_specialize(%arg1, %arg2, %arg3, %arg0)
  scf.for %i = %lb to %ub step %step : i32 {
    // CHECK: partition0(%arg4: i32, %arg5: i32, %arg6: i32, %arg7: f16) num_warps(4)
    // CHECK-NEXT: [[RANGE:%.*]] = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    // CHECK-NEXT: [[SPLAT:%.*]] = tt.splat %arg7 : f16 -> tensor<32xf16>
    // CHECK-NEXT: scf.for
    // CHECK-NEXT: "use"([[RANGE]], [[SPLAT]])
    "use"(%0, %1) {ttg.partition = 1} : (tensor<256xi32>, tensor<32xf16>) -> ()
  } {ttg.partition.stages = [0, 0]}
  tt.return
}

// CHECK-LABEL: @dce_before_warp_allocation
tt.func @dce_before_warp_allocation(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : tensor<128xi32, #blocked>
  // CHECK: ttg.warp_specialize
  // CHECK: partition0({{.*}}) num_warps(4)
  // CHECK: partition1({{.*}}) num_warps(4)
  scf.for %i = %lb to %ub step %step iter_args(%idxs = %cst) -> tensor<128xi32, #blocked> : i32 {
    %do_prologue = "prologue_cond"(%i) : (i32) -> i1
    %0 = scf.if %do_prologue -> tensor<128xi32, #blocked> {
      %1 = tt.splat %i : i32 -> tensor<128xi32, #blocked>
      %2 = arith.addi %1, %idxs : tensor<128xi32, #blocked>
      scf.yield %2 : tensor<128xi32, #blocked>
    } else {
      scf.yield %idxs : tensor<128xi32, #blocked>
    }
    "op_a"(%0) {ttg.partition = 0 : i32} : (tensor<128xi32, #blocked>) -> ()
    "op_b"(%i) {ttg.partition = 1 : i32} : (i32) -> ()
    "op_c"(%0) {ttg.partition = 2 : i32} : (tensor<128xi32, #blocked>) -> ()
    scf.yield %0 : tensor<128xi32, #blocked>
  } {ttg.partition.stages = [0, 0, 0]}
  tt.return
}

// CHECK-LABEL: @capture_order
tt.func public @capture_order(%arg0: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked>
  %1 = arith.extsi %0 : tensor<4xi32, #blocked> to tensor<4xi64, #blocked>
  // CHECK: ttg.warp_specialize
  // CHECK: partition0
  // CHECK: [[VALUE:%.*]] = tt.make_range
  // CHECK-NEXT: [[EXT:%.*]] = arith.extsi [[VALUE]]
  // CHECK-NEXT: scf.for
  scf.for %arg1 = %c0_i32 to %arg0 step %c1_i32  : i32 {
    // CHECK-NEXT: "use"([[VALUE]])
    "use"(%0) : (tensor<4xi32, #blocked>) -> ()
    // CHECK-NEXT: "use"([[EXT]])
    "use"(%1) : (tensor<4xi64, #blocked>) -> ()
  } {ttg.partition.stages = [1 : i32, 0 : i32]}
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @still_has_ssa_deps(%lb: i32, %ub: i32, %step: i32) {
  scf.for %i = %lb to %ub step %step : i32 {
    // expected-warning @below {{non-root partition #0 has direct SSA consumer}}
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // expected-note @below {{use at distance 0 in partition #1 here}}
    "op_b"(%0) {ttg.partition = 1} : (!ty) -> ()
  } {ttg.partition.stages = [0, 1]}
  tt.return
}

}
