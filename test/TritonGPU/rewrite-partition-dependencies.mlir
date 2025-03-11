// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-rewrite-partition-dependencies -verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @invalid_attribute(%lb: i32, %ub: i32, %step: i32) {
  // expected-warning @below {{partition stages attribute 'ttg.partition.stages' has invalid element "a"}}
  scf.for %i = %lb to %ub step %step : i32 {
    scf.yield
  } {ttg.partition.stages = ["a"]}
  scf.for %j = %lb to %ub step %step : i32 {
    scf.yield
  }
  scf.for %k = %lb to %ub step %step : i32 {
    // expected-warning @below {{invalid partition index -1}}
    "op"() {ttg.partition = -1} : () -> ()
    scf.yield
  } {ttg.partition.stages = [2, 2]}
  tt.return
}

tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
  // expected-warning @below {{warp schedule contains a cycle}}
  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> index
    // expected-note @below {{operation in partition #1 uses value defined in partition #0}}
    %1 = "op_b"(%0) {ttg.partition = 1} : (index) -> index
    // expected-note @below {{operation in partition #0 uses value defined in partition #1}}
    "op_c"(%1) {ttg.partition = 0} : (index) -> ()
    scf.yield
  } {ttg.partition.stages = [0, 2]}
  // expected-warning @below {{warp schedule contains a cycle}}
  scf.for %j = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> index
    // expected-note @below {{operation in partition #1 uses value defined in partition #0}}
    %1 = "op_b"(%0) {ttg.partition = 1} : (index) -> index
    // expected-note @below {{operation in partition #2 uses value defined in partition #1}}
    %2 = "op_c"(%1) {ttg.partition = 2} : (index) -> index
    // expected-note @below {{operation in partition #0 uses value defined in partition #2}}
    "op_c"(%2) {ttg.partition = 0} : (index) -> ()
    scf.yield
  } {ttg.partition.stages = [0, 2, 3]}
  tt.return
}

tt.func @invalid_root_partition(%lb: i32, %ub: i32, %step: i32) {
  scf.for %i = %lb to %ub step %step : i32 {
    // expected-note @below {{operand defined here in partition #0 at distance 0}}
    %0 = "partition"() {ttg.partition = 0} : () -> index
    // expected-warning @below {{operation in the root partition depends on a value that originates from a non-root partition through operand #0}}
    "root"(%0) : (index) -> ()
    scf.yield
  } {ttg.partition.stages = [0, 2]}

  %c0 = arith.constant 0 : index
  scf.for %j = %lb to %ub step %step iter_args(%k = %c0) -> index : i32 {
    // expected-warning @below {{operation in the root partition depends on a value that originates from a non-root partition through operand #0}}
    "root"(%k) : (index) -> ()
    // expected-note @below {{operand defined here in partition #0 at distance 1}}
    %0 = "partition"() {ttg.partition = 0} : () -> index
    scf.yield %0 : index
  } {ttg.partition.stages = [0, 2]}
  tt.return
}

tt.func @invalid_partition_stage(%lb: i32, %ub: i32, %step: i32) {
  // expected-warning @below {{partition #0 has stage 2 but is consumed by partition #1 with stage 0}}
  scf.for %i = %lb to %ub step %step : i32 {
    // expected-note @below {{value defined here in partition #0}}
    %0 = "op_a"() {ttg.partition = 0} : () -> index
    // expected-note @below {{use of value defined in partition #0}}
    "op_b"(%0) {ttg.partition = 1} : (index) -> ()
  } {ttg.partition.stages = [2, 0]}
  tt.return
}

tt.func @invalid_future_partition(%lb: i32, %ub: i32, %step: i32) {
  %c0 = arith.constant 0 : index
  // expected-warning @below {{partition #1 has stage 2 but is consumed by partition #0 with stage 0 at distance 1}}
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0) -> index : i32 {
    // expected-note @below {{use of value defined in partition #1 at 1 iterations in the future}}
    "op_a"(%k) {ttg.partition = 0} : (index) -> ()
    // expected-note @below {{value defined here in partition #1}}
    %0 = "op_b"() {ttg.partition = 1} : () -> index
    scf.yield %0 : index
  } {ttg.partition.stages = [0, 2]}
  tt.return
}

tt.func @two_consumers(%lb: i32, %ub: i32, %step: i32) {
  %c0 = arith.constant dense<0> : !ty
  scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    "op_b"(%0) {ttg.partition = 1} : (!ty) -> ()
    "op_d"(%0) {ttg.partition = 2} : (!ty) -> ()
  } {ttg.partition.stages = [0, 2, 2]}
  tt.return
}

tt.func @distance_one(%lb: i32, %ub: i32, %step: i32) {
  %c0 = arith.constant dense<0> : !ty
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0) -> (!ty) : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    "op_b"(%k) {ttg.partition = 1} : (!ty) -> ()
    scf.yield %0 : !ty
  } {ttg.partition.stages = [0, 0]}
  tt.return
}

tt.func @complex_case(%lb: i32, %ub: i32, %step: i32) {
  %c0 = arith.constant dense<0> : !ty
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0, %l = %c0) -> (!ty, !ty) : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    "op_b"(%k) {ttg.partition = 1} : (!ty) -> ()
    "op_c"(%k) {ttg.partition = 2} : (!ty) -> ()
    "op_c"(%k) {ttg.partition = 2} : (!ty) -> ()
    "op_d"(%l) {ttg.partition = 1} : (!ty) -> ()
    "op_d"(%l) {ttg.partition = 2} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {ttg.partition.stages = [0, 2, 2]}
  tt.return
}

tt.func @reuse_argument(%lb: i32, %ub: i32, %step: i32) {
  %c0 = arith.constant dense<0> : !ty
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0, %l = %c0) -> (!ty, !ty) : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    "op_d"(%l) {ttg.partition = 1} : (!ty) -> ()
    "op_d"(%l) {ttg.partition = 2} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {ttg.partition.stages = [1, 0, 0]}
  tt.return
}

}
