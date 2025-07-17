// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-rewrite-partition-dependencies -verify-diagnostics -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @two_consumers
tt.func @two_consumers(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[C0:%.*]] = arith.constant 0 : i32
  // CHECK-NEXT: [[ABUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF:%.*]] = nvws.aref.create [[ABUF]]
  scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // CHECK: [[VAL:%.*]] = "op_a"
    // CHECK-NEXT: [[BUF:%.*]] = nvws.aref.put.enter [[AREF]][[[C0]]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: ttg.local_store [[VAL]], [[BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF]][[[C0]]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}

    "op_b"(%0) {ttg.partition = 1} : (!ty) -> ()
    // CHECK-NEXT: [[BUF:%.*]] = nvws.aref.get.enter [[AREF]][[[C0]]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]][[[C0]]] [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"([[VAL]])

    "op_c"(%0) {ttg.partition = 2} : (!ty) -> ()
    // CHECK-NEXT: [[BUF:%.*]] = nvws.aref.get.enter [[AREF]][[[C0]]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]][[[C0]]] [#nvws.async_op<none>] {ttg.partition = 2 : i32}
    // CHECK-NEXT: "op_c"([[VAL]])
    // CHECK-NEXT: "op_d"([[VAL]])
    "op_d"(%0) {ttg.partition = 2} : (!ty) -> ()
  } {ttg.partition.stages = [0, 2, 2]}
  tt.return
}

// CHECK-LABEL: @distance_one
tt.func @distance_one(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[C0:%.*]] = arith.constant 0 : i32
  // CHECK: [[ABUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF:%.*]] = nvws.aref.create [[ABUF]]
  %cst = arith.constant dense<0> : !ty
  // CHECK: scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[K:%.*]] = {{.*}})
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    // CHECK-NEXT: [[BUF:%.*]] = nvws.aref.put.enter [[AREF]][[[C0]]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: ttg.local_store [[K]], [[BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF]][[[C0]]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // CHECK: [[VAL:%.*]] = "op_a"
    // CHECK-NEXT: [[BUF:%.*]] = nvws.aref.get.enter [[AREF]][[[C0]]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]][[[C0]]] [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"([[VAL]])
    "op_b"(%k) {ttg.partition = 1} : (!ty) -> ()

    scf.yield %0 : !ty
  } {ttg.partition.stages = [0, 0]}
  tt.return
}

// CHECK-LABEL: @self_recursion
tt.func @self_recursion(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-NOT: nvws.aref.create
  %cst = arith.constant dense<0> : !ty
  // CHECK: iter_args([[ARG:%arg[0-9]+]] = %cst)
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    // CHECK-NEXT: [[OUT:%.*]] = "op_a"([[ARG]])
    %0 = "op_a"(%k) {ttg.partition = 0} : (!ty) -> !ty
    // CHECK: yield [[OUT]]
    scf.yield %0 : !ty
  } {ttg.partition.stages = [0]}
  tt.return
}

// CHECK-LABEL: @self_recursion_and_use
tt.func @self_recursion_and_use(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : !ty
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    %0 = "op_a"(%k) {ttg.partition = 0} : (!ty) -> !ty
    // CHECK: "op_a"
    // CHECK-NEXT: nvws.aref.put.enter
    // CHECK-NEXT: local_store
    // CHECK-NEXT: nvws.aref.put.exit

    "op_b"(%0) {ttg.partition = 1} : (!ty) -> !ty
    // CHECK-NEXT: nvws.aref.get.enter
    // CHECK-NEXT: ttg.local_load
    // CHECK-NEXT: nvws.aref.get.exit
    // CHECK-NEXT: "op_b"

    scf.yield %0 : !ty
  } {ttg.partition.stages = [0, 1]}
  tt.return
}

// CHECK-LABEL: @conditional_consumer
tt.func @conditional_consumer(%lb: i32, %ub: i32, %step: i32) {
  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "producer"() {ttg.partition = 0} : () -> !ty
    // CHECK: "producer"
    // CHECK-NEXT: nvws.aref.put.enter
    // CHECK-NEXT: local_store
    // CHECK-NEXT: nvws.aref.put.exit
    %cond = "rand"() : () -> i1
    // CHECK-NEXT: "rand"
    // CHECK-NEXT: nvws.aref.get.enter
    // CHECK-NEXT: [[VALUE:%.*]] = ttg.local_load
    // CHECK-NEXT: nvws.aref.get.exit
    // CHECK-NEXT: scf.if
    %1 = scf.if %cond -> !ty {
      // CHECK-NEXT: "something"
      "something"() : () -> ()
      // CHECK-NEXT: yield [[VALUE]]
      scf.yield %0 : !ty
    } else {
      %2 = "something"() : () -> !ty
      scf.yield %2 : !ty
    } {ttg.partition = 1}
    "keep"(%1) {ttg.partition = 1} : (!ty) -> ()
  } {ttg.partition.stages = [0, 2]}
  tt.return
}

// CHECK-LABEL: @no_def_op
tt.func @no_def_op(%lb: i32, %ub: i32, %step: i32) {
  %c0_i32 = arith.constant 0 : i32
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0_i32) -> i32 : i32 {
    arith.addi %k, %k : i32
    scf.yield %k : i32
  }
  tt.return
}

}

// -----

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
  tt.return
}

}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @invalid_attribute(%lb: i32, %ub: i32, %step: i32) {
  scf.for %k = %lb to %ub step %step : i32 {
    // expected-warning @below {{invalid partition index -1}}
    "op"() {ttg.partition = -1} : () -> ()
    scf.yield
  } {ttg.partition.stages = [2, 2]}
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[C0:%.*]] = arith.constant 0 : i32
  // CHECK: ttg.local_alloc
  // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create
  // CHECK-NEXT: ttg.local_alloc
  // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create

  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // CHECK: "op_a"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF1]][[[C0]]] {ttg.partition = 0 : i32}

    %1 = "op_b"(%0) {ttg.partition = 1} : (!ty) -> !ty
    // CHECK: nvws.aref.get.exit [[AREF1]][[[C0]]] [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF2]][[[C0]]] {ttg.partition = 1 : i32}

    // CHECK: nvws.aref.get.exit [[AREF2]][[[C0]]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}

    "op_c"(%1) {ttg.partition = 0} : (!ty) -> ()
    scf.yield
  } {ttg.partition.stages = [0, 2]}
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[C0:%.*]] = arith.constant 0 : i32
  // CHECK: ttg.local_alloc
  // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create
  // CHECK-NEXT: ttg.local_alloc
  // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create
  // CHECK-NEXT: ttg.local_alloc
  // CHECK-NEXT: [[AREF3:%.*]] = nvws.aref.create
  scf.for %j = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // CHECK: "op_a"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF1]][[[C0]]] {ttg.partition = 0 : i32}

    %1 = "op_b"(%0) {ttg.partition = 1} : (!ty) -> !ty
    // CHECK: nvws.aref.get.exit [[AREF1]][[[C0]]] [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF2]][[[C0]]] {ttg.partition = 1 : i32}

    %2 = "op_c"(%1) {ttg.partition = 2} : (!ty) -> !ty
    // CHECK: nvws.aref.get.exit [[AREF2]][[[C0]]] [#nvws.async_op<none>] {ttg.partition = 2 : i32}
    // CHECK-NEXT: "op_c"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF3]][[[C0]]] {ttg.partition = 2 : i32}

    "op_c"(%2) {ttg.partition = 0} : (!ty) -> ()
    // CHECK: nvws.aref.get.exit [[AREF3]][[[C0]]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}
    // CHECK: "op_c"
    scf.yield
  } {ttg.partition.stages = [0, 2, 3]}
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @invalid_root_partition(%lb: i32, %ub: i32, %step: i32) {
  scf.for %i = %lb to %ub step %step : i32 {
    // expected-note @below {{operand defined here in partition #0 at distance 0}}
    %0 = "partition"() {ttg.partition = 0} : () -> index
    // expected-warning @below {{operation in the root partition depends on a value that originates from a non-root partition through operand #0}}
    "root"(%0) : (index) -> ()
    scf.yield
  } {ttg.partition.stages = [0, 2]}
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @invalid_root_partition(%lb: i32, %ub: i32, %step: i32) {
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

}
