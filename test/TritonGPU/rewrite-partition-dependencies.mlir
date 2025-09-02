// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-rewrite-partition-dependencies -verify-diagnostics -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @two_consumers
tt.func @two_consumers(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[C0:%.*]] = arith.constant 0 : i32
  // CHECK-NEXT: [[ABUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF:%.*]] = nvws.aref.create [[ABUF]]
  scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // CHECK: [[VAL:%.*]] = "op_a"
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.put.enter [[AREF]][[[C0]], [[C0]]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: ttg.local_store [[VAL]], [[BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF]][[[C0]]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}

    "op_b"(%0) {ttg.partition = 1} : (!ty) -> ()
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF]][[[C0]], [[C0]]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]][[[C0]]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"([[VAL]])

    "op_c"(%0) {ttg.partition = 2} : (!ty) -> ()
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF]][[[C0]], [[C0]]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]][[[C0]]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = 2 : i32}
    // CHECK-NEXT: "op_c"([[VAL]])
    // CHECK-NEXT: "op_d"([[VAL]])
    "op_d"(%0) {ttg.partition = 2} : (!ty) -> ()
  } {ttg.partition.stages = [0, 2, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @distance_one
tt.func @distance_one(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[C0:%.*]] = arith.constant 0 : i32
  // CHECK: [[ABUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF:%.*]] = nvws.aref.create [[ABUF]]
  %cst = arith.constant dense<0> : !ty
  // CHECK: scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[K:%.*]] = {{.*}})
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.put.enter [[AREF]][[[C0]], [[C0]]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: ttg.local_store [[K]], [[BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF]][[[C0]]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // CHECK: [[VAL:%.*]] = "op_a"
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF]][[[C0]], [[C0]]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]][[[C0]]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"([[VAL]])
    "op_b"(%k) {ttg.partition = 1} : (!ty) -> ()

    scf.yield %0 : !ty
  } {ttg.partition.stages = [0, 0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

tt.func @complex_case(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[ABUF1:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create [[ABUF1]]
  // CHECK-NEXT: [[ABUF2:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create [[ABUF2]]
  %cst = arith.constant dense<0> : !ty
  // CHECK: scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[K:%.*]] = {{.*}}, [[L:%.*]] = {{.*}})
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst, %l = %cst) -> (!ty, !ty) : i32 {
    // CHECK: [[BUF:%.*]], [[TOKEN2:%.*]] = nvws.aref.put.enter [[AREF2]][[[C0]], [[C0]]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: ttg.local_store [[L]], [[BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF2]][[[C0]]], [[TOKEN2]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN1:%.*]] = nvws.aref.put.enter [[AREF1]][[[C0]], [[C0]]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: ttg.local_store [[K]], [[BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF1]][[[C0]]], [[TOKEN1]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}

    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // CHECK-NEXT: op_a
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF1]][[[C0]], [[C0]]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[K1:%.*]] = ttg.local_load [[BUF]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF1]][[[C0]]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"([[K1]])
    "op_b"(%k) {ttg.partition = 1} : (!ty) -> ()


    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF1]][[[C0]], [[C0]]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[K2:%.*]] = ttg.local_load [[BUF]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF1]][[[C0]]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = 2 : i32}
    // CHECK-NEXT: "op_c"([[K2]])
    // CHECK-NEXT: "op_c"([[K2]])
    "op_c"(%k) {ttg.partition = 2} : (!ty) -> ()
    "op_c"(%k) {ttg.partition = 2} : (!ty) -> ()

    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF2]][[[C0]], [[C0]]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[L1:%.*]] = ttg.local_load [[BUF]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF2]][[[C0]]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_d"([[L1]])
    "op_d"(%l) {ttg.partition = 1} : (!ty) -> ()

    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF2]][[[C0]], [[C0]]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[L2:%.*]] = ttg.local_load [[BUF]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF2]][[[C0]]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = 2 : i32}
    // CHECK-NEXT: "op_d"([[L2]])
    "op_d"(%l) {ttg.partition = 2} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {ttg.partition.stages = [0, 2, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @reuse_argument
tt.func @reuse_argument(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-DAG: [[CST0:%.*]] = arith.constant dense<0>
  // CHECK-DAG: [[CST1:%.*]] = arith.constant dense<1>
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty

  // CHECK: local_alloc
  // CHECK-NEXT: [[AREF:%.*]] = nvws.aref.create
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst0, %l = %cst1) -> (!ty, !ty) : i32 {
    // CHECK-NEXT: {{.*}}, [[TOKEN:%.*]] = nvws.aref.put.enter [[AREF]][{{.*}}] {ttg.partition = 0 : i32}
    // CHECK-NEXT: local_store
    // CHECK-NEXT: nvws.aref.put.exit [[AREF]][{{.*}}], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}
    // CHECK-NEXT: op_a
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty

    // CHECK-NEXT: aref.get.enter [[AREF]][{{.*}}] {ttg.partition = 1 : i32}
    // CHECK-NEXT: local_load {{.*}} {ttg.partition = 1 : i32}
    // CHECK-NEXT: aref.get.exit [[AREF]][{{.*}}], {{.*}} [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: op_d
    "op_d"(%l) {ttg.partition = 1} : (!ty) -> ()

    // CHECK-NEXT: aref.get.enter [[AREF]][{{.*}}] {ttg.partition = 2 : i32}
    // CHECK-NEXT: local_load {{.*}} {ttg.partition = 2 : i32}
    // CHECK-NEXT: aref.get.exit [[AREF]][{{.*}}], {{.*}} [#nvws.async_op<none>] {ttg.partition = 2 : i32}
    // CHECK-NEXT: op_d
    "op_d"(%l) {ttg.partition = 2} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {ttg.partition.stages = [1, 0, 0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @multiplicity_branch
tt.func @multiplicity_branch(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-DAG: [[CST0:%.*]] = arith.constant dense<0>
  // CHECK-DAG: [[CST1:%.*]] = arith.constant dense<1>
  // CHECK-DAG: [[CST2:%.*]] = arith.constant dense<2>
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty
  %cst2 = arith.constant dense<2> : !ty

  // CHECK: local_alloc
  // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create
  // CHECK-NEXT: local_alloc
  // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create
  // CHECK-NEXT: local_alloc
  // CHECK-NEXT: [[AREF3:%.*]] = nvws.aref.create

  // CHECK: scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[A:%.*]] = {{.*}}, [[B:%.*]] = {{.*}}, [[C:%.*]] = {{.*}})
  scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN3:%.*]] = nvws.aref.put.enter [[AREF3]][{{.*}}] {ttg.partition = 0 : i32}
    // CHECK-NEXT: local_store [[C]], [[BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF3]][{{.*}}], [[TOKEN3]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN2:%.*]] = nvws.aref.put.enter [[AREF2]][{{.*}}] {ttg.partition = 0 : i32}
    // CHECK-NEXT: local_store [[B]], [[BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF2]][{{.*}}], [[TOKEN2]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN1:%.*]] = nvws.aref.put.enter [[AREF1]][{{.*}}] {ttg.partition = 0 : i32}
    // CHECK-NEXT: local_store [[A]], [[BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF1]][{{.*}}], [[TOKEN1]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}
    // CHECK-NEXT: op_a
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty

    // CHECK: aref.get.enter [[AREF1]]
    // CHECK-NEXT: local_load
    // CHECK-NEXT: aref.get.exit [[AREF1]]
    // CHECK-NEXT: op_b
    "op_b"(%a) {ttg.partition = 1}: (!ty) -> ()

    // CHECK: aref.get.enter [[AREF2]]
    // CHECK-NEXT: local_load
    // CHECK-NEXT: aref.get.exit [[AREF2]]
    // CHECK-NEXT: op_c
    "op_c"(%b) {ttg.partition = 2}: (!ty) -> ()

    // CHECK: aref.get.enter [[AREF3]]
    // CHECK-NEXT: local_load
    // CHECK-NEXT: aref.get.exit [[AREF3]]
    // CHECK-NEXT: op_d
    "op_d"(%c) {ttg.partition = 3}: (!ty) -> ()

    scf.yield %0, %a, %a : !ty, !ty, !ty
  } {ttg.partition.stages = [0, 0, 0, 0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @multiplicity_branch2
tt.func @multiplicity_branch2(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-DAG: [[CST0:%.*]] = arith.constant dense<0>
  // CHECK-DAG: [[CST1:%.*]] = arith.constant dense<1>
  // CHECK-DAG: [[CST2:%.*]] = arith.constant dense<2>
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty
  %cst2 = arith.constant dense<2> : !ty

  // CHECK: local_alloc
  // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create
  // CHECK-NEXT: local_alloc
  // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create
  // CHECK-NEXT: local_alloc
  // CHECK-NEXT: [[AREF3:%.*]] = nvws.aref.create

  // CHECK: scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[A:%.*]] = {{.*}}, [[B:%.*]] = {{.*}}, [[C:%.*]] = {{.*}})
  scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN3:%.*]] = nvws.aref.put.enter [[AREF3]][{{.*}}] {ttg.partition = 2 : i32}
    // CHECK-NEXT: local_store [[C]], [[BUF]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF3]][{{.*}}], [[TOKEN3]] [#nvws.async_op<none>] {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN2:%.*]] = nvws.aref.put.enter [[AREF2]][{{.*}}] {ttg.partition = 1 : i32}
    // CHECK-NEXT: local_store [[B]], [[BUF]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF2]][{{.*}}], [[TOKEN2]] [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN1:%.*]] = nvws.aref.put.enter [[AREF1]][{{.*}}] {ttg.partition = 0 : i32}
    // CHECK-NEXT: local_store [[A]], [[BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF1]][{{.*}}], [[TOKEN1]] [#nvws.async_op<none>] {ttg.partition = 0 : i32}
    // CHECK-NEXT: op_a
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty

    // CHECK: aref.get.enter [[AREF1]][{{.*}}, {{.*}}] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[A1:%.*]] = ttg.local_load {{.*}} {ttg.partition = 1 : i32}
    // CHECK-NEXT: aref.get.exit [[AREF1]]
    // CHECK-NEXT: "op_b"([[A1]]) {ttg.partition = 1 : i32}
    %d = "op_b"(%a) {ttg.partition = 1}: (!ty) -> !ty

    // CHECK: aref.get.enter [[AREF2]][{{.*}}, {{.*}}] {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[B1:%.*]] = ttg.local_load {{.*}} {ttg.partition = 2 : i32}
    // CHECK-NEXT: aref.get.exit [[AREF2]]
    // CHECK-NEXT: "op_c"([[B1]]) {ttg.partition = 2 : i32}
    %e = "op_c"(%b) {ttg.partition = 2}: (!ty) -> !ty

    // CHECK: aref.get.enter [[AREF3]][{{.*}}, {{.*}}] {ttg.partition = 3 : i32}
    // CHECK-NEXT: [[C1:%.*]] = ttg.local_load {{.*}} {ttg.partition = 3 : i32}
    // CHECK-NEXT: aref.get.exit [[AREF3]]
    // CHECK-NEXT: "op_d"([[C1]]) {ttg.partition = 3 : i32}
    "op_d"(%c) {ttg.partition = 3}: (!ty) -> ()

    scf.yield %0, %d, %e : !ty, !ty, !ty
  } {ttg.partition.stages = [0, 0, 0, 0], ttg.warp_specialize.tag = 0 : i32}
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
  } {ttg.partition.stages = [0], ttg.warp_specialize.tag = 0 : i32}
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
  } {ttg.partition.stages = [0, 1], ttg.warp_specialize.tag = 0 : i32}
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
    // CHECK-NEXT: nvws.aref.get.exit{{.*}}, {{.*}}
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
  } {ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
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
  // expected-error @below {{partition stages attribute 'ttg.partition.stages' has invalid element "a"}}
  scf.for %i = %lb to %ub step %step : i32 {
    scf.yield
  } {ttg.partition.stages = ["a"], ttg.warp_specialize.tag = 0 : i32}
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
    // expected-error @below {{invalid partition index -1}}
    "op"() {ttg.partition = -1} : () -> ()
    scf.yield
  } {ttg.partition.stages = [2, 2], ttg.warp_specialize.tag = 0 : i32}
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
    // CHECK-NEXT: nvws.aref.put.enter [[AREF1]][[[C0]], [[C0]]] {ttg.partition = 0 : i32}

    %1 = "op_b"(%0) {ttg.partition = 1} : (!ty) -> !ty
    // CHECK: nvws.aref.get.exit [[AREF1]][[[C0]]], {{.*}} [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF2]][[[C0]], [[C0]]] {ttg.partition = 1 : i32}

    // CHECK: nvws.aref.get.exit [[AREF2]][[[C0]]], {{.*}} [#nvws.async_op<none>] {ttg.partition = 0 : i32}

    "op_c"(%1) {ttg.partition = 0} : (!ty) -> ()
    scf.yield
  } {ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
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
    // CHECK-NEXT: nvws.aref.put.enter [[AREF1]][[[C0]], [[C0]]] {ttg.partition = 0 : i32}

    %1 = "op_b"(%0) {ttg.partition = 1} : (!ty) -> !ty
    // CHECK: nvws.aref.get.exit [[AREF1]][[[C0]]], {{.*}} [#nvws.async_op<none>] {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF2]][[[C0]], [[C0]]] {ttg.partition = 1 : i32}

    %2 = "op_c"(%1) {ttg.partition = 2} : (!ty) -> !ty
    // CHECK: nvws.aref.get.exit [[AREF2]][[[C0]]], {{.*}} [#nvws.async_op<none>] {ttg.partition = 2 : i32}
    // CHECK-NEXT: "op_c"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF3]][[[C0]], [[C0]]] {ttg.partition = 2 : i32}

    "op_c"(%2) {ttg.partition = 0} : (!ty) -> ()
    // CHECK: nvws.aref.get.exit [[AREF3]][[[C0]]], {{.*}} [#nvws.async_op<none>] {ttg.partition = 0 : i32}
    // CHECK: "op_c"
    scf.yield
  } {ttg.partition.stages = [0, 2, 3], ttg.warp_specialize.tag = 0 : i32}
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
  } {ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
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
  } {ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

}
