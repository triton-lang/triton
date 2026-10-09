// RUN: triton-opt %s --tritongpu-optimize-thread-availability | FileCheck %s --implicit-check-not=ttg.availability --implicit-check-not=ttg.materialize

// CHECK: #[[$SCALAR:[a-zA-Z0-9_]+]] = #ttg.linear<{register = [], lane = {{.*}}, warp = {{.*}}, block = []}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A scalar load/add/store only consumes the canonical thread's copy.
  // CHECK-LABEL: @availability_scalar(
  // CHECK: %[[S_LOAD:.*]] = tt.load {{.*}} : tensor<!tt.ptr<i32>, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: %[[S_ADD:.*]] = arith.addi %[[S_LOAD]], %{{.*}} : tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: tt.store %{{.*}}, %[[S_ADD]] : tensor<!tt.ptr<i32>, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK-NEXT: tt.return
  tt.func @availability_scalar(%a: !tt.ptr<i32>, %b: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    %x = tt.load %a : !tt.ptr<i32>
    %y = arith.addi %x, %one : i32
    tt.store %b, %y : !tt.ptr<i32>
    tt.return
  }

  // arange(0, 32) needs every lane of warp zero, and no other warp.
  // CHECK-LABEL: @availability_warp(
  // CHECK: %[[W_LOAD:.*]] = tt.load {{.*}} : tensor<!tt.ptr<i32>, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK-NEXT: %[[W_BROADCAST:.*]] = ttg.convert_layout %[[W_LOAD]] : tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>> -> tensor<i32, #ttg.partial<#[[$SCALAR]], <0, 3, 0>>>
  // CHECK-NEXT: %[[W_VALUES:.*]] = ttg.splat_scalar %[[W_BROADCAST]] : tensor<i32, #ttg.partial<#[[$SCALAR]], <0, 3, 0>>> -> tensor<32xi32, #ttg.partial<#blocked, <0, 3, 0>>>
  // CHECK: arith.addi %[[W_VALUES]], %{{.*}} : tensor<32xi32, #ttg.partial<#blocked, <0, 3, 0>>>
  // CHECK: tt.store {{.*}} : tensor<32x!tt.ptr<i32>, #ttg.partial<#blocked, <0, 3, 0>>>
  // CHECK-NEXT: tt.return
  tt.func @availability_warp(%a: !tt.ptr<i32>, %b: !tt.ptr<i32>) {
    %x = tt.load %a : !tt.ptr<i32>
    %xs = tt.splat %x : i32 -> tensor<32xi32, #blocked>
    %offsets = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %base = tt.splat %b : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
    %y = arith.addi %xs, %offsets : tensor<32xi32, #blocked>
    tt.store %ptrs, %y : tensor<32x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // The dead final value needs no transport, but completion still needs a
  // rendezvous. Both the first load and every iteration retain acquire semantics.
  // CHECK-LABEL: @availability_poll(
  // CHECK: %[[P_INITIAL:.*]] = tt.atomic_load acquire, gpu, {{.*}} -> tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK-NEXT: %[[P_LEADER:.*]] = ttg.thread_predicate <31, 3, 0>
  // CHECK-NEXT: scf.if %[[P_LEADER]] {
  // CHECK-NEXT: scf.while (%[[P_ARG:.*]] = %[[P_INITIAL]]) : (tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>) -> ()
  // CHECK: %[[P_CMP:.*]] = arith.cmpi slt, %[[P_ARG]], %{{.*}} : tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK-NEXT: %[[P_COND:.*]] = ttg.extract_scalar %[[P_CMP]] <31, 3, 0> : tensor<i1, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>> -> i1
  // CHECK-NEXT: scf.condition(%[[P_COND]])
  // CHECK: } do {
  // CHECK: %[[P_NEXT:.*]] = tt.atomic_load acquire, gpu, {{.*}} -> tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK-NEXT: scf.yield %[[P_NEXT]] : tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: ttg.barrier all
  // CHECK-NEXT: tt.return
  tt.func @availability_poll(%ptr: !tt.ptr<i32>, %target: i32) {
    %true = arith.constant true
    %initial = tt.atomic_load acquire, gpu, %ptr, %true : (!tt.ptr<i32>, i1) -> i32
    scf.while (%a = %initial) : (i32) -> () {
      %cond = arith.cmpi slt, %a, %target : i32
      scf.condition(%cond)
    } do {
      %next = tt.atomic_load acquire, gpu, %ptr, %true : (!tt.ptr<i32>, i1) -> i32
      scf.yield %next : i32
    }
    tt.return
  }

  // A live-out result is materialized only after the leader finishes polling.
  // CHECK-LABEL: @availability_poll_warp_result(
  // CHECK: %[[R_LEADER:.*]] = ttg.thread_predicate <31, 3, 0>
  // CHECK-NEXT: %[[R_FINAL:.*]] = scf.if %[[R_LEADER]] -> (tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>)
  // CHECK: scf.while {{.*}} -> tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: tt.atomic_load acquire, gpu{{.*}} -> tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: ttg.barrier all
  // CHECK-NEXT: %[[R_BROADCAST:.*]] = ttg.convert_layout %[[R_FINAL]] : tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>> -> tensor<i32, #ttg.partial<#[[$SCALAR]], <0, 3, 0>>>
  // CHECK-NEXT: ttg.splat_scalar %[[R_BROADCAST]]
  // CHECK: tt.return
  tt.func @availability_poll_warp_result(%ptr: !tt.ptr<i32>, %target: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %initial = tt.atomic_load acquire, gpu, %ptr, %true : (!tt.ptr<i32>, i1) -> i32
    %final = scf.while (%a = %initial) : (i32) -> i32 {
      %cond = arith.cmpi slt, %a, %target : i32
      scf.condition(%cond) %a : i32
    } do {
    ^bb0(%a: i32):
      %next = tt.atomic_load acquire, gpu, %ptr, %true : (!tt.ptr<i32>, i1) -> i32
      scf.yield %next : i32
    }
    %values = tt.splat %final : i32 -> tensor<32xi32, #blocked>
    %offsets = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
    %result = arith.addi %values, %offsets : tensor<32xi32, #blocked>
    tt.store %ptrs, %result : tensor<32x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // Equal placement maps still require communication when availability widens.
  // CHECK-LABEL: @availability_poll_cta_result(
  // CHECK: %[[C_FINAL:.*]] = scf.if {{.*}} -> (tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>)
  // CHECK: ttg.barrier all
  // CHECK-NEXT: %[[C_FULL:.*]] = ttg.convert_layout %[[C_FINAL]] : tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>> -> tensor<i32, #[[$SCALAR]]>
  // CHECK-NEXT: ttg.splat_scalar %[[C_FULL]] : tensor<i32, #[[$SCALAR]]> -> tensor<128xi32, #blocked>
  // CHECK: tt.store {{.*}} : tensor<128x!tt.ptr<i32>, #blocked>
  // CHECK-NEXT: tt.return
  tt.func @availability_poll_cta_result(%ptr: !tt.ptr<i32>, %target: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %initial = tt.atomic_load acquire, gpu, %ptr, %true : (!tt.ptr<i32>, i1) -> i32
    %final = scf.while (%a = %initial) : (i32) -> i32 {
      %cond = arith.cmpi slt, %a, %target : i32
      scf.condition(%cond) %a : i32
    } do {
    ^bb0(%a: i32):
      %next = tt.atomic_load acquire, gpu, %ptr, %true : (!tt.ptr<i32>, i1) -> i32
      scf.yield %next : i32
    }
    %values = tt.splat %final : i32 -> tensor<128xi32, #blocked>
    %offsets = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %result = arith.addi %values, %offsets : tensor<128xi32, #blocked>
    tt.store %ptrs, %result : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // Availability is part of each result and region argument, even when the
  // carried values have different element types.
  // CHECK-LABEL: @availability_poll_two_values(
  // CHECK: scf.if {{.*}} -> (tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>, tensor<i64, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>)
  // CHECK: scf.while {{.*}} : (tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>, tensor<i64, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>) -> (tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>, tensor<i64, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>)
  // CHECK: scf.condition{{.*}} : tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>, tensor<i64, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: ^bb0(%{{.*}}: tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>, %{{.*}}: tensor<i64, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>):
  // CHECK: arith.addi {{.*}} : tensor<i64, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: ttg.barrier all
  // CHECK: tt.store {{.*}} : tensor<!tt.ptr<i64>, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: tt.return
  tt.func @availability_poll_two_values(%ptr: !tt.ptr<i32>, %target: i32, %out: !tt.ptr<i64>) {
    %true = arith.constant true
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %initial = tt.atomic_load acquire, gpu, %ptr, %true : (!tt.ptr<i32>, i1) -> i32
    %final:2 = scf.while (%a = %initial, %count = %zero) : (i32, i64) -> (i32, i64) {
      %cond = arith.cmpi slt, %a, %target : i32
      scf.condition(%cond) %a, %count : i32, i64
    } do {
    ^bb0(%a: i32, %count: i64):
      %next = tt.atomic_load acquire, gpu, %ptr, %true : (!tt.ptr<i32>, i1) -> i32
      %n = arith.addi %count, %one : i64
      scf.yield %next, %n : i32, i64
    }
    tt.store %out, %final#1 : !tt.ptr<i64>
    tt.return
  }

  // Conditional scalar polling stays inside the leader's execution domain.
  // CHECK-LABEL: @availability_conditional_poll(
  // CHECK: ttg.thread_predicate <31, 3, 0>
  // CHECK: scf.while
  // CHECK: ttg.extract_scalar {{.*}} <31, 3, 0> : tensor<i1, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>> -> i1
  // CHECK: scf.if {{.*}} -> (tensor<i1, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>)
  // CHECK: tt.atomic_load acquire, sys, {{.*}} {ttg.thread_local}
  // CHECK: scf.yield {{.*}} : tensor<i1, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: ttg.barrier all
  // CHECK: tt.return
  tt.func @availability_conditional_poll(%ptr: !tt.ptr<i32>, %abort_ptr: !tt.ptr<i32>, %target: i32) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %three = arith.constant 3 : i32
    %initial = tt.atomic_load acquire, sys, %ptr, %true : (!tt.ptr<i32>, i1) -> i32
    scf.while (%a = %initial, %aborted = %false, %spins = %zero) : (i32, i1, i32) -> (i1, i32) {
      %pending = arith.cmpi slt, %a, %target : i32
      %alive = arith.xori %aborted, %true : i1
      %again = arith.andi %pending, %alive : i1
      scf.condition(%again) %aborted, %spins : i1, i32
    } do {
    ^bb0(%aborted: i1, %spins: i32):
      %a = tt.atomic_load acquire, sys, %ptr, %true : (!tt.ptr<i32>, i1) -> i32
      %next = arith.addi %spins, %one : i32
      %low = arith.andi %next, %three : i32
      %check = arith.cmpi eq, %low, %zero : i32
      %abort = scf.if %check -> i1 {
        %address = tt.addptr %abort_ptr, %one : !tt.ptr<i32>, i32
        %status = tt.atomic_load acquire, sys, %address, %true : (!tt.ptr<i32>, i1) -> i32
        %set = arith.cmpi ne, %status, %zero : i32
        scf.yield %set : i1
      } else {
        scf.yield %aborted : i1
      }
      scf.yield %a, %abort, %next : i32, i1, i32
    }
    tt.return
  }

  // Inspect both arms: a scalar branch containing a write is not a read-only
  // polling region and must retain its original execution contract.
  // CHECK-LABEL: @availability_effectful_branch(
  // CHECK-NOT: ttg.thread_predicate
  // CHECK-NOT: ttg.thread_local
  // CHECK: tt.return
  tt.func @availability_effectful_branch(%ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>, %target: i32, %write: i1) {
    %initial = tt.atomic_load acquire, gpu, %ptr : (!tt.ptr<i32>) -> i32
    scf.while (%a = %initial) : (i32) -> () {
      %again = arith.cmpi slt, %a, %target : i32
      scf.condition(%again)
    } do {
      %next = tt.atomic_load acquire, gpu, %ptr : (!tt.ptr<i32>) -> i32
      scf.if %write {
        tt.store %out, %next : !tt.ptr<i32>
      }
      scf.yield %next : i32
    }
    tt.return
  }

  // A speculatable shift can still produce poison in an unavailable copy.
  // CHECK-LABEL: @availability_poison_fallback(
  // CHECK-NOT: #ttg.partial
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return
  tt.func @availability_poison_fallback(%a: !tt.ptr<i32>, %b: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    %x = tt.load %a : !tt.ptr<i32>
    %amount = arith.subi %x, %one : i32
    %shifted = arith.shli %one, %amount : i32
    %mask = arith.cmpi ne, %shifted, %one : i32
    tt.store %b, %shifted, %mask : !tt.ptr<i32>
    tt.return
  }
  // Queue reservation and publication share one thread. Internal rendezvous
  // are replaced with entry/exit synchronization, including nested barriers.
  // CHECK-LABEL: @availability_scalar_protocol(
  // CHECK: ttg.barrier all
  // CHECK: ttg.thread_predicate <31, 3, 0>
  // CHECK: scf.if
  // CHECK-NOT: ttg.barrier
  // CHECK: scf.while
  // CHECK: tt.atomic_cas relaxed, gpu, {{.*}} {ttg.thread_local}
  // CHECK-SAME: #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK-NOT: ttg.barrier
  // CHECK: tt.atomic_store release, sys, {{.*}} {ttg.thread_local}
  // CHECK: tt.atomic_rmw add, relaxed, gpu, {{.*}} {ttg.thread_local}
  // CHECK-NOT: ttg.barrier
  // CHECK: ttg.execution_domain
  // CHECK-NEXT: ttg.barrier all
  // CHECK: ttg.convert_layout {{.*}} -> tensor<i32, #ttg.partial<#[[$SCALAR]], <0, 3, 0>>>
  // CHECK: ttg.splat_scalar
  // CHECK: tt.return
  tt.func @availability_scalar_protocol(%tail: !tt.ptr<i32>, %ready: !tt.ptr<i32>, %count: !tt.ptr<i32>, %out: !tt.ptr<i32>, %limit: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    ttg.barrier all
    %ticket = scf.while (%current = %zero) : (i32) -> i32 {
      %pending = arith.cmpi slt, %current, %limit : i32
      scf.condition(%pending) %current : i32
    } do {
    ^bb0(%current: i32):
      %next = arith.addi %current, %one : i32
      %old = tt.atomic_cas relaxed, gpu, %tail, %current, %next : (!tt.ptr<i32>, i32, i32) -> i32
      ttg.barrier all
      scf.yield %old : i32
    }
    tt.atomic_store release, sys, %ready, %ticket : !tt.ptr<i32>
    %sent = tt.atomic_rmw add, relaxed, gpu, %count, %one : (!tt.ptr<i32>, i32) -> i32
    ttg.barrier all
    %values = tt.splat %ticket : i32 -> tensor<32xi32, #blocked>
    %offsets = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %ptrs, %values : tensor<32x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // A tensor effect inside the loop needs its original participating threads.
  // CHECK-LABEL: @availability_protocol_tensor_effect(
  // CHECK-NOT: ttg.thread_predicate
  // CHECK-NOT: ttg.thread_local
  // CHECK: tt.return
  tt.func @availability_protocol_tensor_effect(%ptr: !tt.ptr<i32>, %out: tensor<32x!tt.ptr<i32>, #blocked>, %target: i32) {
    %initial = tt.atomic_load acquire, gpu, %ptr : (!tt.ptr<i32>) -> i32
    scf.while (%a = %initial) : (i32) -> () {
      %again = arith.cmpi slt, %a, %target : i32
      scf.condition(%again)
    } do {
      %next = tt.atomic_load acquire, gpu, %ptr : (!tt.ptr<i32>) -> i32
      %values = tt.splat %next : i32 -> tensor<32xi32, #blocked>
      tt.store %out, %values : tensor<32x!tt.ptr<i32>, #blocked>
      tt.atomic_store release, gpu, %ptr, %next : !tt.ptr<i32>
      scf.yield %next : i32
    }
    tt.return
  }
  // A reduction consumes the complete tensor, then keeps only the scalar
  // store's copy of the result. The combine region still operates on scalars.
  // CHECK-LABEL: @availability_reduce_scalar(
  // CHECK: %[[REDUCED:.*]] = "tt.reduce"(%{{.*}})
  // CHECK: arith.addf {{.*}} : f32
  // CHECK: tt.reduce.return {{.*}} : f32
  // CHECK: (tensor<128xf32, #blocked>) -> tensor<f32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.store {{.*}} %[[REDUCED]] : tensor<!tt.ptr<f32>, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: tt.return
  tt.func @availability_reduce_scalar(%x: tensor<128xf32, #blocked>, %out: !tt.ptr<f32>) {
    %sum = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %value = arith.addf %a, %b : f32
      tt.reduce.return %value : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.store %out, %sum : !tt.ptr<f32>
    tt.return
  }

  // Every lane of warp zero consumes the final sum; other warps do not.
  // CHECK-LABEL: @availability_reduce_warp(
  // CHECK: "tt.reduce"
  // CHECK: (tensor<128xf32, #blocked>) -> tensor<f32, #ttg.partial<#[[$SCALAR]], <0, 3, 0>>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: ttg.splat_scalar
  // CHECK: tt.return
  tt.func @availability_reduce_warp(%x: tensor<128xf32, #blocked>, %out: tensor<32x!tt.ptr<f32>, #blocked>) {
    %sum = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %value = arith.addf %a, %b : f32
      tt.reduce.return %value : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    %values = tt.splat %sum : f32 -> tensor<32xf32, #blocked>
    tt.store %out, %values : tensor<32x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // A consumer spanning all warps retains full result availability.
  // CHECK-LABEL: @availability_reduce_full(
  // CHECK-NOT: #ttg.partial
  // CHECK: (tensor<128xf32, #blocked>) -> f32
  // CHECK-NOT: #ttg.partial
  // CHECK: tt.return
  tt.func @availability_reduce_full(%x: tensor<128xf32, #blocked>, %out: tensor<128x!tt.ptr<f32>, #blocked>) {
    %sum = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %value = arith.addf %a, %b : f32
      tt.reduce.return %value : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    %values = tt.splat %sum : f32 -> tensor<128xf32, #blocked>
    tt.store %out, %values : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // Tuple reductions retain their scalar combiner arguments and can return
  // partial copies when all components have narrow consumers.
  // CHECK-LABEL: @availability_reduce_tuple(
  // CHECK: "tt.reduce"
  // CHECK: tt.reduce.return {{.*}} : i32, f32
  // CHECK: -> (tensor<i32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>, tensor<f32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>)
  // CHECK: tt.return
  tt.func @availability_reduce_tuple(%x: tensor<128xi32, #blocked>, %y: tensor<128xf32, #blocked>, %out_x: !tt.ptr<i32>, %out_y: !tt.ptr<f32>) {
    %sum:2 = "tt.reduce"(%x, %y) ({
    ^bb0(%a: i32, %b: f32, %c: i32, %d: f32):
      %i = arith.addi %a, %c : i32
      %f = arith.addf %b, %d : f32
      tt.reduce.return %i, %f : i32, f32
    }) {axis = 0 : i32} : (tensor<128xi32, #blocked>, tensor<128xf32, #blocked>) -> (i32, f32)
    tt.store %out_x, %sum#0 : !tt.ptr<i32>
    tt.store %out_y, %sum#1 : !tt.ptr<f32>
    tt.return
  }
  // Float casts may omit their optional fast-math attribute. The reduction
  // result remains partial through the widening cast to the store's dtype.
  // CHECK-LABEL: @availability_reduce_cast(
  // CHECK: (tensor<128xf16, #blocked>) -> tensor<f16, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: arith.extf {{.*}} : tensor<f16, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>> to tensor<f32, #ttg.partial<#[[$SCALAR]], <31, 3, 0>>>
  // CHECK: tt.return
  tt.func @availability_reduce_cast(%x: tensor<128xf16, #blocked>, %out: !tt.ptr<f32>) {
    %sum = "tt.reduce"(%x) ({
    ^bb0(%a: f16, %b: f16):
      %value = arith.addf %a, %b : f16
      tt.reduce.return %value : f16
    }) {axis = 0 : i32} : (tensor<128xf16, #blocked>) -> f16
    %wide = arith.extf %sum : f16 to f32
    tt.store %out, %wide : !tt.ptr<f32>
    tt.return
  }

  // An associative first-nonzero combiner with control flow keeps its existing
  // lowering: inactive final-stage values must not enter a branch predicate.
  // CHECK-LABEL: @availability_reduce_control_flow(
  // CHECK: "tt.reduce"
  // CHECK: scf.if
  // CHECK: (tensor<128xi32, #blocked>) -> i32
  // CHECK: tt.return
  tt.func @availability_reduce_control_flow(%x: tensor<128xi32, #blocked>, %out: !tt.ptr<i32>) {
    %sum = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %zero = arith.constant 0 : i32
      %nonzero = arith.cmpi ne, %a, %zero : i32
      %value = scf.if %nonzero -> i32 {
        scf.yield %a : i32
      } else {
        scf.yield %b : i32
      }
      tt.reduce.return %value : i32
    }) {axis = 0 : i32} : (tensor<128xi32, #blocked>) -> i32
    tt.store %out, %sum : !tt.ptr<i32>
    tt.return
  }
}
