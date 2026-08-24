// RUN: triton-opt --split-input-file %s -triton-licm | FileCheck %s

tt.func @hoist_load_without_mask(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // Check if the load is hoisted
  // CHECK-LABEL: hoist_load_without_mask
  // CHECK: %[[TRIP_COUNT_CMP:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
  // CHECK: %[[SPLAT:.*]] = tt.splat %[[TRIP_COUNT_CMP]]
  // CHECK: %[[LOAD:.*]] = tt.load %[[_:.*]], %[[SPLAT]]
  // CHECK: arith.addf %[[LOAD]], %[[LOAD]]
  // CHECK: scf.for
  // CHECK-NOT: tt.load
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @hoist_two_loads_without_mask(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>, %arg6: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // CHECK-LABEL: hoist_two_loads_without_mask
  // CHECK: %[[TRIP_COUNT_CMP_1:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
  // CHECK: %[[SPLAT_1:.*]] = tt.splat %[[TRIP_COUNT_CMP_1]]
  // CHECK: %[[LOAD_1:.*]] = tt.load %[[_:.*]], %[[SPLAT_1]]
  // CHECK: %[[TRIP_COUNT_CMP_2:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
  // CHECK: %[[SPLAT_2:.*]] = tt.splat %[[TRIP_COUNT_CMP_2]]
  // CHECK: %[[LOAD_2:.*]] = tt.load %[[_:.*]], %[[SPLAT_2]]
  // CHECK: arith.addf %[[LOAD_1]], %[[LOAD_2]]
  // CHECK: scf.for
  // CHECK-NOT: tt.load
  %1 = scf.for %arg8 = %arg3 to %arg4 step %c1_i32 iter_args(%arg7 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0 : tensor<1024x!tt.ptr<f32>>
    %3 = tt.load %arg6 : tensor<1024x!tt.ptr<f32>>
    %4 = arith.addf %2, %3 : tensor<1024xf32>
    %5 = arith.addf %arg7, %4 : tensor<1024xf32>
    scf.yield %5 : tensor<1024xf32>
  }
  tt.store %arg5, %1 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @hoist_load_with_mask(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // Check if the load is hoisted
  // CHECK-LABEL: hoist_load_with_mask
  // CHECK: %[[MASK:.*]] = arith.cmpi
  // CHECK: %[[TRIP_COUNT_CMP:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
  // CHECK: %[[SPLAT:.*]] = tt.splat %[[TRIP_COUNT_CMP]]
  // CHECK: %[[AND:.*]] = arith.andi %[[SPLAT]], %[[MASK]]
  // CHECK: %[[LOAD:.*]] = tt.load %[[_:.*]], %[[AND]]
  // CHECK: arith.addf %[[LOAD]], %[[LOAD]]
  // CHECK: scf.for
  // CHECK-NOT: tt.load
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @cannot_hoist_with_print_in_loop(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK: arith.addf
  // CHECK: arith.addf
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    tt.print " x: " {hex = false, isSigned = array<i32: 0>} : %4 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @cannot_hoist_with_assert_in_loop(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK: arith.addf
  // CHECK: arith.addf
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %cmp = arith.cmpi sge, %arg4, %arg3 : i32
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    tt.assert %cmp, "cond must be true " : i1
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @cannot_hoist_with_store_in_loop(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>, %tmp: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK: arith.addf
  // CHECK: arith.addf
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    tt.store %tmp, %4, %0 : tensor<1024x!tt.ptr<f32>>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @hoist_cond_no_hoist_load_from_scf_while(%ptr: tensor<1024x!tt.ptr<f32>>, %arg1: i32, %arg2 : i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  // CHECK-LABEL: hoist_cond_no_hoist_load_from_scf_while
  // CHECK: %[[CST42:.*]] = arith.constant 42
  // CHECK: %[[ADD:.*]] = arith.addi %[[_:.*]], %[[CST42]]
  // CHECK: %[[COND:.*]] = arith.cmpi slt, %[[ADD]], %[[_:.*]]
  // CHECK: scf.while
  // CHECK: do
  // CHECK: tt.load
  // CHECK: arith.addf
  // CHECK: scf.yield
  %1 = scf.while (%arg0 = %cst) : (tensor<1024xf32>) -> (tensor<1024xf32>) {
    %cst_42 = arith.constant 42 : i32
    %add_42 = arith.addi %arg1, %cst_42 : i32
    %2 = arith.cmpi slt, %add_42, %arg2 : i32
    scf.condition(%2) %arg0 : tensor<1024xf32>
  } do {
  ^bb0(%arg0: tensor<1024xf32>):
    %3 = tt.load %ptr : tensor<1024x!tt.ptr<f32>>
    %4 = arith.addf %3, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %ptr, %1 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @local_alloc_speculation
  // CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<128xf32, #shared, #smem>
  // CHECK-NOT: ttg.local_alloc
  // CHECK: scf.for
  // CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
  // CHECK: ttg.local_alloc {{.*}} {allocation.offset = 0 : i32}
  // CHECK-NOT: ttg.local_alloc
  // CHECK: scf.yield
  tt.func @local_alloc_speculation(%src: tensor<128xf32, #blocked>, %n: i32) -> tensor<128xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%sum = %src) -> tensor<128xf32, #blocked> : i32 {
      %immutable = ttg.local_alloc %src : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #smem>
      %mutable = ttg.local_alloc %src : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
      %allocated = ttg.local_alloc %src {allocation.offset = 0 : i32} : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #smem>
      %a = ttg.local_load %immutable : !ttg.memdesc<128xf32, #shared, #smem> -> tensor<128xf32, #blocked>
      %b = ttg.local_load %mutable : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
      %c = ttg.local_load %allocated : !ttg.memdesc<128xf32, #shared, #smem> -> tensor<128xf32, #blocked>
      %sum_a = arith.addf %sum, %a : tensor<128xf32, #blocked>
      %sum_b = arith.addf %sum_a, %b : tensor<128xf32, #blocked>
      %sum_c = arith.addf %sum_b, %c : tensor<128xf32, #blocked>
      scf.yield %sum_c : tensor<128xf32, #blocked>
    }
    tt.return %result : tensor<128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_space = #ttng.tensor_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_alloc_speculation
  // CHECK: %[[ALLOC:.*]], %[[TOKEN:.*]] = ttng.tmem_alloc
  // CHECK-NOT: ttng.tmem_alloc
  // CHECK: scf.for
  // CHECK: ttng.tmem_alloc {{.*}} -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK: ttng.tmem_alloc {{.*}} {tensor_memory_col_offset = 0 : i32
  // CHECK-NOT: ttng.tmem_alloc
  // CHECK: ttng.tmem_load %[[ALLOC]][%[[TOKEN]]]
  // CHECK: scf.yield
  tt.func @tmem_alloc_speculation(%src: tensor<128x64xf32, #blocked>, %n: i32) -> tensor<128x64xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%sum = %src) -> tensor<128x64xf32, #blocked> : i32 {
      %immutable, %token = ttng.tmem_alloc %src : (tensor<128x64xf32, #blocked>) -> (!ttg.memdesc<128x64xf32, #tmem, #tmem_space>, !ttg.async.token)
      %mutable = ttng.tmem_alloc %src : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #tmem_space, mutable>
      %allocated = ttng.tmem_alloc %src {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #tmem_space>
      %a, %load_token = ttng.tmem_load %immutable[%token] : !ttg.memdesc<128x64xf32, #tmem, #tmem_space> -> tensor<128x64xf32, #blocked>
      %b = ttng.tmem_load %mutable : !ttg.memdesc<128x64xf32, #tmem, #tmem_space, mutable> -> tensor<128x64xf32, #blocked>
      %c = ttng.tmem_load %allocated : !ttg.memdesc<128x64xf32, #tmem, #tmem_space> -> tensor<128x64xf32, #blocked>
      %sum_a = arith.addf %sum, %a : tensor<128x64xf32, #blocked>
      %sum_b = arith.addf %sum_a, %b : tensor<128x64xf32, #blocked>
      %sum_c = arith.addf %sum_b, %c : tensor<128x64xf32, #blocked>
      scf.yield %sum_c : tensor<128x64xf32, #blocked>
    }
    tt.return %result : tensor<128x64xf32, #blocked>
  }
}
