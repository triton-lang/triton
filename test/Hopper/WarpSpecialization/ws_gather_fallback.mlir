// RUN: triton-opt %s --nvgpu-warp-specialization=num-stages=2 | FileCheck %s --check-prefix=CHECK
// RUN: triton-opt %s --nvgpu-warp-specialization=num-stages=2 | FileCheck %s --check-prefix=FALLBACK

// CHECK-LABEL: @gather_falls_back
// CHECK: tt.gather
// CHECK: ttng.warp_group_dot
// CHECK: tt.descriptor_store

// CHECK-LABEL: @gather_before_loop_falls_back
// CHECK: tt.gather
// CHECK: ttng.warp_group_dot
// CHECK: tt.descriptor_store

// CHECK-LABEL: @gather_after_loop_falls_back
// CHECK: ttng.warp_group_dot
// CHECK: tt.gather
// CHECK: tt.descriptor_store

// CHECK-LABEL: @gather_joined_with_partitioned_epilogue_falls_back
// CHECK: ttng.warp_group_dot
// CHECK: tt.gather
// CHECK: arith.addf
// CHECK: tt.descriptor_store

// CHECK-LABEL: @independent_gathers_feed_dot_falls_back
// CHECK: ttng.warp_group_dot
// CHECK: tt.gather
// CHECK: tt.gather
// CHECK: ttng.warp_group_dot
// CHECK: tt.descriptor_store

// CHECK-LABEL: @independent_dot_feeding_gather_falls_back
// CHECK: ttng.warp_group_dot
// CHECK: ttng.warp_group_dot
// CHECK: tt.gather

// CHECK-LABEL: @unmarked_sibling_loop_gather_fed_dot_falls_back
// CHECK: ttng.warp_group_dot
// CHECK: tt.gather
// CHECK: tt.gather
// CHECK: ttng.warp_group_dot
// CHECK: tt.descriptor_store

// CHECK-LABEL: @unrelated_epilogue_gather_keeps_warp_specialization
// CHECK: ttg.warp_specialize
// CHECK: tt.gather
// CHECK: tt.descriptor_store
// CHECK: tt.store

// FALLBACK-LABEL: @gather_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return
// FALLBACK-LABEL: @gather_before_loop_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.gather
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: ttng.warp_group_dot
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return
// FALLBACK-LABEL: @gather_after_loop_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.gather
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return
// FALLBACK-LABEL: @gather_joined_with_partitioned_epilogue_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.gather
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return
// FALLBACK-LABEL: @independent_gathers_feed_dot_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: ttng.warp_group_dot
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return
// FALLBACK-LABEL: @independent_dot_feeding_gather_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: ttng.warp_group_dot
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return
// FALLBACK-LABEL: @unmarked_sibling_loop_gather_fed_dot_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: ttng.warp_group_dot
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @gather_falls_back(%arg0: !tt.tensordesc<128x128xf16>, %arg1: !tt.tensordesc<128x128xf16>, %arg2: !tt.tensordesc<128x128xf16>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x128xi32, #blocked>
    %init = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x128xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked>
      %gathered = tt.gather %a[%indices] {axis = 1 : i32} : (tensor<128x128xf16, #blocked>, tensor<128x128xi32, #blocked>) -> tensor<128x128xf16, #blocked>
      %a_smem = ttg.local_alloc %gathered : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem> * !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma>
      scf.yield %dot : tensor<128x128xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x128xf16>, tensor<128x128xf16, #blocked1>
    tt.return
  }

  tt.func @gather_before_loop_falls_back(%arg0: !tt.tensordesc<128x128xf16>, %arg1: !tt.tensordesc<128x128xf16>, %arg2: !tt.tensordesc<128x128xf16>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x128xi32, #blocked>
    %a = tt.descriptor_load %arg0[%c0, %c0] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked>
    %gathered = tt.gather %a[%indices] {axis = 1 : i32} : (tensor<128x128xf16, #blocked>, tensor<128x128xi32, #blocked>) -> tensor<128x128xf16, #blocked>
    %init = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x128xf32, #mma> : i32 {
      %a_smem = ttg.local_alloc %gathered : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem> * !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma>
      scf.yield %dot : tensor<128x128xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x128xf16>, tensor<128x128xf16, #blocked1>
    tt.return
  }

  tt.func @gather_after_loop_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<0> : tensor<128x256xi32, #blocked1>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    %gathered = tt.gather %out_blocked[%indices] {axis = 1 : i32} : (tensor<128x256xf16, #blocked1>, tensor<128x256xi32, #blocked1>) -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %gathered : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @gather_joined_with_partitioned_epilogue_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %independent: tensor<128x256xf16, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<0> : tensor<128x256xi32, #blocked1>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    %output_gather = tt.gather %independent[%indices] {axis = 1 : i32} : (tensor<128x256xf16, #blocked1>, tensor<128x256xi32, #blocked1>) -> tensor<128x256xf16, #blocked1>
    %sum = arith.addf %out_blocked, %output_gather : tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %sum : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @independent_gathers_feed_dot_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %arg3: !tt.tensordesc<128x256xf16>, %independent: tensor<128x64xf16, #blocked>, %independent_b: tensor<64x256xf16, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x64xi32, #blocked>
    %indices_b = arith.constant dense<1> : tensor<64x256xi32, #blocked1>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %acc_out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %acc_blocked = ttg.convert_layout %acc_out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg3[%c0, %c0], %acc_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    %gathered = tt.gather %independent[%indices] {axis = 1 : i32} : (tensor<128x64xf16, #blocked>, tensor<128x64xi32, #blocked>) -> tensor<128x64xf16, #blocked>
    %lhs = ttg.local_alloc %gathered : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %gathered_rhs = tt.gather %independent_b[%indices_b] {axis = 1 : i32} : (tensor<64x256xf16, #blocked1>, tensor<64x256xi32, #blocked1>) -> tensor<64x256xf16, #blocked1>
    %rhs_smem = ttg.local_alloc %gathered_rhs : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %epilogue_dot = ttng.warp_group_dot %lhs, %rhs_smem, %init {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %out = arith.truncf %epilogue_dot : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @independent_dot_feeding_gather_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %independent: tensor<128x64xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x256xi32, #blocked1>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    %lhs = ttg.local_alloc %independent : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %rhs = tt.descriptor_load %arg1[%c0, %c0] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
    %rhs_smem = ttg.local_alloc %rhs : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %epilogue_dot = ttng.warp_group_dot %lhs, %rhs_smem, %init {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %epilogue_blocked = ttg.convert_layout %epilogue_dot : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
    %unused = tt.gather %epilogue_blocked[%indices] {axis = 1 : i32} : (tensor<128x256xf32, #blocked1>, tensor<128x256xi32, #blocked1>) -> tensor<128x256xf32, #blocked1>
    tt.return
  }

  tt.func @unmarked_sibling_loop_gather_fed_dot_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %arg3: !tt.tensordesc<128x256xf16>, %independent_a: tensor<128x64xf16, #blocked>, %independent_b: tensor<64x256xf16, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices_a = arith.constant dense<1> : tensor<128x64xi32, #blocked>
    %indices_b = arith.constant dense<1> : tensor<64x256xi32, #blocked1>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %selected = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %selected_f16 = arith.truncf %selected : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %selected_blocked = ttg.convert_layout %selected_f16 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %selected_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
      %gathered_a = tt.gather %independent_a[%indices_a] {axis = 1 : i32} : (tensor<128x64xf16, #blocked>, tensor<128x64xi32, #blocked>) -> tensor<128x64xf16, #blocked>
      %lhs = ttg.local_alloc %gathered_a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %gathered_b = tt.gather %independent_b[%indices_b] {axis = 1 : i32} : (tensor<64x256xf16, #blocked1>, tensor<64x256xi32, #blocked1>) -> tensor<64x256xf16, #blocked1>
      %rhs = ttg.local_alloc %gathered_b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %epilogue_dot = ttng.warp_group_dot %lhs, %rhs, %init {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %out = arith.truncf %epilogue_dot : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.descriptor_store %arg3[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
      scf.yield
    }
    tt.return
  }

  tt.func @unrelated_epilogue_gather_keeps_warp_specialization(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c128 = arith.constant 128 : i32
    %c256 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %a_desc = tt.make_tensor_descriptor %arg0, [%c128, %c256], [%c256_i64, %c1_i64] : <f16>, <128x64xf16, #shared>
    %b_desc = tt.make_tensor_descriptor %arg1, [%c256, %c256], [%c256_i64, %c1_i64] : <f16>, <64x256xf16, #shared>
    %c_desc = tt.make_tensor_descriptor %arg2, [%c128, %c256], [%c256_i64, %c1_i64] : <f16>, <128x256xf16, #shared>
    %independent_desc = tt.make_tensor_descriptor %arg3, [%c128, %c256], [%c256_i64, %c1_i64] : <f16>, <128x256xf16, #shared>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %a_desc[%c0, %i] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%c0, %i] : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    %independent = tt.descriptor_load %independent_desc[%c0, %c0] : !tt.tensordesc<128x256xf16, #shared> -> tensor<128x256xf16, #blocked1>
    %output_indices = arith.constant dense<0> : tensor<128x256xi32, #blocked1>
    %output_gather = tt.gather %independent[%output_indices] {axis = 1 : i32} : (tensor<128x256xf16, #blocked1>, tensor<128x256xi32, #blocked1>) -> tensor<128x256xf16, #blocked1>
    %gather_ptrs = tt.splat %arg4 : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
    %mask = arith.constant dense<false> : tensor<128x256xi1, #blocked1>
    tt.descriptor_store %c_desc[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16, #shared>, tensor<128x256xf16, #blocked1>
    // Consumer partitions repeat this independent store, so keep it inert.
    tt.store %gather_ptrs, %output_gather, %mask : tensor<128x256x!tt.ptr<f16>, #blocked1>
    tt.return
  }

}
