// RUN: triton-opt %s --nvgpu-warp-specialization=num-stages=2 | FileCheck %s

// CHECK-LABEL: @gather_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-LABEL: @dead_gather_on_dot_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-LABEL: @gather_before_loop_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-LABEL: @gather_after_loop_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-LABEL: @independent_gather_joined_with_loop_scalar_keeps_warp_specialization
// CHECK-DAG: arith.constant {{.*}}dense<false> : tensor<128x256xi1, #blocked1>
// CHECK-DAG: ttg.warp_specialize
// CHECK: tt.gather
// CHECK: arith.addf
// CHECK: tt.store
// CHECK-LABEL: @gather_joined_with_partitioned_epilogue_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: arith.addf
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.descriptor_store
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-LABEL: @independent_gather_fed_dot_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-LABEL: @independent_dot_feeding_gather_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-LABEL: @unselected_loop_dot_feeding_gather_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-LABEL: @unselected_loop_gather_feeds_dot_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-LABEL: @unselected_loop_gather_feeds_next_iteration_dot_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.gather
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @gather_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x64xi32, #blocked>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %gathered = tt.gather %a[%indices] {axis = 1 : i32} : (tensor<128x64xf16, #blocked>, tensor<128x64xi32, #blocked>) -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %gathered : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @dead_gather_on_dot_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x256xi32, #blocked1>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %dot_blocked = ttg.convert_layout %dot : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
      %unused = tt.gather %dot_blocked[%indices] {axis = 1 : i32} : (tensor<128x256xf32, #blocked1>, tensor<128x256xi32, #blocked1>) -> tensor<128x256xf32, #blocked1>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @gather_before_loop_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x64xi32, #blocked>
    %a = tt.descriptor_load %arg0[%c0, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
    %gathered = tt.gather %a[%indices] {axis = 1 : i32} : (tensor<128x64xf16, #blocked>, tensor<128x64xi32, #blocked>) -> tensor<128x64xf16, #blocked>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a_smem = ttg.local_alloc %gathered : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
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

  tt.func @independent_gather_joined_with_loop_scalar_keeps_warp_specialization(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<f16>, %independent: f16) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32
    %c256 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %a_desc = tt.make_tensor_descriptor %arg0, [%c128, %c256], [%c256_i64, %c1_i64] : <f16>, <128x64xf16, #shared>
    %b_desc = tt.make_tensor_descriptor %arg1, [%c256, %c256], [%c256_i64, %c1_i64] : <f16>, <64x256xf16, #shared>
    %c_desc = tt.make_tensor_descriptor %arg2, [%c128, %c256], [%c256_i64, %c1_i64] : <f16>, <128x256xf16, #shared>
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %last_i = scf.for %i = %c0 to %c4 step %c1 iter_args(%index = %c0) -> i32 : i32 {
      %a = tt.descriptor_load %a_desc[%c0, %i] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%c0, %i] : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %init {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %out = arith.truncf %dot : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.descriptor_store %c_desc[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16, #shared>, tensor<128x256xf16, #blocked1>
      scf.yield %i : i32
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %output_indices = arith.constant dense<0> : tensor<128x256xi32, #blocked1>
    %independent_tensor = tt.splat %independent : f16 -> tensor<128x256xf16, #blocked1>
    %output_gather = tt.gather %independent_tensor[%output_indices] {axis = 1 : i32} : (tensor<128x256xf16, #blocked1>, tensor<128x256xi32, #blocked1>) -> tensor<128x256xf16, #blocked1>
    %scalar = arith.sitofp %last_i : i32 to f16
    %scalar_tensor = tt.splat %scalar : f16 -> tensor<128x256xf16, #blocked1>
    %sum = arith.addf %output_gather, %scalar_tensor : tensor<128x256xf16, #blocked1>
    %gather_ptrs = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
    // This consumer can be cloned into each partition, so keep the store inert.
    %mask = arith.constant dense<false> : tensor<128x256xi1, #blocked1>
    tt.store %gather_ptrs, %sum, %mask : tensor<128x256x!tt.ptr<f16>, #blocked1>
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

  tt.func @independent_gather_fed_dot_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %arg3: !tt.tensordesc<128x256xf16>, %independent: tensor<128x64xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x64xi32, #blocked>
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
    %rhs = tt.descriptor_load %arg1[%c0, %c0] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
    %rhs_smem = ttg.local_alloc %rhs : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
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

  tt.func @unselected_loop_dot_feeding_gather_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x256xi32, #blocked1>
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
    %loop_dot = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    }
    %loop_dot_f16 = arith.truncf %loop_dot : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %loop_dot_blocked = ttg.convert_layout %loop_dot_f16 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    %unused = tt.gather %loop_dot_blocked[%indices] {axis = 1 : i32} : (tensor<128x256xf16, #blocked1>, tensor<128x256xi32, #blocked1>) -> tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @unselected_loop_gather_feeds_dot_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %independent: tensor<128x64xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<1> : tensor<128x64xi32, #blocked>
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
    %loop_gather = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %independent) -> tensor<128x64xf16, #blocked> : i32 {
      %gathered = tt.gather %iter[%indices] {axis = 1 : i32} : (tensor<128x64xf16, #blocked>, tensor<128x64xi32, #blocked>) -> tensor<128x64xf16, #blocked>
      scf.yield %gathered : tensor<128x64xf16, #blocked>
    }
    %lhs = ttg.local_alloc %loop_gather : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %rhs = tt.descriptor_load %arg1[%c0, %c0] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
    %rhs_smem = ttg.local_alloc %rhs : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %dot = ttng.warp_group_dot %lhs, %rhs_smem, %init {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    tt.return
  }

  tt.func @unselected_loop_gather_feeds_next_iteration_dot_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %arg3: !tt.tensordesc<128x256xf16>, %independent: tensor<128x64xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %indices = arith.constant dense<1> : tensor<128x64xi32, #blocked>
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
    %loop_gather = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %independent) -> tensor<128x64xf16, #blocked> : i32 {
      %lhs = ttg.local_alloc %iter : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %rhs = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %rhs_smem = ttg.local_alloc %rhs : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %lhs, %rhs_smem, %init {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %dot_f16 = arith.truncf %dot : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %dot_blocked = ttg.convert_layout %dot_f16 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.descriptor_store %arg3[%c0, %c0], %dot_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
      %gathered = tt.gather %iter[%indices] {axis = 1 : i32} : (tensor<128x64xf16, #blocked>, tensor<128x64xi32, #blocked>) -> tensor<128x64xf16, #blocked>
      scf.yield %gathered : tensor<128x64xf16, #blocked>
    }
    tt.return
  }
}
