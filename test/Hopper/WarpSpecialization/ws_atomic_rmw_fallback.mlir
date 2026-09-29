// RUN: triton-opt %s --nvgpu-warp-specialization=num-stages=2 | FileCheck %s
// RUN: triton-opt %s --nvgpu-warp-specialization=num-stages=2 | FileCheck %s --check-prefix=FALLBACK
// RUN: triton-opt %s --nvgpu-test-ws-task-partition=num-warp-groups=3 --nvgpu-test-taskid-propagate=num-warp-groups=3 | FileCheck %s --check-prefix=TASK-ID

// CHECK-LABEL: @atomic_rmw_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: async_task_id
// CHECK: tt.atomic_rmw fadd
// CHECK-NOT: tt.atomic_rmw fadd
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: async_task_id

// FALLBACK-LABEL: @atomic_rmw_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return

// CHECK-LABEL: @atomic_epilogue_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.warp_group_dot
// CHECK-NOT: async_task_id
// CHECK: tt.num_stages = 2 : i32
// CHECK-NOT: async_task_id
// CHECK: tt.atomic_rmw fadd
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: tt.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-LABEL: @producer_atomic_keeps_warp_specialization
// CHECK-NOT: tt.atomic_rmw add
// CHECK: ttg.warp_specialize
// CHECK: tt.atomic_rmw add
// CHECK-NOT: tt.atomic_rmw add
// CHECK: partition0
// CHECK-NOT: tt.atomic_rmw add
// CHECK: partition1
// CHECK-NOT: tt.atomic_rmw add

// FALLBACK-LABEL: @atomic_epilogue_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return
// FALLBACK-LABEL: @producer_atomic_keeps_warp_specialization
// FALLBACK-LABEL: @producer_tensor_atomic_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return

// CHECK-LABEL: @producer_tensor_atomic_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK: tt.atomic_rmw add
// CHECK: tt.load
// CHECK: ttng.warp_group_dot
// CHECK-NOT: ttg.warp_specialize
// CHECK-LABEL: @vector_consumer_atomic_falls_back
// CHECK-NOT: ttg.warp_specialize
// CHECK: ttng.warp_group_dot
// CHECK: tt.atomic_rmw fadd
// CHECK: tt.store
// CHECK-NOT: ttg.warp_specialize

// FALLBACK-LABEL: @vector_consumer_atomic_falls_back
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.num_stages = 2 : i32
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: tt.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK: tt.return

// TASK-ID-LABEL: @producer_atomic_keeps_warp_specialization
// TASK-ID: tt.atomic_rmw add{{.*}}async_task_id = array<i32: 0>
// TASK-ID-LABEL: @producer_tensor_atomic_falls_back
// TASK-ID: tt.atomic_rmw add{{.*}}async_task_id = array<i32: 0>
// TASK-ID-LABEL: @vector_consumer_atomic_falls_back
// TASK-ID: tt.atomic_rmw fadd{{.*}}async_task_id = array<i32: 1, 2>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @atomic_rmw_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %atomic_ptr: !tt.ptr<f32>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %value = ttg.convert_layout %dot : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
      %ptrs = tt.splat %atomic_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
      %mask = arith.constant dense<true> : tensor<128x256xi1, #blocked1>
      %atomic = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %value, %mask : (tensor<128x256x!tt.ptr<f32>, #blocked1>, tensor<128x256xf32, #blocked1>, tensor<128x256xi1, #blocked1>) -> tensor<128x256xf32, #blocked1>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @atomic_epilogue_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %atomic_ptr: !tt.ptr<f32>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %value = ttg.convert_layout %acc : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
    %ptrs = tt.splat %atomic_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
    %mask = arith.constant dense<true> : tensor<128x256xi1, #blocked1>
    %atomic = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %value, %mask : (tensor<128x256x!tt.ptr<f32>, #blocked1>, tensor<128x256xf32, #blocked1>, tensor<128x256xi1, #blocked1>) -> tensor<128x256xf32, #blocked1>
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %arg2[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @producer_atomic_keeps_warp_specialization(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<i32>, %arg4: !tt.ptr<i32>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32
    %c256 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %a_desc = tt.make_tensor_descriptor %arg0, [%c128, %c256], [%c256_i64, %c1_i64] : <f16>, <128x64xf16, #shared>
    %b_desc = tt.make_tensor_descriptor %arg1, [%c256, %c256], [%c256_i64, %c1_i64] : <f16>, <64x256xf16, #shared>
    %c_desc = tt.make_tensor_descriptor %arg2, [%c128, %c256], [%c256_i64, %c1_i64] : <f16>, <128x256xf16, #shared>
    %c_m = arith.constant 0 : i32
    %b_n = arith.constant 0 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %acc = scf.for %k = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %offset = tt.load %arg4 : !tt.ptr<i32>
      %coordinate = tt.atomic_rmw add, relaxed, gpu, %arg3, %c64 : (!tt.ptr<i32>, i32) -> i32
      %index = arith.addi %offset, %coordinate : i32
      %a = tt.descriptor_load %a_desc[%c0, %index] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%index, %b_n] : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %c_desc[%c_m, %c0], %out_blocked : !tt.tensordesc<128x256xf16, #shared>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @producer_tensor_atomic_falls_back(%a_ptrs: tensor<128x64x!tt.ptr<f16>, #blocked>, %counter_ptrs: tensor<128x64x!tt.ptr<i32>, #blocked>, %b_desc: !tt.tensordesc<64x256xf16>, %out_desc: !tt.tensordesc<128x256xf16>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %one = arith.constant dense<1> : tensor<128x64xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<128x64xi1, #blocked>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %offsets = tt.atomic_rmw add, relaxed, gpu, %counter_ptrs, %one, %mask : (tensor<128x64x!tt.ptr<i32>, #blocked>, tensor<128x64xi32, #blocked>, tensor<128x64xi1, #blocked>) -> tensor<128x64xi32, #blocked>
      %load_ptrs = tt.addptr %a_ptrs, %offsets : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
      %a = tt.load %load_ptrs : tensor<128x64x!tt.ptr<f16>, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %out = arith.truncf %acc : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
    tt.descriptor_store %out_desc[%c0, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    tt.return
  }

  tt.func @vector_consumer_atomic_falls_back(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %atomic_ptrs: tensor<128x256x!tt.ptr<f32>, #blocked1>, %result_ptrs: tensor<128x256x!tt.ptr<f32>, #blocked1>, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %value = arith.constant dense<1.000000e+00> : tensor<128x256xf32, #blocked1>
    %mask = arith.constant dense<true> : tensor<128x256xi1, #blocked1>
    %acc = scf.for %i = %c0 to %iterations step %c1 iter_args(%iter = %init) -> tensor<128x256xf32, #mma> : i32 {
      %a = tt.descriptor_load %arg0[%i, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %arg1[%c0, %i] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_smem = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_smem, %iter {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %old = tt.atomic_rmw fadd, relaxed, gpu, %atomic_ptrs, %value, %mask : (tensor<128x256x!tt.ptr<f32>, #blocked1>, tensor<128x256xf32, #blocked1>, tensor<128x256xi1, #blocked1>) -> tensor<128x256xf32, #blocked1>
      %out = arith.truncf %dot : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %out_blocked = ttg.convert_layout %out : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.descriptor_store %arg2[%i, %c0], %out_blocked : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
      tt.store %result_ptrs, %old, %mask : tensor<128x256x!tt.ptr<f32>, #blocked1>
      scf.yield %dot : tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
