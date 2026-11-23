// RUN: triton-opt %s -split-input-file --nvgpu-warp-specialization=num-stages=3 | FileCheck %s

// When data partitioning fails for 3 warp groups (e.g. a 64x64 dot cannot be
// split across 2 consumer groups), stale async_task_id attributes must be
// cleared before retrying with 2 warp groups (1 producer + 1 consumer).
// CHECK-LABEL: @fallback_to_single_consumer_group
// CHECK: ttg.warp_specialize{{.*}}requestedRegisters = array<i32: 232>
// CHECK: default
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: partition0
// CHECK: ttng.warp_group_dot
// CHECK: tt.descriptor_store
// CHECK-NOT: partition1
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared_f32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fallback_to_single_consumer_group(%arg0: !tt.tensordesc<64x64xf8E4M3FN, #shared>, %arg1: !tt.tensordesc<64x64xf8E4M3FN, #shared>, %arg2: !tt.tensordesc<64x64xf32, #shared_f32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res = scf.for %iv = %c0 to %c3 step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %a = tt.descriptor_load %arg0[%c0, %iv] : !tt.tensordesc<64x64xf8E4M3FN, #shared> -> tensor<64x64xf8E4M3FN, #blocked>
      %b = tt.descriptor_load %arg1[%iv, %c0] : !tt.tensordesc<64x64xf8E4M3FN, #shared> -> tensor<64x64xf8E4M3FN, #blocked>
      %la = ttg.local_alloc %a : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
      %lb0 = ttg.local_alloc %b : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
      %lb = ttg.memdesc_trans %lb0 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<64x64xf8E4M3FN, #shared1, #smem>
      %dot = ttng.warp_group_dot %la, %lb, %acc {inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem> * !ttg.memdesc<64x64xf8E4M3FN, #shared1, #smem> -> tensor<64x64xf32, #mma>
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    %out = ttg.convert_layout %res : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
    tt.descriptor_store %arg2[%c0, %c0], %out : !tt.tensordesc<64x64xf32, #shared_f32>, tensor<64x64xf32, #blocked>
    tt.return
  }
}

// -----

// Hopper token lowering only supports TMA producer commits; skip warp
// specialization cleanly when producer loads are non-TMA tt.load ops so later
// software pipelining can handle the loop.
// CHECK-LABEL: @skip_non_tma_producer_load
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: scf.for
// CHECK: ttng.warp_group_dot
// CHECK: tt.num_stages = 3 : i32
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @skip_non_tma_producer_load(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %pa = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>
    %pb = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>
    %a = tt.load %pa : tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>
    %b = tt.load %pb : tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>
    %res = scf.for %iv = %c0 to %c3 step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %la = ttg.local_alloc %a : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
      %lb = ttg.local_alloc %b : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared1, #smem>
      %dot = ttng.warp_group_dot %la, %lb, %acc {inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem> * !ttg.memdesc<64x64xf8E4M3FN, #shared1, #smem> -> tensor<64x64xf32, #mma>
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    %out = ttg.convert_layout %res : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
    %po = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.store %po, %out : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A consumer-side epilogue tt.load (task 1) in a 2-warp-group TMA kernel must
// not be mistaken for a non-TMA producer load (task 0).
// CHECK-LABEL: @tma_loop_with_consumer_epilogue_load
// CHECK: ttg.warp_specialize{{.*}}requestedRegisters = array<i32: 232>
// CHECK: default
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: partition0
// CHECK: ttng.warp_group_dot
// CHECK: tt.load
// CHECK: arith.addf
// CHECK: tt.descriptor_store
// CHECK-NOT: partition1
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared_f32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_loop_with_consumer_epilogue_load(%arg0: !tt.tensordesc<64x64xf8E4M3FN, #shared>, %arg1: !tt.tensordesc<64x64xf8E4M3FN, #shared>, %arg2: !tt.tensordesc<64x64xf32, #shared_f32>, %bias_ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res = scf.for %iv = %c0 to %c3 step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %a = tt.descriptor_load %arg0[%c0, %iv] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x64xf8E4M3FN, #shared> -> tensor<64x64xf8E4M3FN, #blocked>
      %b = tt.descriptor_load %arg1[%iv, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x64xf8E4M3FN, #shared> -> tensor<64x64xf8E4M3FN, #blocked>
      %la = ttg.local_alloc %a : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
      %lb0 = ttg.local_alloc %b : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
      %lb = ttg.memdesc_trans %lb0 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<64x64xf8E4M3FN, #shared1, #smem>
      %dot = ttng.warp_group_dot %la, %lb, %acc {async_task_id = array<i32: 1>, inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem> * !ttg.memdesc<64x64xf8E4M3FN, #shared1, #smem> -> tensor<64x64xf32, #mma>
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    %out = ttg.convert_layout %res : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
    %p_bias = tt.splat %bias_ptr : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %bias = tt.load %p_bias {async_task_id = array<i32: 1>} : tensor<64x64x!tt.ptr<f32>, #blocked>
    %sum = arith.addf %out, %bias : tensor<64x64xf32, #blocked>
    tt.descriptor_store %arg2[%c0, %c0], %sum {async_task_id = array<i32: 1>} : !tt.tensordesc<64x64xf32, #shared_f32>, tensor<64x64xf32, #blocked>
    tt.return
  }
}
