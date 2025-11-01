// RUN: triton-opt %s -split-input-file -tritongpu-pipeline | FileCheck %s --check-prefixes=CHECK

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipeline_load_mmav3
  tt.func public @pipeline_load_mmav3(%arg0: tensor<256x128xf32, #mma>, %arg1: tensor<256x32x!tt.ptr<f32>, #blocked>, %arg2: tensor<32x128x!tt.ptr<f32>, #blocked1>, %arg3: tensor<256x32xi32, #blocked>, %arg4: tensor<32x128xi32, #blocked1>) -> (tensor<256x128xf32, #mma>, tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<32x128x!tt.ptr<f32>, #blocked1>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<4x256x32xf32
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<4x32x128xf32
    %0:3 = scf.for %arg5 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2) -> (tensor<256x128xf32, #mma>, tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<32x128x!tt.ptr<f32>, #blocked1>)  : i32 {
      // CHECK: ttg.memdesc_index {{.*}} : !ttg.memdesc<4x256x32xf32
      // CHECK: ttg.async_wait {{.*}} {num = 4 : i32}
      // CHECK: ttg.memdesc_index {{.*}} : !ttg.memdesc<4x32x128xf32
      // CHECK: ttng.warp_group_dot {{.*}} {inputPrecision = 0 : i32, isAsync = true}
      // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 1 : i32}
      %1 = tt.load %arg7 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<256x32x!tt.ptr<f32>, #blocked>
      %2 = ttg.local_alloc %1 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<256x32xf32, #blocked>) -> !ttg.memdesc<256x32xf32, #shared, #smem>
      %3 = tt.load %arg8 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<32x128x!tt.ptr<f32>, #blocked1>
      %4 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<32x128xf32, #blocked1>) -> !ttg.memdesc<32x128xf32, #shared1, #smem>
      %5 = ttng.warp_group_dot %2, %4, %arg6 {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<256x32xf32, #shared, #smem> * !ttg.memdesc<32x128xf32, #shared1, #smem> -> tensor<256x128xf32, #mma>
      %6 = tt.addptr %arg7, %arg3 {loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<256x32xi32, #blocked>
      %7 = tt.addptr %arg8, %arg4 {loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<32x128x!tt.ptr<f32>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %5, %6, %7 : tensor<256x128xf32, #mma>, tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<32x128x!tt.ptr<f32>, #blocked1>
    } {tt.num_stages = 4 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return %0#0, %0#1, %0#2 : tensor<256x128xf32, #mma>, tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<32x128x!tt.ptr<f32>, #blocked1>
  }
}

// -----

#s = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @expand_loop_without_results
  tt.func public @expand_loop_without_results() {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %true = arith.constant true
    %a = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #s, #ttg.shared_memory, mutable>
    %c = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    // CHECK: scf.for
    // CHECK:   ttng.tc_gen5_mma
    // CHECK:   ttng.wait_barrier
    scf.for %j = %c0 to %c16 step %c16 : i32 {
      ttng.tc_gen5_mma %a, %b, %c, %true, %true, %bar[%true] {is_async, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x64xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xbf16, #s, #ttg.shared_memory, mutable>, !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
      ttng.wait_barrier %bar, %c0 deps %a, %b {loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x64xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xbf16, #s, #ttg.shared_memory, mutable>
      scf.yield
    } {tt.num_stages = 4 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return
  }
}
