// RUN: triton-opt %s -split-input-file -tritongpu-pipeline | FileCheck %s --check-prefixes=CHECK,STAGES-ZERO,STAGES-ONE,WARP-SCHEDULE
// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=1 | FileCheck %s --check-prefix=DEFAULT-ONE

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // STAGES-ZERO-LABEL: @num_stages_zero
  // STAGES-ZERO-NEXT:    %[[RESULT:.*]] = scf.for
  // STAGES-ZERO-NEXT:      %[[FIRST:.*]] = arith.addi
  // STAGES-ZERO-NEXT:      %[[SECOND:.*]] = arith.addi
  // STAGES-ZERO-NEXT:      scf.yield %[[SECOND]]
  // STAGES-ZERO-NEXT:    } {tt.num_stages = 0 : i32}
  tt.func public @num_stages_zero(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32 {
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> i32 : i32 {
      %first = arith.addi %iv, %acc {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %second = arith.addi %first, %acc {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      scf.yield %second : i32
    } {tt.num_stages = 0 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return %result : i32
  }

  // STAGES-ONE-LABEL: @num_stages_one
  // STAGES-ONE-NEXT:    %[[RESULT:.*]] = scf.for
  // STAGES-ONE-NEXT:      %[[FIRST:.*]] = arith.addi
  // STAGES-ONE-NEXT:      %[[SECOND:.*]] = arith.addi
  // STAGES-ONE-NEXT:      scf.yield %[[SECOND]]
  // STAGES-ONE-NEXT:    } {tt.num_stages = 1 : i32}
  tt.func public @num_stages_one(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32 {
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> i32 : i32 {
      %first = arith.addi %iv, %acc {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %second = arith.addi %first, %acc {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      scf.yield %second : i32
    } {tt.num_stages = 1 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return %result : i32
  }

  // DEFAULT-ONE-LABEL: @default_num_stages_one
  // DEFAULT-ONE-NEXT:    %[[RESULT:.*]] = scf.for
  // DEFAULT-ONE-NEXT:      %[[FIRST:.*]] = arith.addi
  // DEFAULT-ONE-NEXT:      %[[SECOND:.*]] = arith.addi
  // DEFAULT-ONE-NEXT:      scf.yield %[[SECOND]]
  // DEFAULT-ONE-NEXT:    }
  tt.func public @default_num_stages_one(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32 {
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> i32 : i32 {
      %first = arith.addi %iv, %acc {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %second = arith.addi %first, %acc {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      scf.yield %second : i32
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %result : i32
  }

  // A loop produced inside warp specialization has an internal schedule whose
  // stages are independent of the source loop's num_stages setting.
  // WARP-SCHEDULE-LABEL: @warp_specialized_schedule
  // WARP-SCHEDULE:          ttg.warp_specialize(
  // WARP-SCHEDULE:          default {
  // WARP-SCHEDULE-NEXT:       %[[PROLOGUE:.*]] = arith.addi
  // WARP-SCHEDULE-NEXT:       %{{.*}}:2 = scf.for {{.*}} iter_args({{.*}}, {{.*}}) -> (i32, i32)
  tt.func public @warp_specialized_schedule(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32 {
    %ws_result = ttg.warp_specialize(%init)
    default {
      %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> i32 : i32 {
        %first = arith.addi %iv, %acc {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        %second = arith.addi %first, %acc {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
        scf.yield %second : i32
      } {tt.num_stages = 1 : i32, tt.scheduled_max_stage = 1 : i32}
      ttg.warp_yield %result : i32
    } : (i32) -> i32
    tt.return %ws_result : i32
  }
}

// -----

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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @nested_loop_gen5_mma
  tt.func public @nested_loop_gen5_mma(%arg0: !tt.ptr<bf16>, %arg1: i1) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024x64xf32, #blocked>
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
    %1 = tt.load %0 : tensor<64x64x!tt.ptr<bf16>, #blocked>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %3 = ttg.local_alloc %1 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<1024x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %4 = ttng.tmem_store %cst, %result[%token], %true : tensor<1024x64xf32, #blocked> -> !ttg.memdesc<1024x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0 = ttng.tmem_alloc {loop.cluster = 0 : i32, loop.stage = 0 : i32} : () -> !ttg.memdesc<1024x64xbf16, #tmem, #ttng.tensor_memory, mutable>
    scf.for %arg2 = %c0_i32 to %c32_i32 step %c16_i32  : i32 {
      // In order for both the outer and inner loop to be pipelined, the inner
      // loop cannot be directly nested in the outer loop, so add an if in the
      // middle.
      scf.if %arg1 {
        %5 = scf.for %arg3 = %c0_i32 to %arg2 step %c16_i32 iter_args(%arg4 = %4) -> (!ttg.async.token)  : i32 {
          %6 = ttng.tc_gen5_mma %result_0, %3, %result[%arg4], %false, %true, %2[%true] {is_async, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1024x64xbf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xbf16, #shared1, #smem>, !ttg.memdesc<1024x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.wait_barrier %2, %c0_i32 deps %result_0, %3 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1024x64xbf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xbf16, #shared1, #smem>
          scf.yield %6 : !ttg.async.token
        } {tt.num_stages = 4 : i32, tt.scheduled_max_stage = 1 : i32}
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32}
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return
  }
}
