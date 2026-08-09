// RUN: triton-opt %s -split-input-file -tritongpu-reorder-instructions | FileCheck %s

// check that we don't hoist convert_layout above its operand definition.
// CHECK-LABEL: convert_cannot_hoist
//       CHECK:   %[[CVTS:.+]] = ttg.local_alloc
//       CHECK:   ttg.local_load %[[CVTS]]
//       CHECK:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @convert_cannot_hoist(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %10 = ttg.local_alloc %9 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %11 = ttg.local_load %10 : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %11, %cst_0, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: no_move_alloc_for_scalar_src
//       CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
//       CHECK: %[[SPLAT:.*]] = tt.splat %{{.*}} : f32 -> tensor<32x32xf32, #blocked>
//       CHECK: ttg.async_wait {num = 0 : i32}
//       CHECK: ttg.local_alloc %[[SPLAT]] : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_move_alloc_for_scalar_src() {
    %cst = arith.constant 0.000000e+00 : f32
    %t = tt.splat %cst : f32 -> tensor<32x32xf32, #blocked>
    ttg.async_wait {num = 0 : i32}
    %alloc = ttg.local_alloc %t : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_dealloc
//       CHECK: ttg.async_wait {num = 0 : i32}
//       CHECK: ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
//       CHECK: ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
//       CHECK: %3 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_dealloc(%arg0: tensor<32x32xf32, #blocked>) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %2 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %3 = arith.addf %2, %2 : tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_idx_1
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_idx_1(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD = ttg.local_load %AS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_idx_1_negative
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #{{.*}}, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
//       CHECK: ttng.arrive_barrier
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #{{.*}}, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_idx_1_negative(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    ttng.arrive_barrier %bar, 2, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %AD = ttg.local_load %AS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// check that we don't sink convert_layout if it has multi users
// CHECK-LABEL: convert_cannot_sink
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @convert_cannot_sink(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A0 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS0 = ttg.local_alloc %A0 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD0 = ttg.local_load %AS0 : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD0, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %A1 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS1 = ttg.local_alloc %A1 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD1 = ttg.local_load %AS1 : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %13 = tt.dot %AD1, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: do_not_sink_local_load_into_writing_loop
//       CHECK: %[[LOADED:.*]] = ttg.local_load
//       CHECK: scf.for
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @do_not_sink_local_load_into_writing_loop(
      %buffer: !ttg.memdesc<128xf32, #shared, #smem, mutable>,
      %value: tensor<128xf32, #blocked>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %loaded = ttg.local_load %buffer : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
    scf.for %i = %c0 to %c2 step %c1 {
      %updated = arith.addf %loaded, %value : tensor<128xf32, #blocked>
      ttg.local_store %updated, %buffer : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_local_load_into_loop_with_global_store
//       CHECK: scf.for
//       CHECK:   %[[LOADED:.*]] = ttg.local_load
//       CHECK:   tt.store
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_local_load_into_loop_with_global_store(
      %buffer: !ttg.memdesc<128xf32, #shared, #smem>,
      %ptrs: tensor<128x!tt.ptr<f32>, #blocked>,
      %value: tensor<128xf32, #blocked>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %loaded = ttg.local_load %buffer : !ttg.memdesc<128xf32, #shared, #smem> -> tensor<128xf32, #blocked>
    scf.for %i = %c0 to %c2 step %c1 {
      %sum = arith.addf %loaded, %value : tensor<128xf32, #blocked>
      tt.store %ptrs, %sum : tensor<128x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: do_not_reorder_local_load_across_dealloc
//       CHECK: %[[B:.*]] = ttg.local_load
//       CHECK: ttg.local_dealloc
//       CHECK: %[[A:.*]] = ttg.local_load
//       CHECK: tt.dot %[[A]], %[[B]]
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @do_not_reorder_local_load_across_dealloc(
      %a: !ttg.memdesc<32x32xf32, #shared, #smem>,
      %b: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = ttg.local_load %b : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    ttg.local_dealloc %b : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %A = ttg.local_load %a : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %result = tt.dot %A, %B, %zero, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: do_not_reorder_local_load_across_default_resource
//       CHECK: %[[B:.*]] = ttg.local_load
//       CHECK: tt.elementwise_inline_asm
//       CHECK: %[[A:.*]] = ttg.local_load
//       CHECK: tt.dot %[[A]], %[[B]]
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @do_not_reorder_local_load_across_default_resource(
      %a: !ttg.memdesc<32x32xf32, #shared, #smem>,
      %b: !ttg.memdesc<32x32xf32, #shared, #smem>,
      %value: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = ttg.local_load %b : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %asm = tt.elementwise_inline_asm "mov.b32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %value : i32 -> i32
    %A = ttg.local_load %a : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %result = tt.dot %A, %B, %zero, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: do_not_reorder_local_load_across_unknown_effect
//       CHECK: %[[B:.*]] = ttg.local_load
//       CHECK: tt.call @overwrite
//       CHECK: %[[A:.*]] = ttg.local_load
//       CHECK: tt.dot %[[A]], %[[B]]
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @overwrite(
      %buffer: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
      %value: tensor<32x32xf32, #blocked>) {
    ttg.local_store %value, %buffer : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }

  tt.func public @do_not_reorder_local_load_across_unknown_effect(
      %a: !ttg.memdesc<32x32xf32, #shared, #smem>,
      %b: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
      %value: tensor<32x32xf32, #blocked>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = ttg.local_load %b : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    tt.call @overwrite(%b, %value) : (!ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<32x32xf32, #blocked>) -> ()
    %A = ttg.local_load %a : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %result = tt.dot %A, %B, %zero, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
}
