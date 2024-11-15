// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

// Check that we place local_alloc, local_store (optional) and local_load right after definition of their operands
// in cases where local_alloc is in the loop but it's operand is not.
// This is useful for making sure that Q tensor in FA is hoisted out of the main loop and kept in registers
// throughout the computation.

// CHECK-LABEL: hoist_q_out_of_the_loop
//       CHECK: %[[TRUNCF:.+]] = arith.truncf
//       CHECK-NEXT:  %[[ALLOC:.+]] = triton_gpu.local_alloc %[[TRUNCF]]
//       CHECK-NEXT:  triton_gpu.local_load %[[ALLOC]]
//       CHECK: scf.for
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], hasLeadingOffset = false}>
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @hoist_q_out_of_the_loop(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mfma>
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg7 : i32
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f16>, i32
    %12 = tt.splat %3 : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked1>
    %41 = tt.load %12 : tensor<256x128x!tt.ptr<f16>, #blocked1>
    %42 = arith.extf %41 : tensor<256x128xf16, #blocked1> to tensor<256x128xf32, #blocked1>
    %43 = tt.splat %cst : f32 -> tensor<256x128xf32, #blocked1>
    %44 = arith.mulf %42, %43 : tensor<256x128xf32, #blocked1>
    %45 = arith.truncf %44 : tensor<256x128xf32, #blocked1> to tensor<256x128xf16, #blocked1>
    %54:1 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg26 = %c0_i64) -> (i64)  : i32 {
      %73 = tt.splat %3 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
      %74 = tt.load %73 : tensor<128x128x!tt.ptr<f16>, #blocked2>
      %75 = triton_gpu.local_alloc %45 : (tensor<256x128xf16, #blocked1>) -> !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory>
      %76 = triton_gpu.local_load %75 : !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %77 = triton_gpu.local_alloc %74 : (tensor<128x128xf16, #blocked2>) -> !tt.memdesc<128x128xf16, #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>, #triton_gpu.shared_memory>
      %78 = triton_gpu.local_load %77 : !tt.memdesc<128x128xf16, #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>, #triton_gpu.shared_memory> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %79 = tt.dot %76, %78, %cst_2 : tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %107 = arith.addi %arg26, %c128_i64 : i64
      scf.yield %107 : i64
    } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
    tt.return
  }
}


// -----
// Check that reordering described in hoist_q_out_of_the_loop is not done in the case where both
// local_alloc and it's src tensor defining op are in the loop.
// CHECK-LABEL: no_hoist_q_type_reordering
//       CHECK: scf.for
//       CHECK: %[[TRUNCF:.+]] = arith.truncf
//       CHECK-NEXT:  arith.constant
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], hasLeadingOffset = false}>
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @no_hoist_q_type_reordering(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg7 : i32
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f16>, i32
    %12 = tt.splat %3 : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked1>
    %41 = tt.load %12 : tensor<256x128x!tt.ptr<f16>, #blocked1>
    %42 = arith.extf %41 : tensor<256x128xf16, #blocked1> to tensor<256x128xf32, #blocked1>
    %43 = tt.splat %cst : f32 -> tensor<256x128xf32, #blocked1>
    %44 = arith.mulf %42, %43 : tensor<256x128xf32, #blocked1>
    %54:1 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg26 = %c0_i64) -> (i64)  : i32 {
      %45 = arith.truncf %44 : tensor<256x128xf32, #blocked1> to tensor<256x128xf16, #blocked1>
      %cst_2 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mfma>
      %73 = tt.splat %3 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
      %74 = tt.load %73 : tensor<128x128x!tt.ptr<f16>, #blocked2>
      %75 = triton_gpu.local_alloc %45 : (tensor<256x128xf16, #blocked1>) -> !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory>
      %76 = triton_gpu.local_load %75 : !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %77 = triton_gpu.local_alloc %74 : (tensor<128x128xf16, #blocked2>) -> !tt.memdesc<128x128xf16, #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>, #triton_gpu.shared_memory>
      %78 = triton_gpu.local_load %77 : !tt.memdesc<128x128xf16, #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>, #triton_gpu.shared_memory> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %79 = tt.dot %76, %78, %cst_2 : tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %107 = arith.addi %arg26, %c128_i64 : i64
      scf.yield %107 : i64
    } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>

// CHECK-LABEL: order_load_alloc_local_load_local_store
//       CHECK:   %[[LOAD:.+]] = tt.load
//       CHECK:   %[[ALLOC:.+]] = triton_gpu.local_alloc
//       CHECK:   triton_gpu.local_store %[[LOAD]], %[[ALLOC]]
//       CHECK:   triton_gpu.local_load %[[ALLOC]]
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @order_load_alloc_local_load_local_store(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %10 = triton_gpu.local_alloc : () -> !tt.memdesc<32x32xf32, #shared, mutable>
    triton_gpu.local_store %9, %10 : tensor<32x32xf32, #blocked> -> !tt.memdesc<32x32xf32, #shared, mutable>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %11 = triton_gpu.local_load %10 : !tt.memdesc<32x32xf32, #shared, mutable> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %12 = tt.dot %11, %cst_0, %cst : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %13 = triton_gpu.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

//   CHECK-LABEL: sink_convert_dealloc
// CHECK-COUNT-2: triton_gpu.local_dealloc %{{.+}} : !tt.memdesc<4x128x64xf16, #shared, mutable>
//         CHECK: triton_gpu.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_dealloc(%arg0: tensor<32x32xf32, #blocked>) attributes {noinline = false} {
    %0 = triton_gpu.local_alloc : () -> !tt.memdesc<4x128x64xf16, #shared, mutable>
    %1 = triton_gpu.local_alloc : () -> !tt.memdesc<4x128x64xf16, #shared, mutable>
    %2 = triton_gpu.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    triton_gpu.local_dealloc %0 : !tt.memdesc<4x128x64xf16, #shared, mutable>
    triton_gpu.local_dealloc %1 : !tt.memdesc<4x128x64xf16, #shared, mutable>
    %3 = arith.addf %2, %2 : tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

//   CHECK-LABEL: anchor_barrier
//         CHECK: gpu.barrier
//         CHECK: tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @anchor_barrier(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked>) attributes {noinline = false} {
    %0 = triton_gpu.local_alloc : () -> !tt.memdesc<4x128x64xf16, #shared, mutable>
    gpu.barrier
    %2 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %1 = triton_gpu.local_alloc %2 : (tensor<32x32xf16, #blocked>) -> !tt.memdesc<4x128x64xf16, #shared, mutable>
    triton_gpu.local_dealloc %0 : !tt.memdesc<4x128x64xf16, #shared, mutable>
    triton_gpu.local_dealloc %1 : !tt.memdesc<4x128x64xf16, #shared, mutable>
    tt.return
  }
}
