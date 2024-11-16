// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

// Check the logic of sched-2nd-load optimizations
//

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// Category 1: Single dot with two loads, we make sure the optimization is applied when tile size is large enough
// The following tile sizes should apply the optimization
// 256x256x128
// 256x256x64
// The following tile sizes should NOT apply the optimization
// 256x64x128
// 256x256x32
//

// Should apply: tile size 256x256x128 with single dot
// CHECK-LABEL: sink_2nd_load_256x256x128
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
//  CHECK-NEXT: triton_gpu.local_store %[[tileB]]
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x256x128(%A_ptr: tensor<256x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<128x256xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<256x128xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<128x256xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<128x256xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x128xf16, #dotOp0> * tensor<128x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      %4 = tt.load %A_ptr : tensor<256x128x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<128x256x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS : tensor<256x128xf16, #blocked> -> !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS : tensor<128x256xf16, #blocked1> -> !tt.memdesc<128x256xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// Should apply: tile size 256x256x64 with single dot
// CHECK-LABEL: sink_2nd_load_256x256x64
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
//  CHECK-NEXT: triton_gpu.local_store %[[tileB]]
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x256x64(%A_ptr: tensor<256x64x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<64x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<256x64xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<64x256xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x64xf16, #dotOp0> * tensor<64x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      %4 = tt.load %A_ptr : tensor<256x64x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<64x256x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS : tensor<256x64xf16, #blocked> -> !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS : tensor<64x256xf16, #blocked1> -> !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// Should NOT apply: tile size 256x64x128 with single dot
// CHECK-LABEL: sink_2nd_load_256x64x128
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
//  CHECK-NEXT: triton_gpu.local_store %[[tileB]]
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x64x128(%A_ptr: tensor<256x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x64x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x64x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<128x64xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x64xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<256x128xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<128x64xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<128x64xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x128xf16, #dotOp0> * tensor<128x64xf16, #dotOp1> -> tensor<256x64xf32, #mma>
      %4 = tt.load %A_ptr : tensor<256x128x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<128x64x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS : tensor<256x128xf16, #blocked> -> !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS : tensor<128x64xf16, #blocked1> -> !tt.memdesc<128x64xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<256x64xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// Should NOT apply: tile size 256x256x32 with single dot
// CHECK-LABEL: sink_2nd_load_256x256x32
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
//  CHECK-NEXT: triton_gpu.local_store %[[tileB]]
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x256x32(%A_ptr: tensor<256x32x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<32x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<32x256xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<256x32xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<32x256xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<32x256xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      %4 = tt.load %A_ptr : tensor<256x32x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<32x256x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS : tensor<256x32xf16, #blocked> -> !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS : tensor<32x256xf16, #blocked1> -> !tt.memdesc<32x256xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// Category 2: single dot with two loads and tile size is large enough (128x128x128).
//             We make sure the move is legal.
// Should NOT apply: the 2nd load has a user before the dot
// CHECK-LABEL: sink_2nd_load_128x128x128_user_before_dot
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: tt.store
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_128x128x128_user_before_dot(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x128x!tt.ptr<i64>, #blocked>, %B_ptr2: tensor<128x128x!tt.ptr<f16>, #blocked>, %C_ptr: tensor<128x128x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<128x128xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x128xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<128x128xf16, #dotOp1>
      %4 = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<128x128x!tt.ptr<i64>, #blocked>
      tt.store %B_ptr, %5 : tensor<128x128x!tt.ptr<i64>, #blocked>
      %3 = tt.dot %1, %2, %arg1 : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #mma>
      triton_gpu.local_store %4, %A_LDS : tensor<128x128xf16, #blocked> -> !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<128x128xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<128x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}


// -----

// Category 3: two dots in the for loop. Make sure the optimization is not applied
// should NOT apply: two dots
// CHECK-LABEL: sink_2nd_load_256x256x64_two_dot
//       CHECK: triton_gpu.local_load
//  CHECK-NEXT: triton_gpu.local_load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: tt.load
//  CHECK-NEXT: tt.load
//  CHECK-NEXT: triton_gpu.local_store
//  CHECK-NEXT: triton_gpu.local_store
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x256x64_two_dot(%A_ptr: tensor<256x64x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<64x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<256x64xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<64x256xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x64xf16, #dotOp0> * tensor<64x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      %6 = tt.dot %1, %2, %3 : tensor<256x64xf16, #dotOp0> * tensor<64x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      %4 = tt.load %A_ptr : tensor<256x64x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<64x256x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS : tensor<256x64xf16, #blocked> -> !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS : tensor<64x256xf16, #blocked1> -> !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}
