// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

// Check the condition when sched-2nd-load is applied
// The following tile sizes with single dot should apply
//
// Should apply: tile size 256x256x64 with single dot
// CHECK-LABEL: sink_2nd_load_256x256x64
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
//  CHECK-NEXT: triton_gpu.local_store %[[tileB]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x256x64(%A_ptr: tensor<256x64x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<64x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<256x64xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<64x256xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x64xf16, #dotOp0> * tensor<64x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      %4 = tt.load %A_ptr : tensor<256x64x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<64x256x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x64xf16, #blocked> -> !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256xf16, #blocked1> -> !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Should apply: tile size 256x256x128 with single dot
// CHECK-LABEL: sink_2nd_load_256x256x128
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
//  CHECK-NEXT: triton_gpu.local_store %[[tileB]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x256x128(%A_ptr: tensor<256x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<128x256xf16, #shared1, #triton_gpu.shared_memory, mutable>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<256x128xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<128x256xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<128x256xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x128xf16, #dotOp0> * tensor<128x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      %4 = tt.load %A_ptr : tensor<256x128x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<128x256x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x128xf16, #blocked> -> !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS {OpIdx = #amdgpu.OpIdx<1>} : tensor<128x256xf16, #blocked1> -> !tt.memdesc<128x256xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Should apply: tile size 256x128x128 with single dot
// CHECK-LABEL: sink_2nd_load_256x128x128
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
//  CHECK-NEXT: triton_gpu.local_store %[[tileB]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x128x128(%A_ptr: tensor<256x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x128x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x128xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<256x128xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<128x128xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<256x128xf32, #mma>
      %4 = tt.load %A_ptr : tensor<256x128x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x128xf16, #blocked> -> !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS {OpIdx = #amdgpu.OpIdx<1>} : tensor<128x128xf16, #blocked1> -> !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<256x128xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Should apply: tile size 256x128x64 with single dot
// CHECK-LABEL: sink_2nd_load_256x128x64
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
//  CHECK-NEXT: triton_gpu.local_store %[[tileB]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x128x64(%A_ptr: tensor<256x64x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<64x128x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x128x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x128xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<256x64xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<64x128xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x64xf16, #dotOp0> * tensor<64x128xf16, #dotOp1> -> tensor<256x128xf32, #mma>
      %4 = tt.load %A_ptr : tensor<256x64x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<64x128x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x64xf16, #blocked> -> !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x128xf16, #blocked1> -> !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<256x128xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Should apply: tile size 128x128x128 with single dot
// CHECK-LABEL: sink_2nd_load_128x128x128
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
//  CHECK-NEXT: triton_gpu.local_store %[[tileB]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_128x128x128(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<128x128x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<128x128xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x128xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<128x128xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #mma>
      %4 = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS {OpIdx = #amdgpu.OpIdx<0>} : tensor<128x128xf16, #blocked> -> !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS {OpIdx = #amdgpu.OpIdx<1>} : tensor<128x128xf16, #blocked1> -> !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<128x128xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<128x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Should apply: tile size 128x128x64 with single dot
// CHECK-LABEL: sink_2nd_load_128x128x64
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: triton_gpu.local_store %[[tileA]]
//  CHECK-NEXT: triton_gpu.local_store %[[tileB]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#dotOp0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_128x128x64(%A_ptr: tensor<128x64x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<64x128x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<128x128x!tt.ptr<f32>, #mma>, %A_LDS: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>, %B_LDS: !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<128x128xf32, #mma>)  : i32 {
      %1 = triton_gpu.local_load %A_LDS : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x64xf16, #dotOp0>
      %2 = triton_gpu.local_load %B_LDS : !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<64x128xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<128x64xf16, #dotOp0> * tensor<64x128xf16, #dotOp1> -> tensor<128x128xf32, #mma>
      %4 = tt.load %A_ptr : tensor<128x64x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<64x128x!tt.ptr<f16>, #blocked1>
      triton_gpu.local_store %4, %A_LDS {OpIdx = #amdgpu.OpIdx<0>} : tensor<128x64xf16, #blocked> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %5, %B_LDS {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x128xf16, #blocked1> -> !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %3 : tensor<128x128xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<128x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}
