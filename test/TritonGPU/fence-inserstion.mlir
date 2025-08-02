// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-fence-insertion | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_like_fence
  tt.func public @matmul_like_fence(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked2>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = ttg.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %1 = ttg.local_alloc %arg1 : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    // CHECK: ttng.fence_async_shared
    %2 = ttng.warp_group_dot %0, %1, %cst : !ttg.memdesc<128x128xf16, #shared, #smem> * !ttg.memdesc<128x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_like_fence_local_store
  tt.func public @matmul_like_fence_local_store(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked2>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    ttg.local_store %arg0, %0 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: ttng.fence_async_shared
    %2 = ttng.warp_group_dot %0, %1, %cst : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_like_fence_mma_v5
  tt.func public @matmul_like_fence_mma_v5(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked2>) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked1>
    %0 = ttg.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %1 = ttg.local_alloc %arg1 : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    %acc_tm = ttng.tmem_alloc %cst : (tensor<128x64xf32, #blocked1>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>
    // CHECK: ttng.fence_async_shared
    ttng.tc_gen5_mma %0, %1, %acc_tm, %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x64xf16, #shared1, #ttg.shared_memory>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fence_outside_loop
  tt.func public @fence_outside_loop(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %1 = ttg.local_alloc %arg1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    // CHECK: ttng.fence_async_shared
    // CHECK: scf.for
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK:   ttng.warp_group_dot
    scf.for %iv0 = %c0_i32 to %c64_i32 step %c32_i32 : i32 {
      scf.for %iv1 = %c0_i32 to %c64_i32 step %c32_i32 : i32 {
        %2 = ttng.warp_group_dot %0, %1, %cst : !ttg.memdesc<128x128xf16, #shared, #smem> * !ttg.memdesc<128x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
      }
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fence_store_in_loop
  tt.func public @fence_store_in_loop(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc %arg1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: scf.for
    // CHECK: ttng.fence_async_shared
    // CHECK: ttng.warp_group_dot
    scf.for %iv0 = %c0_i32 to %c64_i32 step %c32_i32 : i32 {
      scf.for %iv1 = %c0_i32 to %c64_i32 step %c32_i32 : i32 {
        ttg.local_store %arg0, %0 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %2 = ttng.warp_group_dot %0, %1, %cst : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
      }
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reg_argument
  tt.func public @reg_argument(%arg0: tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg1: tensor<128x64xf16, #blocked>) {
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.warp_group_dot
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %2 = ttng.warp_group_dot %arg0, %1, %cst : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = true>

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @mma_inside_warp_specialize
tt.func @mma_inside_warp_specialize(%src: tensor<64x64xf16, #blocked>) {
  %A = ttg.local_alloc %src : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
  %B = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %D = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>

  ttg.warp_specialize(%A, %B, %D)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0(%lhs: !ttg.memdesc<64x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: ttng.fence_async_shared
    // CHECK-NEXT: scf.for
    scf.for %i = %c0_i32 to %c32_i32 step %c1_i32 : i32 {
      // CHECK-NEXT: ttng.tc_gen5_mma
      ttng.tc_gen5_mma %lhs, %rhs, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: ttng.tc_gen5_mma
      ttng.tc_gen5_mma %lhs, %rhs, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    }
    ttg.warp_return
  }
  // CHECK: partition1
  partition1(%lhs: !ttg.memdesc<64x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
    // CHECK-NOT: ttng.fence_async_shared
    %true = arith.constant true
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %rhs, %rhs, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
  tt.return
}

}
