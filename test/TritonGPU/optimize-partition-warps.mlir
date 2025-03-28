// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-optimize-partition-warps | FileCheck %s

#blocked8 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4_broadcast = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2d_4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked2d_8 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 8 : i32} {

// CHECK-LABEL: @no_tensor_computations
tt.func @no_tensor_computations(%arg0: i32) {
  ttg.warp_specialize(%arg0)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(1)
  partition0(%arg1: i32) num_warps(8) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(1)
  partition1(%arg1: i32) num_warps(4) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// CHECK-LABEL: @small_tensor_computation
tt.func @small_tensor_computation(%arg0: i32) {
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
  ttg.warp_specialize(%arg0, %alloc)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(1)
  partition0(%arg1: i32, %arg2: !ttg.memdesc<128xi32, #shared, #smem, mutable>) num_warps(8) {
    %0 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked8>
    ttg.local_store %0, %arg2 : tensor<128xi32, #blocked8> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(1)
  partition1(%arg1: i32, %arg2: !ttg.memdesc<128xi32, #shared, #smem, mutable>) num_warps(4) {
    %0 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked4>
    %1 = ttg.convert_layout %0 : tensor<128xi32, #blocked4> -> tensor<128xi32, #blocked4_broadcast>
    ttg.local_store %1, %arg2 : tensor<128xi32, #blocked4_broadcast> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    ttg.warp_return
  } : (i32, !ttg.memdesc<128xi32, #shared, #smem, mutable>) -> ()
  tt.return
}

// CHECK-LABEL: @large_tensor_computation
tt.func @large_tensor_computation(%arg0: i32) {
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
  ttg.warp_specialize(%arg0, %alloc)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(8)
  partition0(%arg1: i32, %arg2: !ttg.memdesc<128x256xf16, #shared, #smem, mutable>) num_warps(8) {
    %0 = ttg.local_load %arg2 : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> tensor<128x256xf16, #blocked2d_8>
    %1 = arith.extf %0 : tensor<128x256xf16, #blocked2d_8> to tensor<128x256xf32, #blocked2d_8>
    %2 = arith.addf %1, %1 : tensor<128x256xf32, #blocked2d_8>
    %3 = arith.truncf %2 : tensor<128x256xf32, #blocked2d_8> to tensor<128x256xf16, #blocked2d_8>
    ttg.local_store %3, %arg2 : tensor<128x256xf16, #blocked2d_8> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
    ttg.warp_return
  } : (i32, !ttg.memdesc<128x256xf16, #shared, #smem, mutable>) -> ()
  tt.return
}

// CHECK-LABEL: @medium_tensor_computation
tt.func @medium_tensor_computation(%arg0: i32) {
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  ttg.warp_specialize(%arg0, %alloc)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(4)
  partition0(%arg1: i32, %arg2: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) num_warps(8) {
    %0 = ttg.local_load %arg2 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked2d_8>
    %1 = arith.extf %0 : tensor<128x64xf16, #blocked2d_8> to tensor<128x64xf32, #blocked2d_8>
    %2 = arith.addf %1, %1 : tensor<128x64xf32, #blocked2d_8>
    %3 = arith.truncf %2 : tensor<128x64xf32, #blocked2d_8> to tensor<128x64xf16, #blocked2d_8>
    ttg.local_store %3, %arg2 : tensor<128x64xf16, #blocked2d_8> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttg.warp_return
  } : (i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) -> ()
  tt.return
}

// CHECK-LABEL: @fits_after_shrink
tt.func @fits_after_shrink(%arg0: i32) {
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  ttg.warp_specialize(%arg0, %alloc)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(4)
  partition0(%arg1: i32, %arg2: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) num_warps(8) {
    %0 = ttg.local_load %arg2 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked2d_8>
    %1 = arith.extf %0 : tensor<128x64xf16, #blocked2d_8> to tensor<128x64xf32, #blocked2d_8>
    %2 = arith.addf %1, %1 : tensor<128x64xf32, #blocked2d_8>
    %3 = arith.truncf %2 : tensor<128x64xf32, #blocked2d_8> to tensor<128x64xf16, #blocked2d_8>
    ttg.local_store %3, %arg2 : tensor<128x64xf16, #blocked2d_8> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(1)
  partition1(%arg1: i32, %arg2: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) num_warps(8) {
    ttg.warp_return
  } : (i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) -> ()
  tt.return
}

}
