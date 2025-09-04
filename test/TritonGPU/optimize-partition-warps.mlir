// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-optimize-partition-warps | FileCheck %s

#blocked8 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4_broadcast = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2d_4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked2d_8 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked2d_16 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 4], order = [0, 1]}>
#blocked_tmem = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 2], order = [0, 1]}>
#shared_1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
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
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>
  ttg.warp_specialize(%arg0, %alloc)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(1)
  partition0(%arg1: i32, %arg2: !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>) num_warps(8) {
    %0 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked8>
    ttg.local_store %0, %arg2 : tensor<128xi32, #blocked8> -> !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(1)
  partition1(%arg1: i32, %arg2: !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>) num_warps(4) {
    %0 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked4>
    %1 = ttg.convert_layout %0 : tensor<128xi32, #blocked4> -> tensor<128xi32, #blocked4_broadcast>
    ttg.local_store %1, %arg2 : tensor<128xi32, #blocked4_broadcast> -> !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>
    ttg.warp_return
  } : (i32, !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>) -> ()
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

// CHECK-LABEL: @register_use_heuristic
tt.func @register_use_heuristic() {
  // CHECK: requestedRegisters = array<i32: 24, 88>
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(4) {
    %cst = arith.constant dense<0> : tensor<128x64xi32, #blocked2d_4>
    ttg.warp_return
  } : () -> ()
  tt.return
}

// CHECK-LABEL: @tmem_min_4_warps
tt.func @tmem_min_4_warps(%tensor_desc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
  ttg.warp_specialize(%tensor_desc)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0{{.*}} num_warps(4)
  partition0(%desc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(8) {
    %result = ttng.tmem_load %desc : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked_tmem>
    "use"(%result) : (tensor<64x64xf32, #blocked_tmem>) -> ()
    ttg.warp_return
  }
  // CHECK: partition1{{.*}} num_warps(4)
  partition1(%desc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(8) {
    %cst = arith.constant dense<0.0> : tensor<64x64xf32, #blocked_tmem>
    %true = arith.constant true
    ttng.tmem_store %cst, %desc, %true : tensor<64x64xf32, #blocked_tmem> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_return
  }
  // CHECK: partition2{{.*}} num_warps(4)
  partition2(%desc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(8) {
    %cst = arith.constant dense<0.0> : tensor<64x64xf32, #blocked_tmem>
    %result = ttng.tmem_alloc %cst : (tensor<64x64xf32, #blocked_tmem>) -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory>
    "use"(%result) : (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory>) -> ()
    ttg.warp_return
  } : (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
  tt.return
}

}
