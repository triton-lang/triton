// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory | FileCheck %s


#blocked1 = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

// This test checks swizzling based converter.
//
// Swizzling converter tries to find swizzling pattern, which provides widest load and store instructions and avoids as much back conflicts as possible.
// Current converter implementation decides that best swizzling patter requires allocation of tile with shape [256, 128], which takes 256*128*4(size of one element) = 131072 bytes
//
// For implementation see mlir::triton::getNumScratchElemsSwizzledCvt function,
// in particular mlir::triton::gpu::optimalSwizzling to get shape of repeat tile.

// CHECK: ttg.shared = 131072 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: @convert_layout_swizzled
tt.func @convert_layout_swizzled(%arg0: tensor<256x256xi32, #blocked1>) {
  // CHECK-NEXT: allocation.offset = 0 : i32
  %0 = ttg.convert_layout %arg0 : tensor<256x256xi32, #blocked1> -> tensor<256x256xi32, #blocked2>
  tt.return
}

}

// -----

// WarpSpecialize with a 2D TensorDesc capture.
// 2D TensorDesc = 12 dwords = 48 bytes.
// Capture size + Warp state buffer size = 48 + 4 = 52 bytes.
// CHECK: ttg.shared = 52 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {

// CHECK-LABEL: @ws_tensordesc_2d_capture
// CHECK: allocation.offset = 48 : i32
tt.func @ws_tensordesc_2d_capture(%desc: !tt.tensordesc<tensor<64x64xf16>>) {
  ttg.warp_specialize(%desc) attributes {warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: !tt.tensordesc<tensor<64x64xf16>>) num_warps(4) {
    ttg.warp_return
  } : (!tt.tensordesc<tensor<64x64xf16>>) -> ()
  tt.return
}

}

// -----

// WarpSpecialize with a 5D TensorDesc capture.
// 5D TensorDesc = 20 dwords = 80 bytes.
// Capture size + Warp state buffer size = 80 + 4 = 84 bytes.
// CHECK: ttg.shared = 84 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {

// CHECK-LABEL: @ws_tensordesc_5d_capture
// CHECK: allocation.offset = 80 : i32
tt.func @ws_tensordesc_5d_capture(%desc: !tt.tensordesc<tensor<8x8x8x16x16xf16>>) {
  ttg.warp_specialize(%desc) attributes {warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: !tt.tensordesc<tensor<8x8x8x16x16xf16>>) num_warps(4) {
    ttg.warp_return
  } : (!tt.tensordesc<tensor<8x8x8x16x16xf16>>) -> ()
  tt.return
}

}

// -----

// Test that allocation.scope and allocation.scope.noalias attrs are set on ops
// with two non-overlapping LDS buffers.

#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared3 = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [0, 1]}>
#smem3 = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: @two_lds_buffers_alias_scope
tt.func @two_lds_buffers_alias_scope(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
  %cst_a = arith.constant dense<0.0> : tensor<64x1xf16, #blocked3>
  %cst_b = arith.constant dense<0.0> : tensor<64x1xf16, #blocked3>

  // CHECK: ttg.local_alloc
  // CHECK-SAME: allocation.scope = 0 : i32
  // CHECK-SAME: allocation.scope.noalias = [1 : i32]
  %alloc_a = ttg.local_alloc %cst_a : (tensor<64x1xf16, #blocked3>) -> !ttg.memdesc<64x1xf16, #shared3, #smem3>

  // CHECK: ttg.local_alloc
  // CHECK-SAME: allocation.scope = 1 : i32
  // CHECK-SAME: allocation.scope.noalias = [0 : i32]
  %alloc_b = ttg.local_alloc %cst_b : (tensor<64x1xf16, #blocked3>) -> !ttg.memdesc<64x1xf16, #shared3, #smem3>

  // CHECK: ttg.local_load
  // CHECK-SAME: allocation.scope = 0 : i32
  // CHECK-SAME: allocation.scope.noalias = [1 : i32]
  %load_a = ttg.local_load %alloc_a : !ttg.memdesc<64x1xf16, #shared3, #smem3> -> tensor<64x1xf16, #blocked3>

  // CHECK: ttg.local_load
  // CHECK-SAME: allocation.scope = 1 : i32
  // CHECK-SAME: allocation.scope.noalias = [0 : i32]
  %load_b = ttg.local_load %alloc_b : !ttg.memdesc<64x1xf16, #shared3, #smem3> -> tensor<64x1xf16, #blocked3>

  // Stores to keep the local_loads alive
  %ptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked3>
  tt.store %ptr, %load_a : tensor<64x1x!tt.ptr<f16>, #blocked3>
  tt.store %ptr, %load_b : tensor<64x1x!tt.ptr<f16>, #blocked3>
  tt.return
}

}
