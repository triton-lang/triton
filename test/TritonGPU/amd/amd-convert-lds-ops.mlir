// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1250" | FileCheck %s

// CHECK-LABEL: memdesc_reinterpret_0

// This testing is just to make sure MemDescReinterpretOp::verify do not
// trigger assertion when it comes across padded-shared layout. No need to
// inspect resulting IR.

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
//#shared = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [16, 128]}>
#shared = #ttg.padded_shared<[128:+8,256:+4] {order = [1, 0], shape = [16, 128]}>
#shared2 = #ttg.padded_shared<[128:+8,256:+4] {order = [1, 0], shape = [16, 128]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 17376 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @memdesc_reinterpret_0() {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xbf16, #mma>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable>
    %1 = ttg.memdesc_reinterpret %0 : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<2x16x128xf16, #shared2, #smem, mutable>
    tt.return
  }
}
