// RUN: triton-opt %s  -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" | FileCheck %s

// CHECK-LABEL:mfma_16x16x32_f16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_f16(%arg0: tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // CHECK: rocdl.mfma.f32.16x16x32.f16 {{.*}} : (vector<8xf16>, vector<8xf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_16x16x32_bf16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_bf16(%arg0: tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // CHECK: rocdl.mfma.f32.16x16x32.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_32x32x16_f16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_f16(%arg0: tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK: rocdl.mfma.f32.32x32x16.f16 {{.*}} : (vector<8xf16>, vector<8xf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    tt.return
 }
}


// -----

// CHECK-LABEL:mfma_32x32x16_bf16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_bf16(%arg0: tensor<32x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<16x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK: rocdl.mfma.f32.32x32x16.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<32x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    tt.return
 }
}

// -----

// When kWidth is set to 4, still generate double rated mfma instructions.

// CHECK-LABEL:mfma_16x16x32_f16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_f16(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x128xf16, #dotOp1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK: rocdl.mfma.f32.16x16x32.f16 {{.*}} : (vector<8xf16>, vector<8xf16>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_16x16x32_bf16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_bf16(
      %q: tensor<128x128xbf16, #dotOp0>,
      %k: tensor<128x128xbf16, #dotOp1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK: rocdl.mfma.f32.16x16x32.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xbf16, #dotOp0> * tensor<128x128xbf16, #dotOp1> -> tensor<128x128xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_32x32x16_f16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_f16(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x128xf16, #dotOp1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK: rocdl.mfma.f32.32x32x16.f16 {{.*}} : (vector<8xf16>, vector<8xf16>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_32x32x16_bf16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_bf16(
      %q: tensor<128x128xbf16, #dotOp0>,
      %k: tensor<128x128xbf16, #dotOp1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK: rocdl.mfma.f32.32x32x16.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xbf16, #dotOp0> * tensor<128x128xbf16, #dotOp1> -> tensor<128x128xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mxfp4_2step
#linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 128], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mxfp4_2step(%arg0: tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<256x8xi8, #linear>, %arg2: tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<256x8xi8, #linear1>) {
    // CHECK-COUNT-32: rocdl.mfma.scale.f32.16x16x128.f8f6f4
    // CHECK: rocdl.sched.barrier none
    // CHECK: rocdl.s.barrier
    // CHECK: rocdl.sched.barrier none
    // CHECK-COUNT-32: rocdl.mfma.scale.f32.16x16x128.f8f6f4
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %dots = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {fastMath = false, pingpong_2step} : tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<256x256xf32, #mma>
    tt.return
 }
}

// -----

// With pingpong_2step and 2x2 tiles per warp, a scale word holds two M (or N)
// tiles t0/t1 at two K steps k0/k1. Its bytes are reordered from K-first
// [t0k0, t0k1, t1k0, t1k1] to [t0k0, t1k0, t0k1, t1k1], so K step 0 selects
// bytes 0/1 and K step 1 selects bytes 2/3, and M/N tiles are visited in 2x2
// groups. Without pingpong_2step, packing and traversal are unchanged.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 128], isTransposed = false, tilesPerWarp = [2, 2]}>
#sa = #ttg.linear<{register = [[0, 4], [16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [], block = []}>
#sb = #ttg.linear<{register = [[0, 4], [16, 0], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [], block = []}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // The MFMA checks match the trailing operands: opSelA, scaleA, opSelB, scaleB.
  // CHECK-LABEL: mxfp4_2step_scale_packing
  tt.func public @mxfp4_2step_scale_packing(%a: tensor<32x128xi8, #dotA>, %sa: tensor<32x8xi8, #sa>, %b: tensor<128x64xi8, #dotB>, %sb: tensor<64x8xi8, #sb>, %c: tensor<32x64xf32, #mma>) {
    // CHECK: %[[SA_BYTES:.*]] = llvm.shufflevector {{.*}} [0, 2, 1, 3] : vector<4xi8>
    // CHECK: %[[SA:.*]] = llvm.bitcast %[[SA_BYTES]] : vector<4xi8> to i32
    // CHECK: %[[SB0_BYTES:.*]] = llvm.shufflevector {{.*}} [0, 2, 1, 3] : vector<4xi8>
    // CHECK: %[[SB0:.*]] = llvm.bitcast %[[SB0_BYTES]] : vector<4xi8> to i32
    // CHECK: %[[SB1_BYTES:.*]] = llvm.shufflevector {{.*}} [0, 2, 1, 3] : vector<4xi8>
    // CHECK: %[[SB1:.*]] = llvm.bitcast %[[SB1_BYTES]] : vector<4xi8> to i32
    // K step 0: tiles (M0-1, N0-1), then (M0-1, N2-3).
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 0, %[[SA]], 0, %[[SB0]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 0, %[[SA]], 1, %[[SB0]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 1, %[[SA]], 0, %[[SB0]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 1, %[[SA]], 1, %[[SB0]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 0, %[[SA]], 0, %[[SB1]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 0, %[[SA]], 1, %[[SB1]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 1, %[[SA]], 0, %[[SB1]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 1, %[[SA]], 1, %[[SB1]] :
    // CHECK: rocdl.s.barrier
    // K step 1 uses bytes 2/3.
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 2, %[[SA]], 2, %[[SB0]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 2, %[[SA]], 3, %[[SB0]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 3, %[[SA]], 2, %[[SB0]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 3, %[[SA]], 3, %[[SB0]] :
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false, pingpong_2step} : tensor<32x128xi8, #dotA>, tensor<32x8xi8, #sa> * tensor<128x64xi8, #dotB>, tensor<64x8xi8, #sb> -> tensor<32x64xf32, #mma>
    tt.return
  }

  // CHECK-LABEL: mxfp4_scale_packing_without_2step
  tt.func public @mxfp4_scale_packing_without_2step(%a: tensor<32x128xi8, #dotA>, %sa: tensor<32x8xi8, #sa>, %b: tensor<128x64xi8, #dotB>, %sb: tensor<64x8xi8, #sb>, %c: tensor<32x64xf32, #mma>) {
    // CHECK-NOT: llvm.shufflevector
    // K-first: both K steps of tile (M0, N0) select bytes 0 and 1.
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 0, %[[SA:[^,]+]], 0, %[[SB0:[^ ]+]] :
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 1, %[[SA]], 1, %[[SB0]] :
    // Linear M/N order: (M0, N2) is reached before any M1 tile (opSelA 2/3).
    // CHECK-NOT: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, {{[23]}}, %[[SA]],
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, 0, %[[SA]], 0, %{{[^ ]+}} :
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x128xi8, #dotA>, tensor<32x8xi8, #sa> * tensor<128x64xi8, #dotB>, tensor<64x8xi8, #sb> -> tensor<32x64xf32, #mma>
    tt.return
  }
}
