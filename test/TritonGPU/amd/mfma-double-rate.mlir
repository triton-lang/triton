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

// Packed FP6 operands hold two 16-byte groups per lane (kWidth = 32), each with
// 12 bytes of codes and 4 padding bytes; the lowering keeps dwords 0-2 and 4-6.
// CHECK-LABEL: mfma_scaled_fp6_16x16x128
// CHECK: %[[B:.*]] = llvm.bitcast %{{.*}} : vector<32xi8> to vector<8xi32>
// CHECK-NOT: llvm.lshr
// CHECK-DAG: llvm.extractelement %[[B]][%{{.*}} : i32] : vector<8xi32>
// CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, fp6_e3m2, fp6_e2m3, {{.*}} : (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>
// CHECK-NOT: rocdl.mfma
#linear = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 128], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_scaled_fp6_16x16x128(%arg0: tensor<64x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 32}>>, %arg1: tensor<64x4xi8, #linear>, %arg2: tensor<128x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 32}>>, %arg3: tensor<16x4xi8, #linear1>, %out: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #mma>
    %d = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m3 rhs = e3m2 {fastMath = false} : tensor<64x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 32}>>, tensor<64x4xi8, #linear> * tensor<128x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 32}>>, tensor<16x4xi8, #linear1> -> tensor<64x16xf32, #mma>
    %p = tt.splat %out : !tt.ptr<f32> -> tensor<64x16x!tt.ptr<f32>, #mma>
    tt.store %p, %d : tensor<64x16x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: mfma_scaled_fp6_32x32x64
// CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}}, fp6_e2m3, fp6_e3m2, {{.*}} : (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>
// CHECK-NOT: rocdl.mfma
#linear = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 64], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_scaled_fp6_32x32x64(%arg0: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 32}>>, %arg1: tensor<32x2xi8, #linear>, %arg2: tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 32}>>, %arg3: tensor<32x2xi8, #linear>, %out: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %d = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e3m2 rhs = e2m3 {fastMath = false} : tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 32}>>, tensor<32x2xi8, #linear> * tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 32}>>, tensor<32x2xi8, #linear> -> tensor<32x32xf32, #mma>
    %p = tt.splat %out : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #mma>
    tt.store %p, %d : tensor<32x32x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Mixed FP8 x FP6: the FP8 operand stays eight dwords.
// CHECK-LABEL: mfma_scaled_fp8_fp6
// CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, fp6_e2m3, fp8_e4m3, {{.*}} : (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>
#linear = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 128], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_scaled_fp8_fp6(%arg0: tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<64x4xi8, #linear>, %arg2: tensor<128x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 32}>>, %arg3: tensor<16x4xi8, #linear1>, %out: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #mma>
    %d = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e4m3 rhs = e2m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<64x4xi8, #linear> * tensor<128x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 32}>>, tensor<16x4xi8, #linear1> -> tensor<64x16xf32, #mma>
    %p = tt.splat %out : !tt.ptr<f32> -> tensor<64x16x!tt.ptr<f32>, #mma>
    tt.store %p, %d : tensor<64x16x!tt.ptr<f32>, #mma>
    tt.return
  }
}
