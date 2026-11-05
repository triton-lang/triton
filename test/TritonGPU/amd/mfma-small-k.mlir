// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" | FileCheck %s

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_small_k_lanes
  tt.func public @mfma_small_k_lanes(%a: tensor<32x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                                     %b: tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
                                     %c: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma> {
    // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-NEXT: %[[KLANE:.*]] = llvm.and %{{.*}}, %[[C32]] : i32
    // CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[VALID:.*]] = llvm.icmp "eq" %[[KLANE]], %[[C0]] : i32
    // CHECK: llvm.select %[[VALID]], %{{.*}}, %{{.*}} : i1, vector<8xf16>
    // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, vector<8xf16>
    // CHECK: %[[D:.*]] = rocdl.mfma.f32.32x32x16.f16
    // CHECK-NOT: rocdl.mfma
    // CHECK-NOT: llvm.fmul
    // CHECK: llvm.extractelement %[[D]]
    %d = tt.dot %a, %b, %c : tensor<32x8xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    tt.return %d : tensor<32x32xf32, #mma>
  }
}

// -----

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_small_k_kpack
  tt.func public @mfma_small_k_kpack(%a: tensor<16x4xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                                     %b: tensor<4x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
                                     %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma> {
    // CHECK-COUNT-2: rocdl.mfma.f32.16x16x16f16
    // CHECK-NOT: rocdl.mfma
    // CHECK-NOT: llvm.fmul
    %d = tt.dot %a, %b, %c : tensor<16x4xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<4x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return %d : tensor<16x16xf32, #mma>
  }
}

// -----

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_small_k_registers
  tt.func public @mfma_small_k_registers(%a: tensor<32x4xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>,
                                         %b: tensor<4x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>,
                                         %c: tensor<32x32xi32, #mma>) -> tensor<32x32xi32, #mma> {
    // CHECK: %[[Z:.*]] = llvm.mlir.zero : i8
    // CHECK: llvm.insertelement %[[Z]], %{{.*}}[%{{.*}} : i32] : vector<16xi8>
    // CHECK: rocdl.mfma.i32.32x32x32.i8
    // CHECK-NOT: rocdl.mfma
    // CHECK-NOT: llvm.ashr
    %d = tt.dot %a, %b, %c : tensor<32x4xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<4x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<32x32xi32, #mma>
    tt.return %d : tensor<32x32xi32, #mma>
  }
}
