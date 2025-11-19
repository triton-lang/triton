// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1250 | FileCheck %s

// CGA layout has no broadcasting so we should not emit cluster loads
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0], [4, 0]]}>
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: load_multi_cta_but_no_broadcast
  tt.func public @load_multi_cta_but_no_broadcast(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
    // CHECK-NOT: llvm.amdgcn.cluster.load.b128
    %6 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// 8 CTAs, 2 multicast groups of 4 CTAs each. Each group is strided by 1 so the base mask should be 0b1010101 (85) and the non free mask is -7 (~0b110)
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [0, 0], [0, 0]]}>
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: cluster_load_b128
  tt.func public @cluster_load_b128(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
    // CHECK: %[[CTA_ID:.*]] = {{.*}}llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[NON_FREE_BITS:.*]] = llvm.mlir.constant(-7 : i32) : i32
    // CHECK: %[[SHIFT_AMOUNT:.*]] = llvm.and %[[CTA_ID]], %[[NON_FREE_BITS]]
    // CHECK: %[[GROUP_MASK:.*]] = llvm.mlir.constant(85 : i32) : i32
    // CHECK: %[[CTA_MASK:.*]] = llvm.shl %[[GROUP_MASK]], %[[SHIFT_AMOUNT]]
    // CHECK: llvm.amdgcn.cluster.load.b128{{.*}}, {{.*}}, %[[CTA_MASK]]
    // CHECK-NOT: llvm.amdgcn.cluster.load
    %6 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Note that we already check the correct multicast mask in previous tests, so we only check the cluster load instruction here
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [0, 0], [0, 0]]}>
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: cluster_load_b64
  tt.func public @cluster_load_b64(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
    // CHECK-COUNT-2: llvm.amdgcn.cluster.load.b64
    // CHECK-NOT: llvm.amdgcn.cluster.load
    %6 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Note that we already check the correct multicast mask in previous tests, so we only check the cluster load instruction here
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [0, 0], [0, 0]]}>
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: cluster_load_b32
  tt.func public @cluster_load_b32(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
    // CHECK-COUNT-4: llvm.amdgcn.cluster.load.b32
    // CHECK-NOT: llvm.amdgcn.cluster.load
    %6 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Smaller vector size than 2 (32bit) should not produce cluster loads
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [0, 0], [0, 0]]}>
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: not_cluster_load_for_b16
  tt.func public @not_cluster_load_for_b16(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
    // CHECK-NOT: llvm.amdgcn.cluster.load
    %6 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Check that we break sizePerThread > 4 (>128bit) into multiple cluster loads b128
// Note that we already check the correct multicast mask in previous tests, so we only check the cluster load instruction here
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [0, 0], [0, 0]]}>
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: cluster_load_2_b128
  tt.func public @cluster_load_2_b128(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
    // CHECK-COUNT-2: llvm.amdgcn.cluster.load.b128
    // CHECK-NOT: llvm.amdgcn.cluster.load
    %6 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
