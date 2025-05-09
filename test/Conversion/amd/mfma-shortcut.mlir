// RUN: triton-opt %s --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx942" -split-input-file | FileCheck %s --check-prefix=GFX942
// RUN: triton-opt %s --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx950" -split-input-file | FileCheck %s --check-prefix=GFX950

#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: shortcut_mfma16
  tt.func public @shortcut_mfma16(%arg0: tensor<16x16xf16, #mfma>) {
    // GFX942-NOT: store
    // GFX942-NOT: load
    // GFX942: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mfma> -> tensor<16x16xf16, #dotop>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: no_shortcut_mfma16
  tt.func public @no_shortcut_mfma16(%arg0: tensor<16x16xf16, #mfma>) {
    // GFX942: store
    // GFX942: load
    // GFX942: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mfma> -> tensor<16x16xf16, #dotop>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: mfma_dot_cvt_f8_mfma32
  tt.func public @mfma_dot_cvt_f8_mfma32(%arg0: tensor<128x32xf8E4M3FNUZ, #mfma>) {
    // GFX942-NOT: store
    // GFX942-NOT: load

    // GFX942: [[val3:%.*]] = llvm.extractvalue %arg0[3]
    // GFX942: [[val7:%.*]] = llvm.extractvalue %arg0[7]

    // GFX942-DAG: [[c32:%.*]] = llvm.mlir.constant(32 : i32)
    // GFX942-DAG: [[c64:%.*]] = llvm.mlir.constant(64 : i32)

    // GFX942: [[threadId:%.*]] = rocdl.workitem.id.x
    // GFX942: [[laneId:%.*]] = llvm.urem [[threadId]], [[c64]]
    // GFX942: [[mask0:%.*]] = llvm.icmp "slt" [[laneId]], [[c32]]

    // GFX942: [[shflLaneId:%.*]] = llvm.add [[laneId]], [[c32]]
    // GFX942: [[addr32:%.*]] = llvm.urem [[shflLaneId]], [[c64]]

    // GFX942: [[vec0:%.*]] = llvm.insertelement [[val3]], {{.*}} : vector<4xi8>
    // GFX942: [[vec1:%.*]] = llvm.insertelement [[val7]], {{.*}} : vector<4xi8>

    // GFX942: [[bvec0:%.*]] = llvm.bitcast [[vec0]]
    // GFX942: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // GFX942: [[addr:%.*]] = llvm.shl [[addr32]], [[c2]]
    // GFX942: [[bShflVec0:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec0]]
    // GFX942: [[shflVec0:%.*]] = llvm.bitcast [[bShflVec0]]

    // GFX942: [[bvec1:%.*]] = llvm.bitcast [[vec1]]
    // GFX942: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // GFX942: [[addr:%.*]] = llvm.shl [[addr32]], [[c2]]
    // GFX942: [[bShflVec1:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec1]]
    // GFX942: [[shflVec1:%.*]] = llvm.bitcast [[bShflVec1]]

    // Input (8 values): (vec0, vec1)
    // Output (8 values shuffled, '>> n' - take the value from (lane + n) % 64):
    //                 resVec0     resVec1
    //   lanes  0-31: (vec0      , vec0 >> 32) (mask0=1)
    //   lanes 32-63: (vec1 >> 32, vec1      ) (mask0=0)

    // GFX942: [[resVec0:%.*]] = llvm.select [[mask0]], [[vec0]], [[shflVec1]]
    // GFX942: [[resVec1:%.*]] = llvm.select [[mask0]], [[shflVec0]], [[vec1]]

    // GFX942: [[c3:%.*]] = llvm.mlir.constant(3 : i32)
    // GFX942: [[resVal3:%.*]] = llvm.extractelement [[resVec0]][[[c3]] : i32] : vector<4xi8>
    // GFX942: [[c3:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // GFX942: [[resVal7:%.*]] = llvm.extractelement [[resVec1]][[[c3]] : i32] : vector<4xi8>

    // GFX942: llvm.insertvalue [[resVal3]], {{.*}}[3]
    // GFX942: llvm.insertvalue [[resVal7]], {{.*}}[7]

    // GFX942: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E4M3FNUZ, #mfma> -> tensor<128x32xf8E4M3FNUZ, #dotop0>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: mfma_dot_cvt_bf8_mfma32
  tt.func public @mfma_dot_cvt_bf8_mfma32(%arg0: tensor<128x32xf8E5M2, #mfma>) {
    // GFX942-NOT: store
    // GFX942-NOT: load
    // GFX942: rocdl.ds_bpermute
    // GFX942: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E5M2, #mfma> -> tensor<128x32xf8E5M2, #dotop0>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: mfma_dot_cvt_f8_mfma16
  tt.func public @mfma_dot_cvt_f8_mfma16(%arg0: tensor<128x32xf8E4M3FNUZ, #mfma>) {
    // GFX942-NOT: store
    // GFX942-NOT: load

    // GFX942: [[val3:%.*]] = llvm.extractvalue %arg0[3]
    // GFX942: [[val7:%.*]] = llvm.extractvalue %arg0[7]

    // GFX942-DAG: [[c16:%.*]] = llvm.mlir.constant(16 : i32)
    // GFX942-DAG: [[c32:%.*]] = llvm.mlir.constant(32 : i32)
    // GFX942-DAG: [[c48:%.*]] = llvm.mlir.constant(48 : i32)
    // GFX942-DAG: [[c64:%.*]] = llvm.mlir.constant(64 : i32)

    // GFX942: [[threadId:%.*]] = rocdl.workitem.id.x
    // GFX942: [[laneId:%.*]] = llvm.urem [[threadId]], [[c64]]
    // GFX942: [[mask0:%.*]] = llvm.icmp "slt" [[laneId]], [[c32]]

    // GFX942: [[laneIdRem:%.*]] = llvm.urem [[laneId]], [[c32]]
    // GFX942: [[mask1:%.*]] = llvm.icmp "slt" [[laneIdRem]], [[c16]]

    // GFX942: [[shflLaneId:%.*]] = llvm.add [[laneId]], [[c16]]
    // GFX942: [[addr16:%.*]] = llvm.urem [[shflLaneId]], [[c64]]

    // GFX942: [[shflLaneId:%.*]] = llvm.add [[laneId]], [[c32]]
    // GFX942: [[addr32:%.*]] = llvm.urem [[shflLaneId]], [[c64]]

    // GFX942: [[shflLaneId:%.*]] = llvm.add [[laneId]], [[c48]]
    // GFX942: [[addr48:%.*]] = llvm.urem [[shflLaneId]], [[c64]]

    // GFX942: [[vec0:%.*]] = llvm.insertelement [[val3]], {{.*}} : vector<4xi8>
    // GFX942: [[vec1:%.*]] = llvm.insertelement [[val7]], {{.*}} : vector<4xi8>

    // GFX942: [[bvec0:%.*]] = llvm.bitcast [[vec0]]
    // GFX942: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // GFX942: [[addr:%.*]] = llvm.shl [[addr16]], [[c2]]
    // GFX942: [[bShflVec0_16:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec0]]
    // GFX942: [[shflVec0_16:%.*]] = llvm.bitcast [[bShflVec0_16]]

    // GFX942: [[bvec0:%.*]] = llvm.bitcast [[vec0]]
    // GFX942: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // GFX942: [[addr:%.*]] = llvm.shl [[addr32]], [[c2]]
    // GFX942: [[bShflVec0_32:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec0]]
    // GFX942: [[shflVec0_32:%.*]] = llvm.bitcast [[bShflVec0_32]]

    // GFX942: [[bvec1:%.*]] = llvm.bitcast [[vec1]]
    // GFX942: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // GFX942: [[addr:%.*]] = llvm.shl [[addr32]], [[c2]]
    // GFX942: [[bShflVec1_32:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec1]]
    // GFX942: [[shflVec1_32:%.*]] = llvm.bitcast [[bShflVec1_32]]

    // GFX942: [[bvec1:%.*]] = llvm.bitcast [[vec1]]
    // GFX942: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // GFX942: [[addr:%.*]] = llvm.shl [[addr48]], [[c2]]
    // GFX942: [[bShflVec1_48:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec1]]
    // GFX942: [[shflVec1_48:%.*]] = llvm.bitcast [[bShflVec1_48]]

    // Input (8 values): (vec0, vec1)
    // Output (8 values shuffled, '>> n' - take the value from (lane + n) % 64):
    //                 resVec0     resVec1
    //   lanes  0-15: (vec0      , vec0 >> 16) (mask0=1, mask1=1)
    //   lanes 16-31: (vec0 >> 16, vec0 >> 32) (mask0=1, mask1=0)
    //   lanes 32-47: (vec1 >> 32, vec1 >> 48) (mask0=0, mask1=1)
    //   lanes 48-63: (vec1 >> 48, vec1      ) (mask0=0, mask1=0)

    // GFX942-DAG: [[mask0_true:%.*]] = llvm.select [[mask1]], [[vec0]], [[shflVec0_16]] : i1, vector<4xi8>
    // GFX942-DAG: [[mask0_false:%.*]] = llvm.select [[mask1]], [[shflVec1_32]], [[shflVec1_48]] : i1, vector<4xi8>
    // GFX942: [[resVec0:%.*]] = llvm.select [[mask0]], [[mask0_true]], [[mask0_false]] : i1, vector<4xi8>

    // GFX942-DAG: [[mask0_true:%.*]] = llvm.select [[mask1]], [[shflVec0_16]], [[shflVec0_32]] : i1, vector<4xi8>
    // GFX942-DAG: [[mask0_false:%.*]] = llvm.select [[mask1]], [[shflVec1_48]], [[vec1]] : i1, vector<4xi8>
    // GFX942: [[resVec1:%.*]] = llvm.select [[mask0]], [[mask0_true]], [[mask0_false]] : i1, vector<4xi8>

    // GFX942: [[c3:%.*]] = llvm.mlir.constant(3 : i32)
    // GFX942: [[resVal3:%.*]] = llvm.extractelement [[resVec0]][[[c3]] : i32] : vector<4xi8>
    // GFX942: [[c3:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // GFX942: [[resVal7:%.*]] = llvm.extractelement [[resVec1]][[[c3]] : i32] : vector<4xi8>

    // GFX942: llvm.insertvalue [[resVal3]], {{.*}}[3]
    // GFX942: llvm.insertvalue [[resVal7]], {{.*}}[7]

    // GFX942: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E4M3FNUZ, #mfma> -> tensor<128x32xf8E4M3FNUZ, #dotop0>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: mfma_dot_cvt_bf8_mfma16
  tt.func public @mfma_dot_cvt_bf8_mfma16(%arg0: tensor<128x32xf8E5M2, #mfma>) {
    // GFX942-NOT: store
    // GFX942-NOT: load
    // GFX942: rocdl.ds_bpermute
    // GFX942: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E5M2, #mfma> -> tensor<128x32xf8E5M2, #dotop0>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [64, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 4, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: mfma_linear_permlane_swap
  tt.func public @mfma_linear_permlane_swap(%arg0: tensor<128x128xf16, #mma>) attributes {noinline = false} {
  // GFX950-COUNT-16: llvm.call_intrinsic "llvm.amdgcn.permlane32.swap"
    %1 = ttg.convert_layout %arg0: tensor<128x128xf16, #mma> -> tensor<128x128xf16, #linear>
    tt.return
  }
}
