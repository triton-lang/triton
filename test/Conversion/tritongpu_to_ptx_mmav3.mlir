// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx83 | FileCheck --dump-input-context=20 %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#dot_op = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth=4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// CHECK-LABEL: cvt_mma_to_dot_fp8
  tt.func @cvt_mma_to_dot_fp8(%ptr : !llvm.ptr, %arg0: tensor<128x64xf8E5M2, #mma>) {

    // As there are 64 elements per lane, we don't use variables to track them.

    // CHECK-COUNT-64: ld.param.b8

    // Intra-warp layout conversions can be viewed as a permutation of register
    // and lane basis vectors. This can be read off from the linear layouts:
    //
    // #mma:     register: [[0,1], [8,0], [0,8], [0,16], [0,32], [64,0]]
    //               lane: [[0,2], [0,4], [1,0], [2,0], [4,0]]
    //               warp: [[16,0], [32,0]]
    //
    // #dot_op:  register: [[0,1], [0,2], [8,0], [0,16], [0,32], [64,0]]
    //               lane: [[0,4], [0,8], [1,0], [2,0], [4,0]]
    //               warp: [[16,0], [32,0]]
    //
    // The layout conversion is described by the permutation (r1 r2 l1 l0),
    // which factors as (r1 l1)(l0 l1)(r1 r2).
    //
    // Register basis vectors correspond to the bits of the indices of the 64
    // separate registers which hold the original elements. Since we end up
    // packing 4 elements per register, we end up with only 16 registers in
    // total before shuffling. The `transferWithinWarp` implementation handles
    // register packing by ensuring that elements are packed together only if
    // under the layout conversion, they end up in the same destination lane.
    // To do this, it rearranges the 64 registers so that it can pack 4
    // consecutive elements at a time according to their new register index.
    //
    // The transposition (r1 l1) above indicates that intially, elements with
    // register indices whose r1 bit is on are to be moved to new lanes. We thus
    // need to rearrange the registers. The algorithm chooses the next register
    // bit > 1 which is not used in a mixed transposition. In this case,
    // that bit is r2. Algebrically, this corresponds to conjugating the
    // permutation with (r1 r2). This produces (r1 r2)(r2 l1)(l0 l1). The new
    // (r1 r2) at the end rearranges elements after unpacking, and only
    // (r2 l1)(l0 l1) matters for tracking the movement of the packed registers.
    // From the point of view of the packed registers, the symbol `r2` now
    // corresponds to the 0th bit of a (packed) register's index.
    //
    // The transposition (r2 l1) is a bit swap which is implemented in-place as:
    //  1. r2 ^= l1
    //  2. l1 ^= r2
    //  3. r2 ^= l1.
    // The algorithm conjugates (l0 l1) through the first two stages to produce:
    //  1. r2 ^= l0
    //  2a. l0 ^= r2
    //  2b. (l0 l1)
    //  3. r2 ^= l1.
    // The first step is to get the value of l0.

    // CHECK: mov.u32       [[TID:%.*]], %tid.x;
    // CHECK: and.b32       [[L0_VAL:%.*]], [[TID]], 1;
    // CHECK: setp.eq.s32   [[L0_OFF:%.*]], [[L0_VAL]], 0;

    // This is used to perform 16 independent selects in stage 1.

    // CHECK-COUNT-16: selp.b32     {{.*}}, {{.*}}, [[L0_OFF]];

    // Next, we apply (l0 l1) to the lane id to get the base source lane for
    // the index shuffles. This is step 2b above, but since we must specify
    // the *source* lane for a warp-shuffle, it gets applied first in practice:
    //
    //       dstLane = ((l0 l1) \circ (l0 ^= r2))(srcLane)
    //       srcLane = ((l0 ^= r2) \circ (l0 l1))(dstLane)
    //
    // To apply (l0 l1), we use a compile-time mask to collect the fixed bits,
    // and then we OR it with the shifted l0 and l1 values.

    // CHECK-DAG: and.b32 [[LANEID_FIXED_BITS:%.*]], [[TID]], 28;
    // CHECK-DAG: shl.b32 [[L0_TEMP:%.*]], [[L0_VAL]], 1;
    // CHECK-DAG: or.b32  [[LANEID_PART_PERM:%.*]], [[L0_TEMP]], [[LANEID_FIXED_BITS]];
    // CHECK-DAG: bfe.u32 [[L1_TEMP:%.*]], [[TID]], 1, 1;
    // CHECK-DAG: or.b32  [[LANEID_PERM:%.*]], [[LANEID_PART_PERM]], [[L1_TEMP]];

    // The index shuffles have source lane dependent on the value of the r2 bit.
    // Half of them use `LANEID_PERM` while the other half use `LANEID_PERM`
    // with the l0 bit flipped (step 2a).

    // CHECK-DAG: xor.b32     [[LANEID_PERM_F:%.*]], [[LANEID_PERM]], 1;

    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM_F]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM_F]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM_F]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM_F]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM_F]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM_F]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM_F]], 31, -1;
    // CHECK-DAG: shfl.sync.idx.b32     {{.*}}, [[LANEID_PERM_F]], 31, -1;

    // Finally, the last set of selects are performed, using the value of l1 as
    // the predicate (step 3).

    // CHECK-DAG: and.b32           [[L1_VAL:%.*]], [[TID]], 2;
    // CHECK-DAG: setp.eq.s32       [[L1_OFF:%.*]], [[L1_VAL]], 0;
    // CHECK-COUNT-16: selp.b32     {{.*}}, {{.*}}, [[L1_OFF]];

    // CHECK-COUNT-64: bfe.u32
    // CHECK-COUNT-64: st.volatile.global.b8

    %0 = ttg.convert_layout %arg0 : tensor<128x64xf8E5M2, #mma> -> tensor<128x64xf8E5M2, #dot_op>
    %1 = builtin.unrealized_conversion_cast %0 : tensor<128x64xf8E5M2, #dot_op> to !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    llvm.store volatile %1, %ptr : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>, !llvm.ptr

    tt.return
  }
}
