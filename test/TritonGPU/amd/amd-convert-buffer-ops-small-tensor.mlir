// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="arch-generation-name=gfx942 analyze-small-tensor-ofst=false" | FileCheck %s --check-prefixes=COMMON,GFX942-ONLY
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="arch-generation-name=gfx950 analyze-small-tensor-ofst=false" | FileCheck %s --check-prefixes=COMMON,GFX950-ONLY

//////////////////////////////////////////////////////////////////////////////
//
//   This file contains lit tests primarily for buffer-ops conversion for
// small-tensor (size <= 2G) with analyze-small-tensor-ofst being off
// (default value).
//
//   The initial revision of this file is copied from amd-convert-buffer-ops.mlir
// with following changes:
//    - some completely irrelevant tests are removed
//    - some tests are slightly modified to demonstrate some conversion
//      can be done with skip-small-tensor-ofst-analysis=false
//
// TODO: some testings still need polishing to make them more relevant to
// small-tensor-offset related optimization. Regardless, it's no harm to keep
// them.
//
//////////////////////////////////////////////////////////////////////////////
//
#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // COMMON-LABEL: simple
    tt.func @simple(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 :i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    // COMMON: %[[offset:.*]] = arith.addi
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // COMMON: buffer_load %arg0[%[[offset]]]
    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    // COMMON: buffer_load %arg1[%[[offset]]]
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    // COMMON: %[[data:.*]] = arith.addf
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // COMMON: buffer_store %[[data]], %arg2[%[[offset]]]
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: assume_positive_offset
  tt.func @assume_positive_offset(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) ->  tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %sub = arith.subi %1, %c128_i32 : i32
    %cmp = arith.cmpi sgt, %sub, %c0_i32 : i32
    llvm.intr.assume %cmp : i1
    %2 = tt.splat %sub : i32 -> tensor<1024xi32, #blocked>
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // COMMON: %[[offset:.*]] = arith.addi
    %4 = arith.addi %2, %3 : tensor<1024xi32, #blocked>
    // COMMON: %[[scalar_ptr:.*]] = tt.addptr %arg0
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // COMMON: buffer_load %[[scalar_ptr]][%[[offset]]]
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %10 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32}  {
  // COMMON-LABEL: offset_64_bits
  tt.func @offset_64_bits(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<1024xf32, #blocked> {
    %c1024_i32 = arith.constant 1024 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %sub = arith.subi %1, %c128_i32 : i32
    %2 = tt.splat %sub : i32 -> tensor<1024xi32, #blocked>
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ext2 = arith.extsi %2 : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
    %ext3 = arith.extsi %3 : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
    %4 = arith.addi %ext2, %ext3 : tensor<1024xi64, #blocked>
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi64, #blocked>
    // COMMON: tt.load
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %10 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32}  {
  // COMMON-LABEL: offset_64_bits_narrow
  tt.func public @offset_64_bits_narrow(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) -> tensor<1024xf32, #blocked> {
    %c1024_i32 = arith.constant 1024 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.splat %1: i32 -> tensor<1024xi32, #blocked>
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ext2 = arith.extsi %2 : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
    %ext3 = arith.extsi %3 : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
    %4 = arith.addi %ext2, %ext3 : tensor<1024xi64, #blocked>
    // COMMON: %[[scalar_ptr:.*]] = tt.addptr %arg0
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // COMMON: %[[offset_32_bit:.*]] = arith.trunci
    %narrow4 = arith.trunci %4 : tensor<1024xi64, #blocked> to tensor <1024xi32, #blocked>
    %9 = tt.addptr %8, %narrow4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // COMMON: buffer_load %[[scalar_ptr]][%[[offset_32_bit]]]
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %10 : tensor<1024xf32, #blocked>
  }
}

// -----
// NOTE: compared to @non_canonical_ptr in amd-convert-buffer-ops.mlir, the load
// can be converted to buffer-loads.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32}  {
  // COMMON-LABEL: non_canonical_ptr
  tt.func @non_canonical_ptr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: tensor<1024xi32, #blocked>) -> tensor<1024xf32, #blocked>{
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %arg1: tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // COMMON: buffer_load
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %10 : tensor<1024xf32, #blocked>
  }
}

// -----

// NOTE: compared the @assume_eq_non_neg in amd-convert-buffer-ops.mlir.
//  tt.load and tt.store can be converted without tl.assume.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: assume_eq_non_neg
  tt.func @assume_eq_non_neg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32) {
    %c10_i32 = arith.constant 10 : i32
    // COMMON: %[[range:.*]] = tt.make_range
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    // COMMON: %[[ptr:.*]] = tt.addptr %arg0, %arg2
    %2 = tt.addptr %arg0, %arg2: !tt.ptr<bf16>, i32
    %3 = tt.splat %2 : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>, #blocked>
    %4 = tt.addptr %3, %1 : tensor<16x!tt.ptr<bf16>, #blocked>, tensor<16xi32, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>, #blocked>
    %6 = tt.addptr %5, %1 : tensor<16x!tt.ptr<bf16>, #blocked>, tensor<16xi32, #blocked>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg1[%[[range]]]
    %7 = tt.load %6 : tensor<16x!tt.ptr<bf16>, #blocked>
    // COMMON: amdgpu.buffer_store %[[loaded]], %[[ptr]][%[[range]]]
    tt.store %4, %7 : tensor<16x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// NOTE: compared to the @assume_nonneg_less in amd-convert-buffer-ops.mlir.
//  tl.assume are removed.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: assume_nonneg_less
  tt.func @assume_nonneg_less(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32) {
    %c10_i32 = arith.constant 5 : i32
    // %0 = arith.cmpi slt, %c10_i32, %arg2 : i32
    // llvm.intr.assume %0 : i1
    // COMMON: %[[range:.*]] = tt.make_range
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    // COMMON: %[[ptr:.*]] = tt.addptr %arg0, %arg2
    %2 = tt.addptr %arg0, %arg2: !tt.ptr<bf16>, i32
    %3 = tt.splat %2 : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>, #blocked>
    %4 = tt.addptr %3, %1 : tensor<16x!tt.ptr<bf16>, #blocked>, tensor<16xi32, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>, #blocked>
    %6 = tt.addptr %5, %1 : tensor<16x!tt.ptr<bf16>, #blocked>, tensor<16xi32, #blocked>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg1[%[[range]]]
    %7 = tt.load %6 : tensor<16x!tt.ptr<bf16>, #blocked>
    // COMMON: amdgpu.buffer_store %[[loaded]], %[[ptr]][%[[range]]]
    tt.store %4, %7 : tensor<16x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// NOTE: compared to the @assume_nonneg_less in amd-convert-buffer-ops.mlir.
//  tl.assume are removed.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: assume_cmp_non_const
  tt.func @assume_cmp_non_const(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32, %arg3 : i32, %arg4 : i32, %arg5 : i32, %arg6 : i32) {
    %0 = arith.cmpi sgt, %arg2, %arg3 : i32
    llvm.intr.assume %0 : i1
    %1 = arith.subi %arg2, %arg3 : i32
    %2 = arith.cmpi sge, %1, %arg4 : i32
    // llvm.intr.assume %2 : i1
    %3 = arith.subi %1, %arg4 : i32
    %4 = arith.cmpi slt, %3, %arg5 : i32
    // llvm.intr.assume %4 : i1
    %5 = arith.subi %arg5, %3 : i32
    %6 = arith.cmpi sle, %5, %arg6 : i32
    // llvm.intr.assume %6 : i1
    %7 = arith.subi %arg6, %5 : i32
    %8 = arith.minsi %1, %3 : i32
    %9 = arith.minsi %8, %5 : i32
    %10 = arith.minsi %9, %7 : i32
    // COMMON: %[[range:.*]] = tt.make_range
    %11 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %12 = tt.splat %10 : i32 -> tensor<16xi32, #blocked>
    // COMMON: %[[offsets:.*]] = arith.addi
    %offsets = arith.addi %11, %12 : tensor<16xi32, #blocked>
    %13 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>, #blocked>
    %14 = tt.addptr %13, %11 : tensor<16x!tt.ptr<bf16>, #blocked>, tensor<16xi32, #blocked>
    %15 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>, #blocked>
    %16 = tt.addptr %15, %offsets : tensor<16x!tt.ptr<bf16>, #blocked>, tensor<16xi32, #blocked>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg1[%[[offsets]]]
    %17 = tt.load %16 : tensor<16x!tt.ptr<bf16>, #blocked>
    // COMMON: amdgpu.buffer_store %[[loaded]], %arg0[%[[range]]]
    tt.store %14, %17 : tensor<16x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blockedtrans = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.slice<{dim=0, parent=#blocked}>
#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: unary_triton_ops_transitive_nonneg
  tt.func @unary_triton_ops_transitive_nonneg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %c10_i32 = arith.constant 5 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked1>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32, #blocked1> -> tensor<1x16xi32, #blocked>
    %2 = tt.reshape %1 allow_reorder : tensor<1x16xi32, #blocked> -> tensor<8x2xi32, #blocked>
    %3 = tt.reshape %1 allow_reorder : tensor<1x16xi32, #blocked> -> tensor<2x8xi32, #blocked>
    %4 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<2x8xi32, #blocked> -> tensor<8x2xi32, #blockedtrans>
    %5 = ttg.convert_layout %4 : tensor<8x2xi32, #blockedtrans> -> tensor<8x2xi32, #blocked>
    %6 = arith.addi %5, %2 : tensor<8x2xi32, #blocked>
    %7 = tt.make_range {end = 10 : i32, start = 2 : i32} : tensor<8xi32, #blocked2>
    %8 = ttg.convert_layout %7 : tensor<8xi32, #blocked2> -> tensor<8xi32, #blocked1>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<8xi32, #blocked1> -> tensor<1x8xi32, #blocked>
    %10 = tt.broadcast %9 : tensor<1x8xi32, #blocked> -> tensor<2x8xi32, #blocked>
    %11 = tt.reshape %10 allow_reorder : tensor<2x8xi32, #blocked> -> tensor<8x2xi32, #blocked>
    %12 = tt.splat %c10_i32 : i32 -> tensor<8x2xi32, #blocked>
    %13 = arith.addi %11, %12 : tensor<8x2xi32, #blocked>
    %14 = arith.minsi %13, %5 : tensor<8x2xi32, #blocked>
    // COMMON: %[[lhs:.*]], %[[rhs:.*]] = tt.split
    %15, %16 = tt.split %11: tensor<8x2xi32, #blocked> -> tensor<8xi32, #blocked2>
    %17 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked2>
    %18 = tt.addptr %17, %15 : tensor<8x!tt.ptr<bf16>, #blocked2>, tensor<8xi32, #blocked2>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg0[%[[lhs]]]
    %19 = tt.load %18 : tensor<8x!tt.ptr<bf16>, #blocked2>
    %20 = tt.addptr %17, %16 : tensor<8x!tt.ptr<bf16>, #blocked2>, tensor<8xi32, #blocked2>
    // COMMON: %[[loaded2:.*]] = amdgpu.buffer_load %arg0[%[[rhs]]]
    %21 = tt.load %20 : tensor<8x!tt.ptr<bf16>, #blocked2>
    // COMMON: %[[added:.*]] = arith.addf %[[loaded]], %[[loaded2]]
    %22 = arith.addf %19, %21 : tensor<8xbf16, #blocked2>
    %23 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked2>
    %24 = tt.addptr %23, %7 : tensor<8x!tt.ptr<bf16>, #blocked2>, tensor<8xi32, #blocked2>
    // COMMON: amdgpu.buffer_store %[[added]], %arg1[%{{.*}}]
    tt.store %24, %22 : tensor<8x!tt.ptr<bf16>, #blocked2>
    tt.return
  }
}

// -----


#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: join_cat_transitive_nonneg
  tt.func @join_cat_transitive_nonneg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked1>
    %1 = tt.make_range {end = 10 : i32, start = 2 : i32} : tensor<8xi32, #blocked1>
    %2 = tt.join %0, %1 : tensor<8xi32, #blocked1> -> tensor<8x2xi32, #blocked>
    %3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked1>
    %4 = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32, #blocked1>
    %5 = tt.join %3, %4 : tensor<4xi32, #blocked1> -> tensor<4x2xi32, #blocked>
    %6 = tt.cat %5, %5 : tensor<4x2xi32, #blocked> -> tensor<8x2xi32, #blocked>
    %7 = arith.addi %2, %6 : tensor<8x2xi32, #blocked>
    %zeros = arith.constant dense<0> : tensor<8x1xi32, #blocked>
    %ones = arith.constant dense<1> : tensor<8x1xi32, #blocked>
    %8 = tt.gather %7[%zeros] {axis = 1 : i32} : (tensor<8x2xi32, #blocked>, tensor<8x1xi32, #blocked>) -> tensor<8x1xi32, #blocked>
    %9 = tt.gather %7[%ones] {axis = 1 : i32} : (tensor<8x2xi32, #blocked>, tensor<8x1xi32, #blocked>) -> tensor<8x1xi32, #blocked>
    %10 = arith.addi %8, %9 : tensor<8x1xi32, #blocked>
    %11 = tt.reshape %10 allow_reorder : tensor<8x1xi32, #blocked> -> tensor<8xi32, #blocked1>
    %12 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked1>
    %14 = tt.addptr %12, %11 : tensor<8x!tt.ptr<bf16>, #blocked1>, tensor<8xi32, #blocked1>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg0[%{{.*}}]
    %15 = tt.load %14 : tensor<8x!tt.ptr<bf16>, #blocked1>
    %16 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked1>
    %17 = tt.addptr %16, %0 : tensor<8x!tt.ptr<bf16>, #blocked1>, tensor<8xi32, #blocked1>
    // COMMON: amdgpu.buffer_store %[[loaded]], %arg1[%{{.*}}]
    tt.store %17, %15 : tensor<8x!tt.ptr<bf16>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: histo_nonneg
  tt.func @histo_nonneg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : tensor<256xi32, #blocked>) {
    /// Purposely specify %arg2 so that we can't statically determine the input
    /// data is nonneg.
    // COMMON: tt.histogram
    %0 = tt.histogram %arg2 : tensor<256xi32, #blocked> -> tensor<8xi32, #blocked>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %0 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg0[%{{.*}}]
    %4 = tt.load %3 : tensor<8x!tt.ptr<bf16>, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %6 = tt.addptr %5, %1 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: amdgpu.buffer_store %[[loaded]], %arg1[%{{.*}}]
    tt.store %6, %4 : tensor<8x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: get_num_prog_nonneg
  tt.func @get_num_prog_nonneg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : i32) {
    %0 = tt.get_num_programs x : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.get_num_programs z : i32
    %3 = arith.minsi %0, %1 : i32
    %4 = arith.minsi %2, %3 : i32
    %5 = arith.maxsi %arg2, %4 : i32
    %6 = tt.splat %5 : i32 -> tensor<8xi32, #blocked>
    %7 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<8xi32, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %10 = tt.addptr %9, %8 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg0[%{{.*}}]
    %11 = tt.load %10 : tensor<8x!tt.ptr<bf16>, #blocked>
    %12 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %13 = tt.addptr %12, %7 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: amdgpu.buffer_store %[[loaded]], %arg1[%{{.*}}]
    tt.store %13, %11 : tensor<8x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: unsigned_ops
  tt.func @unsigned_ops(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : i32, %arg3 : i32, %arg4 : f32) {
    %c5_i32 = arith.constant 5 : i32
    %0 = arith.ceildivui %arg2, %c5_i32 : i32
    %1 = arith.divui %arg3, %c5_i32 : i32
    %2 = arith.fptoui %arg4 : f32 to i32
    %4 = arith.maxui %arg2, %arg3 : i32
    %5 = arith.minui %arg2, %arg3 : i32
    %6 = arith.remui %arg2, %c5_i32 : i32
    %7 = arith.shrui %arg3, %c5_i32 : i32
    %8 = arith.addi %0, %1 : i32
    %10 = arith.addi %4, %5 : i32
    %11 = arith.addi %6, %7 : i32
    %12 = arith.addi %8, %2 : i32
    %13 = arith.addi %10, %11 : i32
    %14 = arith.addi %8, %13 : i32
    %15 = tt.splat %14 : i32 -> tensor<8xi32, #blocked>
    %16 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
    %17 = arith.addi %15, %16 : tensor<8xi32, #blocked>
    %18 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %19 = tt.addptr %18, %17 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg0[%{{.*}}]
    %20 = tt.load %19 : tensor<8x!tt.ptr<bf16>, #blocked>
    %21 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %22 = tt.addptr %21, %16 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: amdgpu.buffer_store %[[loaded]], %arg1[%{{.*}}]
    tt.store %22, %20 : tensor<8x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: extui_nonneg
  tt.func @extui_nonneg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : i32) {
    %0 = arith.extui %arg2 : i32 to i64
    %1 = tt.splat %0 : i64 -> tensor<8xi64, #blocked>
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
    %3 = arith.extui %2 : tensor<8xi32, #blocked> to tensor<8xi64, #blocked>
    %4 = arith.addi %1, %3 : tensor<8xi64, #blocked>
    %5 = arith.trunci %4 : tensor<8xi64, #blocked> to tensor<8xi32, #blocked>
    %6 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %7 = tt.addptr %6, %5 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg0[%{{.*}}]
    %8 = tt.load %7: tensor<8x!tt.ptr<bf16>, #blocked>
    %9 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %10 = tt.addptr %9, %2 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: amdgpu.buffer_store %[[loaded]], %arg1[%{{.*}}]
    tt.store %10, %8 : tensor<8x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: traverse_if
  tt.func @traverse_if(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : i32, %arg3 : i32) {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c5_i32 = arith.constant 7 : i32
    %c7_i32 = arith.constant 5 : i32
    %0 = arith.extui %arg2 : i32 to i64
    %1 = arith.remui %arg2, %c2_i32 : i32
    %2 = arith.cmpi eq, %1, %c0_i32 : i32
    %3 = scf.if %2 -> tensor<8xi64, #blocked> {
      %20 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
      %21 = arith.extui %20 : tensor<8xi32, #blocked> to tensor<8xi64, #blocked>
      %22 = tt.splat %arg3 : i32 -> tensor<8xi32, #blocked>
      %23 = arith.extui %22 : tensor<8xi32, #blocked> to tensor<8xi64, #blocked>
      %24 = arith.addi %21, %23 : tensor<8xi64, #blocked>
      scf.yield %24 : tensor<8xi64, #blocked>
    } else {
      %30 = tt.make_range {end = 16 : i32, start = 8 : i32} : tensor<8xi32, #blocked>
      %31 = arith.extui %30 : tensor<8xi32, #blocked> to tensor<8xi64, #blocked>
      %32 = tt.splat %0 : i64 -> tensor<8xi64, #blocked>
      %33 = arith.addi %31, %32 : tensor<8xi64, #blocked>
      scf.yield %33 : tensor<8xi64, #blocked>
    }
    %4 = arith.trunci %3 : tensor<8xi64, #blocked> to tensor<8xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg0[%{{.*}}]
    %7 = tt.load %6: tensor<8x!tt.ptr<bf16>, #blocked>
    %8 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
    %9 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %10 = tt.addptr %9, %8 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: amdgpu.buffer_store %[[loaded]], %arg1[%{{.*}}]
    tt.store %10, %7 : tensor<8x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: traverse_if
  tt.func @traverse_if(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : i32, %arg3 : i32) {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c5_i32 = arith.constant 7 : i32
    %c7_i32 = arith.constant 5 : i32
    %zeros = arith.constant dense<0> : tensor<8xi32, #blocked>
    %0 = arith.extui %arg2 : i32 to i64
    %1 = arith.remui %arg2, %c2_i32 : i32
    %2 = arith.cmpi eq, %1, %c0_i32 : i32
    %3, %4 = scf.if %2 -> (tensor<8xi64, #blocked>, tensor<8xi32, #blocked>) {
      %20 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
      %21 = arith.extui %20 : tensor<8xi32, #blocked> to tensor<8xi64, #blocked>
      %22 = tt.splat %arg3 : i32 -> tensor<8xi32, #blocked>
      %23 = arith.extui %22 : tensor<8xi32, #blocked> to tensor<8xi64, #blocked>
      %24 = arith.addi %21, %23 : tensor<8xi64, #blocked>
      %25 = tt.make_range {end = 9 : i32, start = 1 : i32} : tensor<8xi32, #blocked>
      scf.yield %24, %25 : tensor<8xi64, #blocked>, tensor<8xi32, #blocked>
    } else {
      %30 = tt.make_range {end = 16 : i32, start = 8 : i32} : tensor<8xi32, #blocked>
      %31 = arith.extui %30 : tensor<8xi32, #blocked> to tensor<8xi64, #blocked>
      %32 = tt.splat %0 : i64 -> tensor<8xi64, #blocked>
      %33 = arith.addi %31, %32 : tensor<8xi64, #blocked>
      scf.yield %33, %zeros : tensor<8xi64, #blocked>, tensor<8xi32, #blocked>
    }
    %5 = arith.trunci %3 : tensor<8xi64, #blocked> to tensor<8xi32, #blocked>
    %6 = arith.addi %4, %5 : tensor<8xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %8 = tt.addptr %7, %6 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_load %arg0[%{{.*}}]
    %9 = tt.load %8: tensor<8x!tt.ptr<bf16>, #blocked>
    %10 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
    %11 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %12 = tt.addptr %11, %10 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // COMMON: amdgpu.buffer_store %[[loaded]], %arg1[%{{.*}}]
    tt.store %12, %9 : tensor<8x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: atomic_add_bf16
  tt.func public @atomic_add_bf16(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst = arith.constant dense<true> : tensor<512xi1, #blocked>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<512xbf16, #blocked>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<bf16>, i32
    %4 = tt.splat %3 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
    // GFX942-ONLY-NOT: amdgpu.buffer_atomic_rmw
    // GFX950-ONLY: amdgpu.buffer_atomic_rmw
    %6 = tt.atomic_rmw fadd, acq_rel, gpu, %5, %cst_0, %cst : (tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xbf16, #blocked>, tensor<512xi1, #blocked>) -> tensor<512xbf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: assume_positive_offset_buffer_atomic
  tt.func @assume_positive_offset_buffer_atomic(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: tensor<1024xf32, #blocked>) ->  tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %sub = arith.subi %1, %c128_i32 : i32
    %cmp = arith.cmpi sgt, %sub, %c0_i32 : i32
    llvm.intr.assume %cmp : i1
    %2 = tt.splat %sub : i32 -> tensor<1024xi32, #blocked>
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // COMMON: %[[offset:.*]] = arith.addi
    %4 = arith.addi %2, %3 : tensor<1024xi32, #blocked>
    // COMMON: %[[scalar_ptr:.*]] = tt.addptr %arg0
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %6 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %7 = tt.addptr %6, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // COMMON: %[[loaded:.*]] = amdgpu.buffer_atomic_rmw fadd, acq_rel, gpu, %arg1, %[[scalar_ptr]][%[[offset]]]
    %8 = tt.atomic_rmw fadd, acq_rel, gpu, %7, %arg1 : (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    tt.return %8 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [2, 2], order = [1, 0]}>

module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @extract_slice(%arg0: !tt.ptr<f32>) -> tensor<128x256xf32, #blocked> {
    %0 = arith.constant dense<0> : tensor<256x256xi64, #blocked>
    %1 = amdgpu.extract_slice %0 [0, 0] : tensor<256x256xi64, #blocked> to tensor<128x256xi64, #blocked>
    %2 = arith.trunci %1 : tensor<128x256xi64, #blocked> to tensor<128x256xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<128x256x!tt.ptr<f32>, #blocked>, tensor<128x256xi32, #blocked>
    %5 = tt.load %4 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return %5 : tensor<128x256xf32, #blocked>
  }
}

// COMMON-LABEL: tt.func @extract_slice(
// COMMON-SAME:    %[[ARG_0:.*]]: !tt.ptr<f32>) -> tensor<128x256xf32, #blocked> {
// COMMON:    %[[VAR_0:.*]] = arith.constant dense<0> : tensor<256x256xi64, #blocked>
// COMMON:    %[[VAR_1:.*]] = amdgpu.extract_slice %[[VAR_0]] [0, 0] : tensor<256x256xi64, #blocked> to tensor<128x256xi64, #blocked>
// COMMON:    %[[VAR_2:.*]] = arith.trunci %[[VAR_1]] : tensor<128x256xi64, #blocked> to tensor<128x256xi32, #blocked>
// COMMON:    %[[VAR_3:.*]] = amdgpu.buffer_load %[[ARG_0]][%[[VAR_2]]] : tensor<128x256xf32, #blocked>
// COMMON:    tt.return %[[VAR_3]] : tensor<128x256xf32, #blocked>
// COMMON:  }

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_atomic_cas_i64
  tt.func public @buffer_atomic_cas_i64(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // COMMON: %[[val:.*]] = arith.constant dense<2>
    %cst = arith.constant dense<2> : tensor<1024xi64, #blocked>
    // COMMON: %[[cmp:.*]] = arith.constant dense<0>
    %cst_0 = arith.constant dense<0> : tensor<1024xi64, #blocked>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    // COMMON: %[[offset:.*]] = tt.make_range
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // COMMON: %[[scalar_ptr:.*]] = tt.addptr %arg0
    %3 = tt.addptr %arg0, %1 : !tt.ptr<i64>, i32
    %4 = tt.splat %3 : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<i64>, #blocked>, tensor<1024xi32, #blocked>
    // COMMON: amdgpu.buffer_atomic_cas acq_rel, gpu, %[[cmp]], %[[val]], %[[scalar_ptr]][%[[offset]]]
    %6 = tt.atomic_cas acq_rel, gpu, %5, %cst_0, %cst : (tensor<1024x!tt.ptr<i64>, #blocked>, tensor<1024xi64, #blocked>, tensor<1024xi64, #blocked>) -> tensor<1024xi64, #blocked>
    %7 = tt.addptr %arg1, %1 : !tt.ptr<i64>, i32
    %8 = tt.splat %7 : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>, #blocked>
    %9 = tt.addptr %8, %2 : tensor<1024x!tt.ptr<i64>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %9, %6 : tensor<1024x!tt.ptr<i64>, #blocked>
    tt.return
  }
}
