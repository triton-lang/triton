// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops='arch-generation-name=gfx942'| FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: simple
    tt.func @simple(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 :i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32,
                    %arg4: !ttg.memdesc<256xf32, #shared, #smem, mutable>) -> !ttg.async.token {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    // CHECK: %[[offset:.*]] = arith.addi
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // CHECK: buffer_load_to_local %arg0[%[[offset]]], %arg4
    %9 = ttg.async_copy_global_to_local %6, %arg4 : tensor<256x!tt.ptr<f32>, #blocked0> -> <256xf32, #shared, #smem, mutable>
    tt.return %9 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
// CHECK-LABEL: buffer_stride
  tt.func public @buffer_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32},
                                %arg10: !ttg.memdesc<256x64xf16, #shared, #smem, mutable>) -> !ttg.async.token {
    %c48_i32 = arith.constant 48 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %cmp = arith.cmpi sgt, %arg6, %c0_i32 : i32
    llvm.intr.assume %cmp : i1
    %2 = tt.splat %arg6 : i32 -> tensor<256x1xi32, #blocked>
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #blocked>
    %4 = tt.addptr %arg0, %c32_i32 : !tt.ptr<f16>, i32
    %5 = tt.broadcast %3 : tensor<256x1xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %9 = arith.addi %8, %5 : tensor<256x64xi32, #blocked>
    %10 = tt.splat %4 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %9 : tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<256x64xi32, #blocked>

    // CHECK: %[[splat:.*]] = tt.splat %arg[[#stride:]]
    // CHECK: %[[mul:.*]] = arith.muli %[[#]], %[[splat]]
    // CHECK: %[[ptr:.*]] = tt.addptr %arg0
    // CHECK: %[[bcast1:.*]] = tt.broadcast %[[mul]]
    // CHECK: %[[bcast0:.*]] = tt.broadcast %[[#]]
    // CHECK: %[[offset:.*]] = arith.addi %[[bcast0]], %[[bcast1]]
    // CHECK: %[[buffer:.*]] = amdgpu.buffer_load_to_local %[[ptr]][%[[offset]]], %arg10 stride = %arg[[#stride]]

    %12 = ttg.async_copy_global_to_local %11, %arg10 : tensor<256x64x!tt.ptr<f16>, #blocked> -> <256x64xf16, #shared, #smem, mutable>
    tt.return %12 : !ttg.async.token
  }
}
// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: assume_positive_offset
  tt.func @assume_positive_offset(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) -> !ttg.async.token {
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
    // CHECK: %[[offset:.*]] = arith.addi
    %4 = arith.addi %2, %3 : tensor<1024xi32, #blocked>
    // CHECK: %[[scalar_ptr:.*]] = tt.addptr %arg0
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: buffer_load_to_local %[[scalar_ptr]][%[[offset]]], %arg1
    %10 = ttg.async_copy_global_to_local %9, %arg1 : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    tt.return %10 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32}  {
  // CHECK-LABEL: offset_64_bits
  tt.func @offset_64_bits(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) -> !ttg.async.token {
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
    // CHECK: ttg.async_copy_global_to_local
    %10 = ttg.async_copy_global_to_local %9, %arg1 : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    tt.return %10 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32}  {
  // CHECK-LABEL: offset_64_bits_narrow
  tt.func public @offset_64_bits_narrow(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) -> !ttg.async.token {
    %c1024_i32 = arith.constant 1024 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.splat %1: i32 -> tensor<1024xi32, #blocked>
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ext2 = arith.extsi %2 : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
    %ext3 = arith.extsi %3 : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
    %4 = arith.addi %ext2, %ext3 : tensor<1024xi64, #blocked>
    // CHECK: %[[scalar_ptr:.*]] = tt.addptr %arg0
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[offset_32_bit:.*]] = arith.trunci
    %narrow4 = arith.trunci %4 : tensor<1024xi64, #blocked> to tensor <1024xi32, #blocked>
    %9 = tt.addptr %8, %narrow4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: buffer_load_to_local %[[scalar_ptr]][%[[offset_32_bit]]], %arg4
    %10 = ttg.async_copy_global_to_local %9, %arg4 : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    tt.return %10 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32}  {
  // CHECK-LABEL: non_canonical_ptr
  tt.func @non_canonical_ptr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: tensor<1024xi32, #blocked>, %arg2: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) -> !ttg.async.token {
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %arg1: tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: ttg.async_copy_global_to_local
    %10 = ttg.async_copy_global_to_local %9, %arg2 : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    tt.return %10 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: assume_eq_non_neg
  tt.func @assume_eq_non_neg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32, %arg3: !ttg.memdesc<16xbf16, #shared, #smem, mutable>) -> !ttg.async.token {
    %c10_i32 = arith.constant 10 : i32
    %0 = arith.cmpi eq, %arg2, %c10_i32 : i32
    llvm.intr.assume %0 : i1
    // CHECK: %[[range:.*]] = tt.make_range
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    // CHECK: %[[ptr:.*]] = tt.addptr %arg0, %arg2
    %2 = tt.addptr %arg0, %arg2: !tt.ptr<bf16>, i32
    %3 = tt.splat %2 : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>, #blocked>
    %4 = tt.addptr %3, %1 : tensor<16x!tt.ptr<bf16>, #blocked>, tensor<16xi32, #blocked>
    // CHECK: amdgpu.buffer_load_to_local %[[ptr]][%[[range]]]
    %7 = ttg.async_copy_global_to_local %4, %arg3 : tensor<16x!tt.ptr<bf16>, #blocked> -> <16xbf16, #shared, #smem, mutable>
    tt.return %7 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: assume_nonneg_less
  tt.func @assume_nonneg_less(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32, %arg3: !ttg.memdesc<16xbf16, #shared, #smem, mutable>) -> !ttg.async.token {
    %c10_i32 = arith.constant 5 : i32
    %0 = arith.cmpi slt, %c10_i32, %arg2 : i32
    llvm.intr.assume %0 : i1
    // CHECK: %[[range:.*]] = tt.make_range
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    // CHECK: %[[ptr:.*]] = tt.addptr %arg0, %arg2
    %2 = tt.addptr %arg0, %arg2: !tt.ptr<bf16>, i32
    %3 = tt.splat %2 : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>, #blocked>
    %4 = tt.addptr %3, %1 : tensor<16x!tt.ptr<bf16>, #blocked>, tensor<16xi32, #blocked>
    // CHECK: amdgpu.buffer_load_to_local %[[ptr]][%[[range]]]
    %7 = ttg.async_copy_global_to_local %4, %arg3 : tensor<16x!tt.ptr<bf16>, #blocked> -> <16xbf16, #shared, #smem, mutable>
    tt.return %7 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: assume_cmp_non_const
  tt.func @assume_cmp_non_const(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32, %arg3 : i32, %arg4 : i32, %arg5 : i32, %arg6 : i32, %arg7: !ttg.memdesc<16xbf16, #shared, #smem, mutable>) -> !ttg.async.token {
    %0 = arith.cmpi sgt, %arg2, %arg3 : i32
    llvm.intr.assume %0 : i1
    %1 = arith.subi %arg2, %arg3 : i32
    %2 = arith.cmpi sge, %1, %arg4 : i32
    llvm.intr.assume %2 : i1
    %3 = arith.subi %1, %arg4 : i32
    %4 = arith.cmpi slt, %3, %arg5 : i32
    llvm.intr.assume %4 : i1
    %5 = arith.subi %arg5, %3 : i32
    %6 = arith.cmpi sle, %5, %arg6 : i32
    llvm.intr.assume %6 : i1
    %7 = arith.subi %arg6, %5 : i32
    %8 = arith.minsi %1, %3 : i32
    %9 = arith.minsi %8, %5 : i32
    %10 = arith.minsi %9, %7 : i32
    // CHECK: %[[range:.*]] = tt.make_range
    %11 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %12 = tt.splat %10 : i32 -> tensor<16xi32, #blocked>
    // CHECK: %[[offsets:.*]] = arith.addi
    %offsets = arith.addi %11, %12 : tensor<16xi32, #blocked>
    %15 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x!tt.ptr<bf16>, #blocked>
    %16 = tt.addptr %15, %offsets : tensor<16x!tt.ptr<bf16>, #blocked>, tensor<16xi32, #blocked>
    // CHECK: amdgpu.buffer_load_to_local %arg1[%[[offsets]]], %arg7
    %17 = ttg.async_copy_global_to_local %16, %arg7 : tensor<16x!tt.ptr<bf16>, #blocked> -> <16xbf16, #shared, #smem, mutable>
    tt.return %17 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blockedtrans = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.slice<{dim=0, parent=#blocked}>
#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: unary_triton_ops_transitive_nonneg
  tt.func @unary_triton_ops_transitive_nonneg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !ttg.memdesc<8xbf16, #shared, #smem, mutable>) {
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
    // CHECK: %[[lhs:.*]], %[[rhs:.*]] = tt.split
    %15, %16 = tt.split %11: tensor<8x2xi32, #blocked> -> tensor<8xi32, #blocked2>
    %17 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked2>
    %18 = tt.addptr %17, %15 : tensor<8x!tt.ptr<bf16>, #blocked2>, tensor<8xi32, #blocked2>
    // CHECK: amdgpu.buffer_load_to_local %arg0[%[[lhs]]]
    // CHECK: %[[loaded:.*]] = ttg.local_load
    %190 = ttg.async_copy_global_to_local %18, %arg2 : tensor<8x!tt.ptr<bf16>, #blocked2> -> <8xbf16, #shared, #smem, mutable>
    %19 = ttg.local_load %arg2 token %190 : !ttg.memdesc<8xbf16, #shared, #smem, mutable> -> tensor<8xbf16, #blocked2>
    %20 = tt.addptr %17, %16 : tensor<8x!tt.ptr<bf16>, #blocked2>, tensor<8xi32, #blocked2>
    // CHECK: amdgpu.buffer_load_to_local %arg0[%[[rhs]]]
    // CHECK: %[[loaded2:.*]] = ttg.local_load
    %210 = ttg.async_copy_global_to_local %20, %arg2 : tensor<8x!tt.ptr<bf16>, #blocked2> -> <8xbf16, #shared, #smem, mutable>
    %21 = ttg.local_load %arg2 token %210 : !ttg.memdesc<8xbf16, #shared, #smem, mutable> -> tensor<8xbf16, #blocked2>
    // CHECK: %[[added:.*]] = arith.addf %[[loaded]], %[[loaded2]]
    %22 = arith.addf %19, %21 : tensor<8xbf16, #blocked2>
    %23 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked2>
    %24 = tt.addptr %23, %7 : tensor<8x!tt.ptr<bf16>, #blocked2>, tensor<8xi32, #blocked2>
    // CHECK: amdgpu.buffer_store %[[added]], %arg1[%{{.*}}]
    tt.store %24, %22 : tensor<8x!tt.ptr<bf16>, #blocked2>
    tt.return
  }
}

// -----


#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: join_cat_transitive_nonneg
  tt.func @join_cat_transitive_nonneg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !ttg.memdesc<8xbf16, #shared, #smem, mutable>) -> !ttg.async.token {
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
    // CHECK: amdgpu.buffer_load_to_local %arg0[%{{.*}}]
    %15 = ttg.async_copy_global_to_local %14, %arg2 : tensor<8x!tt.ptr<bf16>, #blocked1> -> <8xbf16, #shared, #smem, mutable>
    tt.return %15 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: histo_nonneg
  tt.func @histo_nonneg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : tensor<256xi32, #blocked>, %arg3: !ttg.memdesc<8xbf16, #shared, #smem, mutable>) -> !ttg.async.token {
    /// Purposely specify %arg2 so that we can't statically determine the input
    /// data is nonneg.
    // CHECK: tt.histogram
    %0 = tt.histogram %arg2 : tensor<256xi32, #blocked> -> tensor<8xi32, #blocked>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %0 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // CHECK: amdgpu.buffer_load_to_local %arg0[%{{.*}}]
    %4 = ttg.async_copy_global_to_local %3, %arg3 : tensor<8x!tt.ptr<bf16>, #blocked> -> <8xbf16, #shared, #smem, mutable>
    tt.return %4 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: get_num_prog_nonneg
  tt.func @get_num_prog_nonneg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : i32, %arg3: !ttg.memdesc<8xbf16, #shared, #smem, mutable>) -> !ttg.async.token  {
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
    // CHECK: amdgpu.buffer_load_to_local %arg0[%{{.*}}]
    %11 = ttg.async_copy_global_to_local %10, %arg3 : tensor<8x!tt.ptr<bf16>, #blocked> -> <8xbf16, #shared, #smem, mutable>
    tt.return %11 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: unsigned_ops
  tt.func @unsigned_ops(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : i32, %arg3 : i32, %arg4 : f32, %arg5: !ttg.memdesc<8xbf16, #shared, #smem, mutable>) -> !ttg.async.token {
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
    // CHECK: amdgpu.buffer_load_to_local %arg0[%{{.*}}]
    %20 = ttg.async_copy_global_to_local %19, %arg5 : tensor<8x!tt.ptr<bf16>, #blocked> -> <8xbf16, #shared, #smem, mutable>
    tt.return %20 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: extui_nonneg
  tt.func @extui_nonneg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : i32, %arg3: !ttg.memdesc<8xbf16, #shared, #smem, mutable>) -> !ttg.async.token {
    %0 = arith.extui %arg2 : i32 to i64
    %1 = tt.splat %0 : i64 -> tensor<8xi64, #blocked>
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
    %3 = arith.extui %2 : tensor<8xi32, #blocked> to tensor<8xi64, #blocked>
    %4 = arith.addi %1, %3 : tensor<8xi64, #blocked>
    %5 = arith.trunci %4 : tensor<8xi64, #blocked> to tensor<8xi32, #blocked>
    %6 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked>
    %7 = tt.addptr %6, %5 : tensor<8x!tt.ptr<bf16>, #blocked>, tensor<8xi32, #blocked>
    // CHECK: amdgpu.buffer_load_to_local %arg0[%{{.*}}]
    %8 = ttg.async_copy_global_to_local %7, %arg3 : tensor<8x!tt.ptr<bf16>, #blocked> -> <8xbf16, #shared, #smem, mutable>
    tt.return %8 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: traverse_if
  tt.func @traverse_if(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : i32, %arg3 : i32, %arg4: !ttg.memdesc<8xbf16, #shared, #smem, mutable>) -> !ttg.async.token {
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
    // CHECK: amdgpu.buffer_load_to_local %arg0[%{{.*}}]
    %7 = ttg.async_copy_global_to_local %6, %arg4 : tensor<8x!tt.ptr<bf16>, #blocked> -> <8xbf16, #shared, #smem, mutable>
    tt.return %7 : !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: traverse_if
  tt.func @traverse_if(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2 : i32, %arg3 : i32, %arg4: !ttg.memdesc<8xbf16, #shared, #smem, mutable>) -> !ttg.async.token {
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
    // CHECK: amdgpu.buffer_load_to_local %arg0[%{{.*}}]
    %9 = ttg.async_copy_global_to_local %8, %arg4 : tensor<8x!tt.ptr<bf16>, #blocked> -> <8xbf16, #shared, #smem, mutable>
    tt.return %9 : !ttg.async.token
  }
}
