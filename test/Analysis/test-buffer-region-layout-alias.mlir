// RUN: triton-opt %s -split-input-file -mlir-disable-threading -test-buffer-region-alias -o /dev/null 2>&1 | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: swizzled_q00 vs swizzled_q00
// CHECK: swizzled_q00 vs swizzled_q01: alias=false
// CHECK: swizzled_q00 vs swizzled_q10: alias=false
// CHECK: swizzled_q00 vs swizzled_q11: alias=false
// CHECK: swizzled_q00 vs swizzled_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: swizzled_q01 vs swizzled_q10: alias=false
// CHECK: swizzled_q01 vs swizzled_q11: alias=false
// CHECK: swizzled_q01 vs swizzled_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: swizzled_q10 vs swizzled_q11: alias=false
// CHECK: swizzled_q10 vs swizzled_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: swizzled_q11 vs swizzled_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 2048 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @swizzled_quadrants() {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    %q00 = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %q01 = ttg.memdesc_subslice %parent [0, 8] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %q10 = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %q11 = ttg.memdesc_subslice %parent [8, 8] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %0 = ttg.local_load %q00 {test.region_name = "swizzled_q00"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %1 = ttg.local_load %q01 {test.region_name = "swizzled_q01"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %2 = ttg.local_load %q10 {test.region_name = "swizzled_q10"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %3 = ttg.local_load %q11 {test.region_name = "swizzled_q11"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %4 = ttg.local_load %parent {test.region_name = "swizzled_zfull"} : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> tensor<16x16xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

// A nested physical offset must compose to the same footprint as the direct
// subview. The containing view and a disjoint sibling guard both directions.
// CHECK-LABEL: nested_direct vs nested_direct
// CHECK: nested_direct vs nested_disjoint: alias=false
// CHECK: nested_direct vs nested_nested: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: nested_direct vs nested_outer: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: nested_disjoint vs nested_nested: alias=false
// CHECK: nested_disjoint vs nested_outer: alias=false
// CHECK: nested_nested vs nested_outer: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 8192 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @nested_smem_subviews() {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %outer = ttg.memdesc_subslice %parent [16, 16] : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable, 32x32>
    %nested = ttg.memdesc_subslice %outer [8, 8] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable, 32x32> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 32x32>
    %direct = ttg.memdesc_subslice %parent [24, 24] : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 32x32>
    %disjoint = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 32x32>
    %0 = ttg.local_load %direct {test.region_name = "nested_direct"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 32x32> -> tensor<8x8xf32>
    %1 = ttg.local_load %disjoint {test.region_name = "nested_disjoint"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 32x32> -> tensor<8x8xf32>
    %2 = ttg.local_load %nested {test.region_name = "nested_nested"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 32x32> -> tensor<8x8xf32>
    %3 = ttg.local_load %outer {test.region_name = "nested_outer"} : !ttg.memdesc<16x16xf32, #shared, #smem, mutable, 32x32> -> tensor<16x16xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared_t = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory

// Slicing after a logical transpose must preserve the physical footprint.
// CHECK-LABEL: trans_view_original vs trans_view_original
// CHECK: trans_view_original vs trans_view_same: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: trans_view_original vs trans_view_sibling: alias=false
// CHECK: trans_view_same vs trans_view_sibling: alias=false
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 2048 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @subview_after_transpose() {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
    %original = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<8x32xf16, #shared, #smem, mutable, 16x32>
    %trans = ttg.memdesc_trans %parent {order = array<i32: 1, 0>} : !ttg.memdesc<16x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x16xf16, #shared_t, #smem, mutable>
    %same = ttg.memdesc_subslice %trans [0, 0] : !ttg.memdesc<32x16xf16, #shared_t, #smem, mutable> -> !ttg.memdesc<32x8xf16, #shared_t, #smem, mutable, 32x16>
    %sibling = ttg.memdesc_subslice %trans [0, 8] : !ttg.memdesc<32x16xf16, #shared_t, #smem, mutable> -> !ttg.memdesc<32x8xf16, #shared_t, #smem, mutable, 32x16>
    %0 = ttg.local_load %original {test.region_name = "trans_view_original"} : !ttg.memdesc<8x32xf16, #shared, #smem, mutable, 16x32> -> tensor<8x32xf16>
    %1 = ttg.local_load %same {test.region_name = "trans_view_same"} : !ttg.memdesc<32x8xf16, #shared_t, #smem, mutable, 32x16> -> tensor<32x8xf16>
    %2 = ttg.local_load %sibling {test.region_name = "trans_view_sibling"} : !ttg.memdesc<32x8xf16, #shared_t, #smem, mutable, 32x16> -> tensor<32x8xf16>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// Row-then-column and column-then-row TMEM slicing must commute in the
// canonical encoded word address space.
// CHECK-LABEL: nested_tmem_col_row vs nested_tmem_col_row
// CHECK: nested_tmem_col_row vs nested_tmem_other: alias=false
// CHECK: nested_tmem_col_row vs nested_tmem_row: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: nested_tmem_col_row vs nested_tmem_row_col: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: nested_tmem_other vs nested_tmem_row: alias=false
// CHECK: nested_tmem_other vs nested_tmem_row_col: alias=false
// CHECK: nested_tmem_row vs nested_tmem_row_col: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=false
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @nested_tmem_subviews() {
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %row = ttng.tmem_subslice %parent {offset = 128 : i32, dim = 0 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %row_col = ttng.tmem_subslice %row {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %col = ttng.tmem_subslice %parent {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %col_row = ttng.tmem_subslice %col {offset = 128 : i32, dim = 0 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %other = ttng.tmem_subslice %parent {offset = 0 : i32, dim = 0 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %0 = ttng.tmem_load %col_row {test.region_name = "nested_tmem_col_row"} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<128x64xf32>
    %1 = ttng.tmem_load %other {test.region_name = "nested_tmem_other"} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<128x128xf32>
    %2 = ttng.tmem_load %row {test.region_name = "nested_tmem_row"} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<128x128xf32>
    %3 = ttng.tmem_load %row_col {test.region_name = "nested_tmem_row_col"} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<128x64xf32>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// CHECK-LABEL: tmem_m0 vs tmem_m0
// CHECK: tmem_m0 vs tmem_m1: alias=false
// CHECK: tmem_m0 vs tmem_n0: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false
// CHECK: tmem_m0 vs tmem_n1: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false
// CHECK: tmem_m0 vs tmem_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: tmem_m1 vs tmem_n0: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false
// CHECK: tmem_m1 vs tmem_n1: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false
// CHECK: tmem_m1 vs tmem_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: tmem_n0 vs tmem_n1: alias=false
// CHECK: tmem_n0 vs tmem_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: tmem_n1 vs tmem_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tmem_m_n_splits() {
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %m0 = ttng.tmem_subslice %parent {offset = 0 : i32, dim = 0 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %m1 = ttng.tmem_subslice %parent {offset = 128 : i32, dim = 0 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %n0 = ttng.tmem_subslice %parent {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %n1 = ttng.tmem_subslice %parent {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %0 = ttng.tmem_load %m0 {test.region_name = "tmem_m0"} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<128x128xf32>
    %1 = ttng.tmem_load %m1 {test.region_name = "tmem_m1"} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<128x128xf32>
    %2 = ttng.tmem_load %n0 {test.region_name = "tmem_n0"} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<256x64xf32>
    %3 = ttng.tmem_load %n1 {test.region_name = "tmem_n1"} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<256x64xf32>
    %4 = ttng.tmem_load %parent {test.region_name = "tmem_zfull"} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// f16 pairs adjacent logical columns into one canonical TMEM word. At a
// nonzero row/column base, its two N-halves are disjoint and their union has
// the same word footprint as the f32 view.
// CHECK-LABEL: packing_f16_high vs packing_f16_high
// CHECK: packing_f16_high vs packing_f16_low: alias=false
// CHECK: packing_f16_high vs packing_f32: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: packing_f16_low vs packing_f32: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tmem_subword_packing() {
    %f16 = ttng.tmem_alloc {tensor_memory_col_offset = 16 : i32, tensor_memory_row_offset = 8 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %low = ttng.tmem_subslice %f16 {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable, 128x128>
    %high = ttng.tmem_subslice %f16 {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable, 128x128>
    %f32 = ttng.tmem_alloc {tensor_memory_col_offset = 16 : i32, tensor_memory_row_offset = 8 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = ttng.tmem_load %high {test.region_name = "packing_f16_high"} : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf16>
    %1 = ttng.tmem_load %low {test.region_name = "packing_f16_low"} : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf16>
    %2 = ttng.tmem_load %f32 {test.region_name = "packing_f32"} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory

// CHECK-LABEL: transposed_q00 vs transposed_q00
// CHECK: transposed_q00 vs transposed_q01: alias=false
// CHECK: transposed_q00 vs transposed_q10: alias=false
// CHECK: transposed_q00 vs transposed_q11: alias=false
// CHECK: transposed_q00 vs transposed_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: transposed_q01 vs transposed_q10: alias=false
// CHECK: transposed_q01 vs transposed_q11: alias=false
// CHECK: transposed_q01 vs transposed_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: transposed_q10 vs transposed_q11: alias=false
// CHECK: transposed_q10 vs transposed_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: transposed_q11 vs transposed_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 2048 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @transposed_quadrants() {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    %q00 = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %q01 = ttg.memdesc_subslice %parent [0, 8] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %q10 = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %q11 = ttg.memdesc_subslice %parent [8, 8] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %0 = ttg.local_load %q00 {test.region_name = "transposed_q00"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %1 = ttg.local_load %q01 {test.region_name = "transposed_q01"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %2 = ttg.local_load %q10 {test.region_name = "transposed_q10"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %3 = ttg.local_load %q11 {test.region_name = "transposed_q11"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %4 = ttg.local_load %parent {test.region_name = "transposed_zfull"} : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> tensor<16x16xf32>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [16, 16]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: padded_q00 vs padded_q00
// CHECK: padded_q00 vs padded_q01: alias=false
// CHECK: padded_q00 vs padded_q10: alias=false
// CHECK: padded_q00 vs padded_q11: alias=false
// CHECK: padded_q00 vs padded_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: padded_q01 vs padded_q10: alias=false
// CHECK: padded_q01 vs padded_q11: alias=false
// CHECK: padded_q01 vs padded_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: padded_q10 vs padded_q11: alias=false
// CHECK: padded_q10 vs padded_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: padded_q11 vs padded_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 2048 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @padded_quadrants() {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    %q00 = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %q01 = ttg.memdesc_subslice %parent [0, 8] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %q10 = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %q11 = ttg.memdesc_subslice %parent [8, 8] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %0 = ttg.local_load %q00 {test.region_name = "padded_q00"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %1 = ttg.local_load %q01 {test.region_name = "padded_q01"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %2 = ttg.local_load %q10 {test.region_name = "padded_q10"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %3 = ttg.local_load %q11 {test.region_name = "padded_q11"} : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %4 = ttg.local_load %parent {test.region_name = "padded_zfull"} : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> tensor<16x16xf32>
    tt.return
  }
}

// -----

// A partitioned allocation owns every physical base, not the interval starting
// at its first base. Subviews select a physical partition and remain exact
// through nested views and runtime selection.
#inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner}>
#smem = #ttg.shared_memory

// CHECK-LABEL: partition_a_parent vs partition_a_parent: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: partition_a_parent vs partition_b_first: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=false
// CHECK: partition_a_parent vs partition_c_second: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=false
// CHECK: partition_a_parent vs partition_d_first_base: alias=true
// CHECK: partition_a_parent vs partition_e_second_base: alias=true
// CHECK: partition_a_parent vs partition_g_nested: alias=true
// CHECK: partition_a_parent vs partition_h_nested_base: alias=true
// CHECK: partition_b_first vs partition_c_second: alias=false
// CHECK: partition_b_first vs partition_d_first_base: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: partition_b_first vs partition_e_second_base: alias=false
// CHECK: partition_b_first vs partition_f_selected: alias=true
// CHECK: partition_c_second vs partition_d_first_base: alias=false
// CHECK: partition_c_second vs partition_e_second_base: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: partition_c_second vs partition_f_selected: alias=true
// CHECK: partition_c_second vs partition_g_nested: alias=false
// CHECK: partition_d_first_base vs partition_e_second_base: alias=false
// CHECK: partition_g_nested vs partition_h_nested_base: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: partition_a_parent case [0, 512]: mask={0,1,2,3}
// CHECK: partition_b_first case [0, 256]: mask={0}
// CHECK: partition_c_second case [2048, 256]: mask={2}
// CHECK: partition_d_first_base case [0, 128]: mask={0}
// CHECK: partition_e_second_base case [2048, 128]: mask={2}
// CHECK: partition_f_selected case [0, 256]: mask={0}
// CHECK: partition_f_selected case [2048, 256]: mask={2}
// CHECK: partition_g_nested case [2176, 256]: mask={3}
// CHECK: partition_h_nested_base case [2176, 128]: mask={3}
// CHECK: state-plan: lanes=4
module attributes {test.print_state_plan, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.shared = 4096 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 2 : i32} {
  tt.func public @partitioned_shared_physical_alias(%choose: i1) {
    %parent = ttg.local_alloc {allocation.offset = [0 : i32, 2048 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %first = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %second = ttg.memdesc_subslice %parent [4, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %outer = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<8x16xf16, #partitioned, #smem, mutable, 16x16>
    %nested = ttg.memdesc_subslice %outer [4, 0] : !ttg.memdesc<8x16xf16, #partitioned, #smem, mutable, 16x16> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %selected = arith.select %choose, %first, %second : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %first_base = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x16xf16, #inner, #smem, mutable>
    %second_base = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<4x16xf16, #inner, #smem, mutable>
    %nested_base = ttg.local_alloc {allocation.offset = 2176 : i32} : () -> !ttg.memdesc<4x16xf16, #inner, #smem, mutable>
    %0 = ttg.local_load %parent {test.region_name = "partition_a_parent"} : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16>
    %1 = ttg.local_load %first {test.region_name = "partition_b_first"} : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16> -> tensor<4x16xf16>
    %2 = ttg.local_load %second {test.region_name = "partition_c_second"} : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16> -> tensor<4x16xf16>
    %3 = ttg.local_load %first_base {test.region_name = "partition_d_first_base"} : !ttg.memdesc<4x16xf16, #inner, #smem, mutable> -> tensor<4x16xf16>
    %4 = ttg.local_load %second_base {test.region_name = "partition_e_second_base"} : !ttg.memdesc<4x16xf16, #inner, #smem, mutable> -> tensor<4x16xf16>
    %5 = ttg.local_load %selected {test.region_name = "partition_f_selected"} : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16> -> tensor<4x16xf16>
    %6 = ttg.local_load %nested {test.region_name = "partition_g_nested"} : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16> -> tensor<4x16xf16>
    %7 = ttg.local_load %nested_base {test.region_name = "partition_h_nested_base"} : !ttg.memdesc<4x16xf16, #inner, #smem, mutable> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

// Padding is local to each physical partition; a gap between partition bases
// cannot be mistaken for part of either padded piece.
#piece_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [4, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #piece_padded}>
#smem = #ttg.shared_memory

// CHECK-LABEL: padded_partition_a_parent vs padded_partition_a_parent: alias=true
// CHECK: padded_partition_a_parent vs padded_partition_b_first: alias=true
// CHECK: padded_partition_a_parent vs padded_partition_c_second: alias=true
// CHECK: padded_partition_b_first vs padded_partition_c_second: alias=false
// CHECK: padded_partition_b_first vs padded_partition_d_first_base: alias=true
// CHECK: padded_partition_b_first vs padded_partition_e_second_base: alias=false
// CHECK: padded_partition_c_second vs padded_partition_d_first_base: alias=false
// CHECK: padded_partition_c_second vs padded_partition_e_second_base: alias=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.shared = 4096 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 2 : i32} {
  tt.func public @partitioned_padded_physical_alias() {
    %parent = ttg.local_alloc {allocation.offset = [0 : i32, 2048 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %first = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %second = ttg.memdesc_subslice %parent [4, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %first_base = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x16xf16, #piece_padded, #smem, mutable>
    %second_base = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<4x16xf16, #piece_padded, #smem, mutable>
    %0 = ttg.local_load %parent {test.region_name = "padded_partition_a_parent"} : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16>
    %1 = ttg.local_load %first {test.region_name = "padded_partition_b_first"} : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16> -> tensor<4x16xf16>
    %2 = ttg.local_load %second {test.region_name = "padded_partition_c_second"} : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16> -> tensor<4x16xf16>
    %3 = ttg.local_load %first_base {test.region_name = "padded_partition_d_first_base"} : !ttg.memdesc<4x16xf16, #piece_padded, #smem, mutable> -> tensor<4x16xf16>
    %4 = ttg.local_load %second_base {test.region_name = "padded_partition_e_second_base"} : !ttg.memdesc<4x16xf16, #piece_padded, #smem, mutable> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

// Advancing a pipeline stage advances every physical partition base. A dynamic
// stage aliases each possible static stage without making the stages overlap.
#inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner}>
#smem = #ttg.shared_memory

// CHECK-LABEL: stage_partition_a_first vs stage_partition_a_first: alias=true
// CHECK: stage_partition_a_first vs stage_partition_b_second: alias=false
// CHECK: stage_partition_a_first vs stage_partition_c_dynamic: alias=true
// CHECK: stage_partition_a_first vs stage_partition_d_second_first: alias=false
// CHECK: stage_partition_b_second vs stage_partition_c_dynamic: alias=true
// CHECK: stage_partition_b_second vs stage_partition_d_second_first: alias=true
// CHECK: stage_partition_b_second vs stage_partition_e_second_peer: alias=true
// CHECK: stage_partition_d_second_first vs stage_partition_e_second_peer: alias=false
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.shared = 4096 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 2 : i32} {
  tt.func public @partitioned_multibuffer_physical_alias(%index: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %parent = ttg.local_alloc {allocation.offset = [0 : i32, 2048 : i32]} : () -> !ttg.memdesc<3x16x16xf16, #partitioned, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<3x16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<3x16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %dynamic = ttg.memdesc_index %parent[%index] : !ttg.memdesc<3x16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %second_first = ttg.memdesc_subslice %second [0, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %second_peer = ttg.memdesc_subslice %second [4, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %0 = ttg.local_load %first {test.region_name = "stage_partition_a_first"} : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16>
    %1 = ttg.local_load %second {test.region_name = "stage_partition_b_second"} : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16>
    %2 = ttg.local_load %dynamic {test.region_name = "stage_partition_c_dynamic"} : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16>
    %3 = ttg.local_load %second_first {test.region_name = "stage_partition_d_second_first"} : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16> -> tensor<4x16xf16>
    %4 = ttg.local_load %second_peer {test.region_name = "stage_partition_e_second_peer"} : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

// Partition selection follows the linear layout and physical base order, not
// the slicing dimension or the numerical ordering of allocation offsets.
#inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 4, numGroups = 2, partitionDim = 1, partitionLayout = #inner}>
#smem = #ttg.shared_memory

// CHECK-LABEL: column_partition_a_parent vs column_partition_a_parent: alias=true
// CHECK: column_partition_a_parent vs column_partition_b_first: alias=true
// CHECK: column_partition_a_parent vs column_partition_c_second: alias=true
// CHECK: column_partition_a_parent vs column_partition_d_third: alias=true
// CHECK: column_partition_a_parent vs column_partition_e_fourth: alias=true
// CHECK: column_partition_a_parent vs column_partition_f_next_group: alias=true
// CHECK: column_partition_a_parent vs column_partition_g_first_base: alias=true
// CHECK: column_partition_a_parent vs column_partition_h_second_base: alias=true
// CHECK: column_partition_a_parent vs column_partition_i_third_base: alias=true
// CHECK: column_partition_a_parent vs column_partition_j_fourth_base: alias=true
// CHECK: column_partition_a_parent vs column_partition_k_next_group_base: alias=true
// CHECK: column_partition_b_first vs column_partition_c_second: alias=false
// CHECK: column_partition_b_first vs column_partition_d_third: alias=false
// CHECK: column_partition_b_first vs column_partition_e_fourth: alias=false
// CHECK: column_partition_b_first vs column_partition_f_next_group: alias=false
// CHECK: column_partition_b_first vs column_partition_g_first_base: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: column_partition_c_second vs column_partition_h_second_base: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: column_partition_d_third vs column_partition_i_third_base: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: column_partition_e_fourth vs column_partition_j_fourth_base: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: column_partition_f_next_group vs column_partition_k_next_group_base: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.shared = 4096 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 2 : i32} {
  tt.func public @partitioned_four_column_physical_alias() {
    %parent = ttg.local_alloc {allocation.offset = [3072 : i32, 1024 : i32, 0 : i32, 2048 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %first = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<16x2xf16, #partitioned, #smem, mutable, 16x16>
    %second = ttg.memdesc_subslice %parent [0, 2] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<16x2xf16, #partitioned, #smem, mutable, 16x16>
    %third = ttg.memdesc_subslice %parent [0, 4] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<16x2xf16, #partitioned, #smem, mutable, 16x16>
    %fourth = ttg.memdesc_subslice %parent [0, 6] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<16x2xf16, #partitioned, #smem, mutable, 16x16>
    %next_group = ttg.memdesc_subslice %parent [0, 8] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<16x2xf16, #partitioned, #smem, mutable, 16x16>
    %first_base = ttg.local_alloc {allocation.offset = 3072 : i32} : () -> !ttg.memdesc<16x2xf16, #inner, #smem, mutable>
    %second_base = ttg.local_alloc {allocation.offset = 1024 : i32} : () -> !ttg.memdesc<16x2xf16, #inner, #smem, mutable>
    %third_base = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x2xf16, #inner, #smem, mutable>
    %fourth_base = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<16x2xf16, #inner, #smem, mutable>
    %next_group_base = ttg.local_alloc {allocation.offset = 3136 : i32} : () -> !ttg.memdesc<16x2xf16, #inner, #smem, mutable>
    %0 = ttg.local_load %parent {test.region_name = "column_partition_a_parent"} : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16>
    %1 = ttg.local_load %first {test.region_name = "column_partition_b_first"} : !ttg.memdesc<16x2xf16, #partitioned, #smem, mutable, 16x16> -> tensor<16x2xf16>
    %2 = ttg.local_load %second {test.region_name = "column_partition_c_second"} : !ttg.memdesc<16x2xf16, #partitioned, #smem, mutable, 16x16> -> tensor<16x2xf16>
    %3 = ttg.local_load %third {test.region_name = "column_partition_d_third"} : !ttg.memdesc<16x2xf16, #partitioned, #smem, mutable, 16x16> -> tensor<16x2xf16>
    %4 = ttg.local_load %fourth {test.region_name = "column_partition_e_fourth"} : !ttg.memdesc<16x2xf16, #partitioned, #smem, mutable, 16x16> -> tensor<16x2xf16>
    %5 = ttg.local_load %next_group {test.region_name = "column_partition_f_next_group"} : !ttg.memdesc<16x2xf16, #partitioned, #smem, mutable, 16x16> -> tensor<16x2xf16>
    %6 = ttg.local_load %first_base {test.region_name = "column_partition_g_first_base"} : !ttg.memdesc<16x2xf16, #inner, #smem, mutable> -> tensor<16x2xf16>
    %7 = ttg.local_load %second_base {test.region_name = "column_partition_h_second_base"} : !ttg.memdesc<16x2xf16, #inner, #smem, mutable> -> tensor<16x2xf16>
    %8 = ttg.local_load %third_base {test.region_name = "column_partition_i_third_base"} : !ttg.memdesc<16x2xf16, #inner, #smem, mutable> -> tensor<16x2xf16>
    %9 = ttg.local_load %fourth_base {test.region_name = "column_partition_j_fourth_base"} : !ttg.memdesc<16x2xf16, #inner, #smem, mutable> -> tensor<16x2xf16>
    %10 = ttg.local_load %next_group_base {test.region_name = "column_partition_k_next_group_base"} : !ttg.memdesc<16x2xf16, #inner, #smem, mutable> -> tensor<16x2xf16>
    tt.return
  }
}

// -----

// A valid shared-linear layout can interleave CTA ownership within one
// descriptor instead of expressing the CTA as a single affine view offset.
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0], [4, 0]], block = [[2, 0]]}, alignment = 16>
#smem = #ttg.shared_memory

// CHECK-LABEL: cta_block_a_parent vs cta_block_a_parent: alias=true
// CHECK: cta_block_a_parent vs cta_block_b_lower: alias=true
// CHECK: cta_block_a_parent vs cta_block_c_upper: alias=true
// CHECK: cta_block_b_lower vs cta_block_c_upper: alias=false
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 128 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @cross_cta_interleaved_linear_footprint() {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x4xf32, #shared, #smem, mutable>
    %lower = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<8x4xf32, #shared, #smem, mutable> -> !ttg.memdesc<4x4xf32, #shared, #smem, mutable, 8x4>
    %upper = ttg.memdesc_subslice %parent [4, 0] : !ttg.memdesc<8x4xf32, #shared, #smem, mutable> -> !ttg.memdesc<4x4xf32, #shared, #smem, mutable, 8x4>
    %0 = ttg.local_load %parent {test.region_name = "cta_block_a_parent"} : !ttg.memdesc<8x4xf32, #shared, #smem, mutable> -> tensor<8x4xf32>
    %1 = ttg.local_load %lower {test.region_name = "cta_block_b_lower"} : !ttg.memdesc<4x4xf32, #shared, #smem, mutable, 8x4> -> tensor<4x4xf32>
    %2 = ttg.local_load %upper {test.region_name = "cta_block_c_upper"} : !ttg.memdesc<4x4xf32, #shared, #smem, mutable, 8x4> -> tensor<4x4xf32>
    tt.return
  }
}

// -----

// Peer-CTA views can share local byte offsets while selecting different
// recipient CTAs. Runtime selection must preserve both candidates.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#blocked_sharded = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: cta_affine_a_parent vs cta_affine_a_parent: alias=true
// CHECK: cta_affine_a_parent vs cta_affine_b_local: alias=true
// CHECK: cta_affine_a_parent vs cta_affine_c_remote: alias=true
// CHECK: cta_affine_a_parent vs cta_affine_d_disjoint: alias=false
// CHECK: cta_affine_a_parent vs cta_affine_e_selected: alias=true
// CHECK: cta_affine_b_local vs cta_affine_d_disjoint: alias=false
// CHECK: cta_affine_c_remote vs cta_affine_d_disjoint: alias=false
// CHECK: cta_affine_d_disjoint vs cta_affine_e_selected: alias=false
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 1024 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @cross_cta_affine_physical_alias(%choose: i1) {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x32xi32, #shared, #smem, mutable>
    %local = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<4x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>
    %remote = ttg.memdesc_subslice %parent [2, 0] : !ttg.memdesc<4x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>
    %selected = arith.select %choose, %local, %remote : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>
    %disjoint = ttg.local_alloc {allocation.offset = 256 : i32} : () -> !ttg.memdesc<2x32xi32, #shared, #smem, mutable>
    %0 = ttg.local_load %parent {test.region_name = "cta_affine_a_parent"} : !ttg.memdesc<4x32xi32, #shared, #smem, mutable> -> tensor<4x32xi32, #blocked_sharded>
    %1 = ttg.local_load %local {test.region_name = "cta_affine_b_local"} : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32> -> tensor<2x32xi32, #blocked_sharded>
    %2 = ttg.local_load %remote {test.region_name = "cta_affine_c_remote"} : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32> -> tensor<2x32xi32, #blocked_sharded>
    %3 = ttg.local_load %disjoint {test.region_name = "cta_affine_d_disjoint"} : !ttg.memdesc<2x32xi32, #shared, #smem, mutable> -> tensor<2x32xi32, #blocked_sharded>
    %4 = ttg.local_load %selected {test.region_name = "cta_affine_e_selected"} : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32> -> tensor<2x32xi32, #blocked_sharded>
    tt.return
  }
}

// -----

// Callee allocations are relative to their own frame. Incoming descriptors
// remain disjoint from callee-local storage across direct and indirect callers.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: frame_a_argument vs frame_a_argument: alias=true
// CHECK: frame_a_argument vs frame_b_local: alias=false
// CHECK: frame_b_local vs frame_b_local: alias=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 256 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func private @callee_local_is_disjoint_from_argument(
      %incoming: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    %local = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %0 = ttg.local_load %incoming {test.region_name = "frame_a_argument"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    %1 = ttg.local_load %local {test.region_name = "frame_b_local"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }

  tt.func private @forward_to_callee(
      %incoming: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    tt.call @callee_local_is_disjoint_from_argument(%incoming) {allocation.offset = 64 : i32} : (!ttg.memdesc<16xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }

  tt.func public @call_with_disjoint_allocation_frame() {
    %incoming = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %other = ttg.local_alloc {allocation.offset = 64 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    tt.call @callee_local_is_disjoint_from_argument(%incoming) {allocation.offset = 128 : i32} : (!ttg.memdesc<16xi32, #shared, #smem, mutable>) -> ()
    tt.call @forward_to_callee(%other) {allocation.offset = 128 : i32} : (!ttg.memdesc<16xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }

  tt.func public @another_call_with_disjoint_allocation_frame() {
    %incoming = ttg.local_alloc {allocation.offset = 128 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    tt.call @callee_local_is_disjoint_from_argument(%incoming) {allocation.offset = 0 : i32} : (!ttg.memdesc<16xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

// A descriptor returned through a callee retains its caller allocation frame.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: return_frame_a_direct vs return_frame_a_direct: alias=true
// CHECK: return_frame_a_direct vs return_frame_a_returned: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: return_frame_a_direct vs return_frame_b_local: alias=false
// CHECK: return_frame_a_returned vs return_frame_b_local: alias=false
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func private @return_argument(
      %incoming: !ttg.memdesc<16xi32, #shared, #smem, mutable>)
      -> !ttg.memdesc<16xi32, #shared, #smem, mutable> {
    %local = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %0 = ttg.local_load %local {test.region_name = "return_frame_b_local"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return %incoming : !ttg.memdesc<16xi32, #shared, #smem, mutable>
  }

  tt.func public @returned_argument_preserves_allocation_frame() {
    %direct = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %returned = tt.call @return_argument(%direct) : (!ttg.memdesc<16xi32, #shared, #smem, mutable>) -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %0 = ttg.local_load %direct {test.region_name = "return_frame_a_direct"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    %1 = ttg.local_load %returned {test.region_name = "return_frame_a_returned"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }
}

// -----

// An externally supplied descriptor is unknown, not an empty physical region.
// It may alias an in-function allocation and certainly aliases itself.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: unknown_a_known vs unknown_a_known: alias=true
// CHECK: unknown_a_known vs unknown_b_external: alias=true
// CHECK: unknown_b_external vs unknown_b_external: alias=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @external_memdesc_is_unknown(%incoming: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    %known = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %0 = ttg.local_load %known {test.region_name = "unknown_a_known"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    %1 = ttg.local_load %incoming {test.region_name = "unknown_b_external"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }
}

// -----

// Extracting a runtime descriptor address is pure and must not be rejected as
// an unaccounted-for memory access.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: pure_a_descriptor vs pure_a_descriptor: alias=true
// CHECK: pure_a_descriptor vs pure_b_load: alias=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @pure_memdesc_address_is_supported() {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %address = tti.experimental_memdesc_to_i32 %buffer {test.region_name = "pure_a_descriptor"} : !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %value = ttg.local_load %buffer {test.region_name = "pure_b_load"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }
}

// -----

// Deallocation consumes a descriptor but is a free, not an unsupported read
// or write.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: deallocation_a_load vs deallocation_a_load: alias=true
// CHECK: deallocation_a_load vs deallocation_b_free: alias=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @local_deallocation_is_supported() {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %value = ttg.local_load %buffer {test.region_name = "deallocation_a_load"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    ttg.local_dealloc %buffer {test.region_name = "deallocation_b_free"} : !ttg.memdesc<16xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// AMD direct-to-LDS buffer loads have an explicit shared-memory write effect.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: amd_direct_a_buffer_load vs amd_direct_a_buffer_load: alias=true
// CHECK: amd_direct_a_buffer_load vs amd_direct_b_local_load: alias=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @amd_buffer_load_to_local_alias(%ptr: !tt.ptr<f32>, %offsets: tensor<32x64xi32, #blocked>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
    %token = amdg.buffer_load_to_local %ptr[%offsets] into %buffer {test.region_name = "amd_direct_a_buffer_load"} : <f32>[tensor<32x64xi32, #blocked>] -> <32x64xf32, #shared, #smem, mutable>
    %value = ttg.local_load %buffer {test.region_name = "amd_direct_b_local_load"} : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> tensor<32x64xf32, #blocked>
    tt.return
  }
}

// -----

// AMD packed, transposed LDS loads read the same physical bytes as an ordinary
// local load even though their result shape and layout differ.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: amd_packed_a_transposed vs amd_packed_a_transposed: alias=true
// CHECK: amd_packed_a_transposed vs amd_packed_b_local_load: alias=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 1024 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @amd_packed_transposed_local_load_alias() {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x64xi8, #shared, #smem, mutable>
    %packed = amdg.local_load_packed_transposed %buffer {test.region_name = "amd_packed_a_transposed"} : !ttg.memdesc<16x64xi8, #shared, #smem, mutable> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    %plain = ttg.local_load %buffer {test.region_name = "amd_packed_b_local_load"} : !ttg.memdesc<16x64xi8, #shared, #smem, mutable> -> tensor<16x64xi8>
    tt.return
  }
}

// -----

// AMD asynchronous LDS-to-global copies read their source descriptor.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: amd_store_a_async_copy vs amd_store_a_async_copy: alias=true
// CHECK: amd_store_a_async_copy vs amd_store_b_local_load: alias=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @amd_async_copy_local_to_global_alias(%dst: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %token = amdg.async_copy_local_to_global %buffer, %dst {test.region_name = "amd_store_a_async_copy"} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %value = ttg.local_load %buffer {test.region_name = "amd_store_b_local_load"} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

// A fused TDM load writes every variadic destination, not just the first one.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: amd_fused_a_first vs amd_fused_a_first: alias=true
// CHECK: amd_fused_a_first vs amd_fused_b_second: alias=false
// CHECK: amd_fused_b_second vs amd_fused_b_second: alias=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16384 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @amd_fused_tdm_variadic_destination_alias(%desc0: !tt.tensordesc<64x64xf16, #shared>, %desc1: !tt.tensordesc<64x64xf16, #shared>) {
    %first = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %second = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %token = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %first, %second {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %first_value = ttg.local_load %first {test.region_name = "amd_fused_a_first"} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked>
    %second_value = ttg.local_load %second {test.region_name = "amd_fused_b_second"} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked>
    tt.return
  }
}

// -----

// A non-power-of-two memdesc uses a power-of-two LinearLayout, but its exact
// footprint must contain only the logical coordinates. A flattened Gray-code
// prefix for the normalized 64x256 layout misses byte 2048 and incorrectly
// includes byte 12288, the first byte past the packed 48x256 allocation.
#nonpow2 = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [32, 0]]}, alignment = 16>
#byte = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: nonpow2_a_footprint vs nonpow2_a_footprint: alias=true
// CHECK: nonpow2_a_footprint vs nonpow2_b_valid_byte: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=false
// CHECK: nonpow2_a_footprint vs nonpow2_c_past_end: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 12289 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @non_power_of_two_exact_footprint() {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<48x256xi8, #nonpow2, #smem, mutable>
    %valid_byte = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi8, #byte, #smem, mutable>
    %past_end = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi8, #byte, #smem, mutable>
    %0 = ttg.local_load %buffer {test.region_name = "nonpow2_a_footprint"} : !ttg.memdesc<48x256xi8, #nonpow2, #smem, mutable> -> tensor<48x256xi8>
    %1 = ttg.local_load %valid_byte {test.region_name = "nonpow2_b_valid_byte"} : !ttg.memdesc<1xi8, #byte, #smem, mutable> -> tensor<1xi8>
    %2 = ttg.local_load %past_end {test.region_name = "nonpow2_c_past_end"} : !ttg.memdesc<1xi8, #byte, #smem, mutable> -> tensor<1xi8>
    tt.return
  }
}

// -----

// A warp-group wait forwards each memory descriptor without accessing or
// replacing the allocation it keeps alive.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: wait_a_source vs wait_a_source: alias=true
// CHECK: wait_a_source vs wait_b_result: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: wait_a_source vs wait_c_load: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: wait_b_result vs wait_c_load: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @warp_group_wait_preserves_memdesc() {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %source = ttg.local_load %buffer {test.region_name = "wait_a_source"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    %waited = ttng.warp_group_dot_wait %buffer {pendings = 0 : i32, test.region_name = "wait_b_result"} : !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %result = ttg.local_load %waited {test.region_name = "wait_c_load"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }
}

// -----

// Unknown memory aliases every exact state while retaining one additional
// state lane for hazards between independently unknown descriptors.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: wildcard_a_known vs wildcard_a_known: alias=true
// CHECK: wildcard_a_known vs wildcard_b_incoming: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false
// CHECK: wildcard_b_incoming vs wildcard_b_incoming: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false
// CHECK: wildcard_a_known case [0, 64]: mask={0}
// CHECK: wildcard_b_incoming case unknown: mask={0,1}
// CHECK: state-plan: lanes=2
module attributes {test.print_state_plan, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @unknown_descriptor_state_masks(%incoming: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    %known = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %0 = ttg.local_load %known {test.region_name = "wildcard_a_known"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    %1 = ttg.local_load %incoming {test.region_name = "wildcard_b_incoming"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }
}

// -----

// An unknown-only module still has one state lane and cannot conclude that
// two external descriptors are disjoint.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: unknown_only_a_first vs unknown_only_a_first: alias=true
// CHECK: unknown_only_a_first vs unknown_only_b_second: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false
// CHECK: unknown_only_a_first case unknown: mask={0}
// CHECK: unknown_only_b_second case unknown: mask={0}
// CHECK: state-plan: lanes=1
module attributes {test.print_state_plan, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @unknown_only_descriptor_state_masks(%first: !ttg.memdesc<16xi32, #shared, #smem, mutable>, %second: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    %0 = ttg.local_load %first {test.region_name = "unknown_only_a_first"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    %1 = ttg.local_load %second {test.region_name = "unknown_only_b_second"} : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }
}

// -----

// Physical CTA identity separates peer-CTA aliases without duplicating state
// lanes for the same projected physical bytes.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: cta_state_a_local vs cta_state_a_local: alias=true
// CHECK: cta_state_a_local vs cta_state_b_remote: alias=false
// CHECK: cta_state_a_local case {{.*}}: mask={0}
// CHECK: cta_state_b_remote case {{.*}}: mask={0}
// CHECK: state-plan: lanes=1
module attributes {test.print_state_plan, "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @peer_ctas_share_projected_state_lane() {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x32xi32, #shared, #smem, mutable>
    %local = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<4x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>
    %remote = ttg.memdesc_subslice %parent [2, 0] : !ttg.memdesc<4x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>
    %0 = ttg.local_load %local {test.region_name = "cta_state_a_local"} : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32> -> tensor<2x32xi32, #blocked>
    %1 = ttg.local_load %remote {test.region_name = "cta_state_b_remote"} : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32> -> tensor<2x32xi32, #blocked>
    tt.return
  }
}

// -----

// TDM gather and scatter expose the exact shared-memory destination and
// source in their declared memory effects.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#idx_parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: amd_tdm_a_gather vs amd_tdm_a_gather: alias=true
// CHECK: amd_tdm_a_gather vs amd_tdm_b_scatter: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true
// CHECK: amd_tdm_a_gather vs amd_tdm_c_other: alias=false
// CHECK: amd_tdm_b_scatter vs amd_tdm_c_other: alias=false
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 131072 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @amd_tdm_gather_scatter_shared_alias(%desc: !tt.tensordesc<64x128xf16>, %rows: tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %other = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %gather = amdg.async_tdm_gather %desc[%rows] to %buffer {test.region_name = "amd_tdm_a_gather"} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>, !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    %scatter = amdg.async_tdm_scatter %desc[%rows] from %buffer {test.region_name = "amd_tdm_b_scatter"} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>, !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    %value = ttg.local_load %other {test.region_name = "amd_tdm_c_other"} : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blocked>
    tt.return
  }
}
