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
