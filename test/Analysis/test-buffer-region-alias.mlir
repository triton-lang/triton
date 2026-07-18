// RUN: triton-opt %s -split-input-file -mlir-disable-threading -test-buffer-region-alias -verify-diagnostics -o /dev/null

// expected-remark @below {{exhaustive AddressSet oracle passed: 65536 ordered pairs}}
module attributes {test.exhaustive_address_sets} {
}

// -----

// The full/even/odd family has three views but only two physical membership
// atoms. All masks collapse to the exact atoms touched by each access.
// expected-remark @below {{state-plan: lanes=2, components=atoms(2)}}
// expected-remark @below {{a_full case [0, 8]: update={0,1}, check={0,1}, complete={0,1}}}
// expected-remark @below {{b_even case [0, 7]: update={0}, check={0}, complete={0}}}
// expected-remark @below {{c_odd case [1, 7]: update={1}, check={1}, complete={1}}}
module attributes {test.print_state_plan, test.state_plan_only} {
  tt.func private @full() attributes {
    test.region_name = "a_full",
    test.region_base = 0 : i32,
    test.region_length = 8 : i32,
    test.region_addresses = array<i32: 0, 1, 2, 3, 4, 5, 6, 7>
  }
  tt.func private @even() attributes {
    test.region_name = "b_even",
    test.region_addresses = array<i32: 0, 2, 4, 6>
  }
  tt.func private @odd() attributes {
    test.region_name = "c_odd",
    test.region_addresses = array<i32: 1, 3, 5, 7>
  }
}

// -----

// Partially overlapping windows would form three physical atoms, so the plan
// retains two view lanes and expands only the check masks.
// expected-remark @below {{state-plan: lanes=2, components=views(2)}}
// expected-remark @below {{a_left case [0, 2]: update={0}, check={0,1}, complete={0}}}
// expected-remark @below {{b_right case [1, 2]: update={1}, check={0,1}, complete={1}}}
module attributes {test.print_state_plan, test.state_plan_only} {
  tt.func private @left() attributes {
    test.region_name = "a_left",
    test.region_addresses = array<i32: 0, 1>
  }
  tt.func private @right() attributes {
    test.region_name = "b_right",
    test.region_addresses = array<i32: 1, 2>
  }
}

// -----

// Direct address sets isolate the relation algebra from layout construction.
// These cover aligned, adjacent, overlapping, contained, and non-power-of-two
// regions.
// expected-remark @below {{a_aligned vs a_aligned: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_aligned vs b_adjacent: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{a_aligned vs c_overlap: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{a_aligned vs d_contained: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=false}}
// expected-remark @below {{b_adjacent vs b_adjacent: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{b_adjacent vs c_overlap: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{b_adjacent vs d_contained: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{c_overlap vs c_overlap: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{c_overlap vs d_contained: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{d_contained vs d_contained: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
module {
  tt.func private @a() attributes {
    test.region_name = "a_aligned",
    test.region_addresses = array<i32: 0, 1, 2, 3>
  }
  tt.func private @b() attributes {
    test.region_name = "b_adjacent",
    test.region_addresses = array<i32: 4, 5, 6>
  }
  tt.func private @c() attributes {
    test.region_name = "c_overlap",
    test.region_addresses = array<i32: 3, 4, 5>
  }
  tt.func private @d() attributes {
    test.region_name = "d_contained",
    test.region_addresses = array<i32: 1, 2>
  }
}

// -----

// Empty and unaligned sparse sets exercise identities that are easy to get
// wrong when canonicalizing a union of ranges.
// expected-remark @below {{a_empty vs a_empty: alias=false, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_empty vs b_non_power_two: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{a_empty vs c_unaligned_sparse: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{b_non_power_two vs b_non_power_two: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{b_non_power_two vs c_unaligned_sparse: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=false}}
// expected-remark @below {{c_unaligned_sparse vs c_unaligned_sparse: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
module {
  tt.func private @empty() attributes {
    test.region_name = "a_empty",
    test.region_addresses = array<i32>
  }
  tt.func private @non_power_two() attributes {
    test.region_name = "b_non_power_two",
    test.region_addresses = array<i32: 10, 11, 12>
  }
  tt.func private @unaligned_sparse() attributes {
    test.region_name = "c_unaligned_sparse",
    test.region_addresses = array<i32: 11, 12>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// expected-remark @below {{adjacent_a vs adjacent_a: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{adjacent_a vs adjacent_b: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{adjacent_a vs overlap: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{adjacent_b vs adjacent_b: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{adjacent_b vs overlap: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{overlap vs overlap: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @smem_intervals() {
    %a = ttg.local_alloc {allocation.offset = 0 : i32, test.region_name = "adjacent_a"} : () -> !ttg.memdesc<8xi8, #shared, #smem, mutable>
    %b = ttg.local_alloc {allocation.offset = 8 : i32, test.region_name = "adjacent_b"} : () -> !ttg.memdesc<8xi8, #shared, #smem, mutable>
    %overlap = ttg.local_alloc {allocation.offset = 4 : i32, test.region_name = "overlap"} : () -> !ttg.memdesc<8xi8, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#plain_2d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// All four logical quadrants are physically disjoint. The full view overlaps
// and contains every quadrant, guarding each negative result against false
// negatives.
// expected-remark @below {{a_q00 vs a_q00: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_q00 vs b_q01: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{a_q00 vs c_q10: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{a_q00 vs d_q11: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{a_q00 vs e_full: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{b_q01 vs b_q01: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{b_q01 vs c_q10: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{b_q01 vs d_q11: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{b_q01 vs e_full: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{c_q10 vs c_q10: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{c_q10 vs d_q11: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{c_q10 vs e_full: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{d_q11 vs d_q11: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{d_q11 vs e_full: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{e_full vs e_full: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 2048 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @plain_quadrants() {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x16xf32, #plain_2d, #smem, mutable>
    %q00 = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x16xf32, #plain_2d, #smem, mutable> -> !ttg.memdesc<8x8xf32, #plain_2d, #smem, mutable, 16x16>
    %q01 = ttg.memdesc_subslice %parent [0, 8] : !ttg.memdesc<16x16xf32, #plain_2d, #smem, mutable> -> !ttg.memdesc<8x8xf32, #plain_2d, #smem, mutable, 16x16>
    %q10 = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x16xf32, #plain_2d, #smem, mutable> -> !ttg.memdesc<8x8xf32, #plain_2d, #smem, mutable, 16x16>
    %q11 = ttg.memdesc_subslice %parent [8, 8] : !ttg.memdesc<16x16xf32, #plain_2d, #smem, mutable> -> !ttg.memdesc<8x8xf32, #plain_2d, #smem, mutable, 16x16>
    %0 = ttg.local_load %q00 {test.region_name = "a_q00"} : !ttg.memdesc<8x8xf32, #plain_2d, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %1 = ttg.local_load %q01 {test.region_name = "b_q01"} : !ttg.memdesc<8x8xf32, #plain_2d, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %2 = ttg.local_load %q10 {test.region_name = "c_q10"} : !ttg.memdesc<8x8xf32, #plain_2d, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %3 = ttg.local_load %q11 {test.region_name = "d_q11"} : !ttg.memdesc<8x8xf32, #plain_2d, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %4 = ttg.local_load %parent {test.region_name = "e_full"} : !ttg.memdesc<16x16xf32, #plain_2d, #smem, mutable> -> tensor<16x16xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Dynamic memdesc_index and select values carry two exact candidates. The
// nested subslice must preserve each candidate's storage base.
// expected-remark @below {{a_first vs a_first: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_first vs b_second: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{a_first vs c_indexed: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{a_first vs d_selected: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{b_second vs b_second: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{b_second vs c_indexed: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{b_second vs d_selected: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{c_indexed vs c_indexed: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{c_indexed vs d_selected: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{d_selected vs d_selected: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @dynamic_candidates(%idx: i32, %cond: i1) {
    %multi = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x16xi8, #shared, #smem, mutable>
    %indexed_page = ttg.memdesc_index %multi[%idx] : !ttg.memdesc<2x16xi8, #shared, #smem, mutable> -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    %indexed = ttg.memdesc_subslice %indexed_page [0] : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> !ttg.memdesc<8xi8, #shared, #smem, mutable, 16>
    %first = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8xi8, #shared, #smem, mutable>
    %second = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<8xi8, #shared, #smem, mutable>
    %selected = arith.select %cond, %first, %second : !ttg.memdesc<8xi8, #shared, #smem, mutable>
    %0 = ttg.local_load %first {test.region_name = "a_first"} : !ttg.memdesc<8xi8, #shared, #smem, mutable> -> tensor<8xi8>
    %1 = ttg.local_load %second {test.region_name = "b_second"} : !ttg.memdesc<8xi8, #shared, #smem, mutable> -> tensor<8xi8>
    %2 = ttg.local_load %indexed {test.region_name = "c_indexed"} : !ttg.memdesc<8xi8, #shared, #smem, mutable, 16> -> tensor<8xi8>
    %3 = ttg.local_load %selected {test.region_name = "d_selected"} : !ttg.memdesc<8xi8, #shared, #smem, mutable> -> tensor<8xi8>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// A constant memdesc_index carries only the selected page. This keeps the
// common static multibuffer path out of runtime candidate selection.
// expected-remark @below {{a_first vs a_first: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_first vs b_second: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{b_second vs b_second: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 32 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @constant_index_candidate() {
    %c1 = arith.constant 1 : i32
    %multi = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x16xi8, #shared, #smem, mutable>
    %second = ttg.memdesc_index %multi[%c1] : !ttg.memdesc<2x16xi8, #shared, #smem, mutable> -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    %first = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    %0 = ttg.local_load %first {test.region_name = "a_first"} : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8>
    %1 = ttg.local_load %second {test.region_name = "b_second"} : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// A 256x64 M-slice occupies alternating 64-row bands in a 128-row TMEM
// layout. The descriptor intervals overlap, but the encoded word sets do not.
// expected-remark @below {{page0_slab0 vs page0_slab0: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{page0_slab0 vs page0_slab1: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{page0_slab0 vs page1_slab0: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{page0_slab0 vs page1_slab1: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{page0_slab1 vs page0_slab1: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{page0_slab1 vs page1_slab0: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{page0_slab1 vs page1_slab1: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{page1_slab0 vs page1_slab0: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{page1_slab0 vs page1_slab1: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{page1_slab1 vs page1_slab1: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 512 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tmem_two_page_two_slab() {
    %page0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %page1 = ttng.tmem_alloc {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %page0_slab0 = ttng.tmem_subslice %page0 {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %page0_slab1 = ttng.tmem_subslice %page0 {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %page1_slab0 = ttng.tmem_subslice %page1 {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %page1_slab1 = ttng.tmem_subslice %page1 {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %0 = ttng.tmem_load %page0_slab0 {test.region_name = "page0_slab0"} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<256x64xf32>
    %1 = ttng.tmem_load %page0_slab1 {test.region_name = "page0_slab1"} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<256x64xf32>
    %2 = ttng.tmem_load %page1_slab0 {test.region_name = "page1_slab0"} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<256x64xf32>
    %3 = ttng.tmem_load %page1_slab1 {test.region_name = "page1_slab1"} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<256x64xf32>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// The full page and slab have the same runtime descriptor key ([0, 128]) but
// distinct exact footprints. Candidate identity must preserve that distinction.
// expected-remark @below {{full vs full: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{full vs slab: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=false}}
// expected-remark @below {{slab vs slab: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tmem_same_runtime_key() {
    %tm = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %slab = ttng.tmem_subslice %tm {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    %0 = ttng.tmem_load %tm {test.region_name = "full"} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32>
    %1 = ttng.tmem_load %slab {test.region_name = "slab"} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<256x64xf32>
    tt.return
  }
}
