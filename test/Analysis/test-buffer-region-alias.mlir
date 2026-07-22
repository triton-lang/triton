// RUN: triton-opt %s -split-input-file -mlir-disable-threading -test-buffer-region-alias -verify-diagnostics -o /dev/null

// expected-remark @below {{exhaustive AddressSet oracle passed: 65536 ordered pairs}}
module attributes {test.exhaustive_address_sets} {
}

// -----

// The full/even/odd family has three views but only two physical membership
// atoms. All masks collapse to the exact atoms touched by each access.
// expected-remark @below {{state-plan: lanes=2, components=atoms(2)}}
// expected-remark @below {{a_full case [0, 8]: mask={0,1}}}
// expected-remark @below {{b_even case [0, 7]: mask={0}}}
// expected-remark @below {{c_odd case [1, 7]: mask={1}}}
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

// Partially overlapping windows require three physical atoms. Keeping the
// shared atom exact is necessary when a completion barrier publishes proxy
// fence state for only one of the two windows.
// expected-remark @below {{state-plan: lanes=3, components=atoms(3)}}
// expected-remark @below {{a_left case [0, 2]: mask={0,1}}}
// expected-remark @below {{b_right case [1, 2]: mask={1,2}}}
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

#padded = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [16, 16]}>
#smem = #ttg.shared_memory

// A padded page occupies 1152 bytes, while the two nested nonzero subslices
// compose to an unpadded affine offset of 544 bytes. Dynamic indexing and
// select must preserve both exact candidates and their runtime identities.
// expected-remark @below {{a_page0 vs a_page0: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_page0 vs b_page1: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{a_page0 vs c_nested0: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=false}}
// expected-remark @below {{a_page0 vs d_nested1: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{a_page0 vs e_complement1: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{a_page0 vs f_dynamic: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{a_page0 vs g_selected: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{b_page1 vs b_page1: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{b_page1 vs c_nested0: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{b_page1 vs d_nested1: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=false}}
// expected-remark @below {{b_page1 vs e_complement1: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=false}}
// expected-remark @below {{b_page1 vs f_dynamic: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{b_page1 vs g_selected: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{c_nested0 vs c_nested0: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{c_nested0 vs d_nested1: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{c_nested0 vs e_complement1: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{c_nested0 vs f_dynamic: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{c_nested0 vs g_selected: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{d_nested1 vs d_nested1: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{d_nested1 vs e_complement1: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{d_nested1 vs f_dynamic: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{d_nested1 vs g_selected: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true}}
// expected-remark @below {{e_complement1 vs e_complement1: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{e_complement1 vs f_dynamic: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{e_complement1 vs g_selected: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{f_dynamic vs f_dynamic: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{f_dynamic vs g_selected: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{g_selected vs g_selected: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_page0 case [0, 1024]: mask={0,1}}}
// expected-remark @below {{b_page1 case [1152, 1024]: mask={2,3,4}}}
// expected-remark @below {{c_nested0 case [544, 256]: mask={1}}}
// expected-remark @below {{d_nested1 case [1696, 256]: mask={4}}}
// expected-remark @below {{e_complement1 case [1664, 256]: mask={3}}}
// expected-remark @below {{f_dynamic case [544, 256]: mask={1}}}
// expected-remark @below {{f_dynamic case [1696, 256]: mask={4}}}
// expected-remark @below {{g_selected case [544, 256]: mask={1}}}
// expected-remark @below {{g_selected case [1696, 256]: mask={4}}}
// expected-remark @below {{state-plan: lanes=5, components=atoms(2), atoms(3)}}
module attributes {test.print_state_plan, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 4096 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @padded_dynamic_nested(%idx: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %multi = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x16x16xf32, #padded, #smem, mutable>
    %dynamic = ttg.memdesc_index %multi[%idx] : !ttg.memdesc<2x16x16xf32, #padded, #smem, mutable> -> !ttg.memdesc<16x16xf32, #padded, #smem, mutable>
    %page0 = ttg.memdesc_index %multi[%c0] : !ttg.memdesc<2x16x16xf32, #padded, #smem, mutable> -> !ttg.memdesc<16x16xf32, #padded, #smem, mutable>
    %page1 = ttg.memdesc_index %multi[%c1] : !ttg.memdesc<2x16x16xf32, #padded, #smem, mutable> -> !ttg.memdesc<16x16xf32, #padded, #smem, mutable>
    %dynamic_row = ttg.memdesc_subslice %dynamic [8, 0] : !ttg.memdesc<16x16xf32, #padded, #smem, mutable> -> !ttg.memdesc<8x16xf32, #padded, #smem, mutable, 16x16>
    %dynamic_nested = ttg.memdesc_subslice %dynamic_row [0, 8] : !ttg.memdesc<8x16xf32, #padded, #smem, mutable, 16x16> -> !ttg.memdesc<8x8xf32, #padded, #smem, mutable, 16x16>
    %row0 = ttg.memdesc_subslice %page0 [8, 0] : !ttg.memdesc<16x16xf32, #padded, #smem, mutable> -> !ttg.memdesc<8x16xf32, #padded, #smem, mutable, 16x16>
    %nested0 = ttg.memdesc_subslice %row0 [0, 8] : !ttg.memdesc<8x16xf32, #padded, #smem, mutable, 16x16> -> !ttg.memdesc<8x8xf32, #padded, #smem, mutable, 16x16>
    %row1 = ttg.memdesc_subslice %page1 [8, 0] : !ttg.memdesc<16x16xf32, #padded, #smem, mutable> -> !ttg.memdesc<8x16xf32, #padded, #smem, mutable, 16x16>
    %nested1 = ttg.memdesc_subslice %row1 [0, 8] : !ttg.memdesc<8x16xf32, #padded, #smem, mutable, 16x16> -> !ttg.memdesc<8x8xf32, #padded, #smem, mutable, 16x16>
    %complement1 = ttg.memdesc_subslice %row1 [0, 0] : !ttg.memdesc<8x16xf32, #padded, #smem, mutable, 16x16> -> !ttg.memdesc<8x8xf32, #padded, #smem, mutable, 16x16>
    %selected = arith.select %cond, %dynamic_nested, %nested1 : !ttg.memdesc<8x8xf32, #padded, #smem, mutable, 16x16>
    %0 = ttg.local_load %page0 {test.region_name = "a_page0"} : !ttg.memdesc<16x16xf32, #padded, #smem, mutable> -> tensor<16x16xf32>
    %1 = ttg.local_load %page1 {test.region_name = "b_page1"} : !ttg.memdesc<16x16xf32, #padded, #smem, mutable> -> tensor<16x16xf32>
    %2 = ttg.local_load %nested0 {test.region_name = "c_nested0"} : !ttg.memdesc<8x8xf32, #padded, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %3 = ttg.local_load %nested1 {test.region_name = "d_nested1"} : !ttg.memdesc<8x8xf32, #padded, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %4 = ttg.local_load %complement1 {test.region_name = "e_complement1"} : !ttg.memdesc<8x8xf32, #padded, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %5 = ttg.local_load %dynamic_nested {test.region_name = "f_dynamic"} : !ttg.memdesc<8x8xf32, #padded, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %6 = ttg.local_load %selected {test.region_name = "g_selected"} : !ttg.memdesc<8x8xf32, #padded, #smem, mutable, 16x16> -> tensor<8x8xf32>
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

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// A shrinking reinterpret only covers the first half of its parent. It must
// not inherit the parent's footprint and alias the untouched second half.
// expected-remark @below {{a_reinterpreted_first_half vs a_reinterpreted_first_half: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_reinterpreted_first_half vs b_upper_half: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{b_upper_half vs b_upper_half: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_reinterpreted_first_half case [0, 32]: mask={0}}}
// expected-remark @below {{b_upper_half case [32, 32]: mask={1}}}
// expected-remark @below {{state-plan: lanes=2, components=atoms(1), atoms(1)}}
module attributes {test.print_state_plan, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @shared_shrinking_reinterpret() {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %smaller = ttg.memdesc_reinterpret %parent : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> !ttg.memdesc<8xi32, #shared, #smem, mutable>
    %upper = ttg.memdesc_subslice %parent [8] : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> !ttg.memdesc<8xi32, #shared, #smem, mutable, 16>
    %0 = ttg.local_load %smaller {test.region_name = "a_reinterpreted_first_half"} : !ttg.memdesc<8xi32, #shared, #smem, mutable> -> tensor<8xi32>
    %1 = ttg.local_load %upper {test.region_name = "b_upper_half"} : !ttg.memdesc<8xi32, #shared, #smem, mutable, 16> -> tensor<8xi32>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// Changing f32 to f16 shrinks the live TMEM columns by half, leaving the
// upper f32 subslice physically disjoint from the reinterpreted view.
// expected-remark @below {{a_reinterpreted_first_half vs a_reinterpreted_first_half: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_reinterpreted_first_half vs b_upper_half: alias=false, lhs_contains_rhs=false, rhs_contains_lhs=false}}
// expected-remark @below {{b_upper_half vs b_upper_half: alias=true, lhs_contains_rhs=true, rhs_contains_lhs=true}}
// expected-remark @below {{a_reinterpreted_first_half case [0, 64]: mask={0}}}
// expected-remark @below {{b_upper_half case [64, 64]: mask={1}}}
// expected-remark @below {{state-plan: lanes=2, components=atoms(1), atoms(1)}}
module attributes {test.print_state_plan, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tensor_shrinking_reinterpret() {
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %smaller = ttg.memdesc_reinterpret %parent : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %upper = ttng.tmem_subslice %parent {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    %0 = ttng.tmem_load %smaller {test.region_name = "a_reinterpreted_first_half"} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf16>
    %1 = ttng.tmem_load %upper {test.region_name = "b_upper_half"} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32>
    tt.return
  }
}
