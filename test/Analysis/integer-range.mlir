// RUN: triton-opt %s -test-print-integer-range -verify-diagnostics=only-expected -o /dev/null
// RUN: triton-opt %s -test-tritonamdgpu-range-analysis -verify-diagnostics=only-expected -o /dev/null

// The same endpoints must be available to the upstream and AMD analyses.
tt.func @make_range() {
  // expected-remark@+1 {{unsigned : [0, 127] signed : [0, 127]}}
  %positive = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  // expected-remark@+1 {{unsigned : [16, 31] signed : [16, 31]}}
  %offset = tt.make_range {start = 16 : i32, end = 32 : i32} : tensor<16xi32>
  // expected-remark@+1 {{unsigned : [7, 7] signed : [7, 7]}}
  %single = tt.make_range {start = 7 : i32, end = 8 : i32} : tensor<1xi32>
  // expected-remark@+1 {{unsigned : [4294967168, 4294967295] signed : [-128, -1]}}
  %negative = tt.make_range {start = -128 : i32, end = 0 : i32} : tensor<128xi32>
  // A signed interval crossing zero needs the full unsigned enclosure.
  // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-64, 63]}}
  %cross_zero = tt.make_range {start = -64 : i32, end = 64 : i32} : tensor<128xi32>
  // expected-remark@+1 {{unsigned : [2147483648, 2147483649] signed : [-2147483648, -2147483647]}}
  %min = tt.make_range {start = -2147483648 : i32, end = -2147483646 : i32} : tensor<2xi32>
  // The exclusive end is itself signed i32, so INT_MAX is not an element.
  // expected-remark@+1 {{unsigned : [2147483645, 2147483646] signed : [2147483645, 2147483646]}}
  %max = tt.make_range {start = 2147483645 : i32, end = 2147483647 : i32} : tensor<2xi32>
  tt.return
}

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @encoded_make_range() {
    // expected-remark@+1 {{unsigned : [0, 127] signed : [0, 127]}}
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    tt.return
  }
}
