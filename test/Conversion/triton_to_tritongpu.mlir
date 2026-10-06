// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:80 num-warps=2' | FileCheck %s
// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:80 num-warps=2 num-ctas=2' | FileCheck %s --check-prefixes=CHECK-TWO-CTAS

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
tt.func @ops() {
  // CHECK: module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {{.*}}
  %a = arith.constant dense<1.00e+00> : tensor<128x32xf16>
  %b = arith.constant dense<2.00e+00> : tensor<32x128xf16>
  %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
  %0 = tt.dot %a, %b, %c : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
  tt.return
}
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
tt.func @load_ops(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // Test if LoadOp is lowered properly (see #771)
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<128xi1>
  %other = arith.constant dense<0.0e+0> : tensor<128xf32>
  // CHECK: %{{.*}} = tt.load %{{.*}} : {{.*}}
  %a = tt.load %ptrs : tensor<128x!tt.ptr<f32>>
  // CHECK: %{{.*}} = tt.load %{{.*}}, %{{.*}} : {{.*}}
  %b = tt.load %ptrs, %mask : tensor<128x!tt.ptr<f32>>
  // CHECK: %{{.*}} = tt.load %{{.*}}, %{{.*}}, %{{.*}} : {{.*}}
  %c = tt.load %ptrs, %mask, %other : tensor<128x!tt.ptr<f32>>
  tt.store %ptrs, %a : tensor<128x!tt.ptr<f32>>
  tt.store %ptrs, %b : tensor<128x!tt.ptr<f32>>
  tt.store %ptrs, %c : tensor<128x!tt.ptr<f32>>
  tt.return
}
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
tt.func @reduce_ops(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // Test if the total number of threadsPerWarp is 32
  // Test if the total number of warps is 2
  // CHECK: #[[blocked0:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
  // CHECK: #[[blocked1:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 1], order = [1, 0]}>
  // CHECK: #[[blocked2:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
  // CHECK: module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {{.*}}
  %c0 = arith.constant dense<1.00e+00> : tensor<4x4xf32>
  %c1 = arith.constant dense<2.00e+00> : tensor<8x2xf32>
  %c2 = arith.constant dense<3.00e+00> : tensor<16x16xf32>
  // CHECK: (tensor<4x4xf32, #[[blocked0]]>) -> tensor<4xf32, #ttg.slice<{dim = 0, parent = #[[blocked0]]}>>
  %c0_ = "tt.reduce" (%c0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %add = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32} : (tensor<4x4xf32>) -> tensor<4xf32>
  // CHECK: (tensor<8x2xf32, #[[blocked1]]>) -> tensor<2xf32, #ttg.slice<{dim = 0, parent = #[[blocked1]]}>
  %c1_ = "tt.reduce" (%c1) ({
  ^bb0(%arg3: f32, %arg4: f32):
    %add = arith.addf %arg3, %arg4 : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32} : (tensor<8x2xf32>) -> tensor<2xf32>
  // CHECK: (tensor<8x2xf32, #[[blocked1]]>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #[[blocked1]]}>>
  %c2_ = "tt.reduce" (%c1) ({
  ^bb0(%arg5: f32, %arg6: f32):
    %add = arith.addf %arg5, %arg6 : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32} : (tensor<8x2xf32>) -> tensor<8xf32>
  // CHECK: (tensor<16x16xf32, #[[blocked2]]>) -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #[[blocked2]]}>>
  %c3_ = "tt.reduce" (%c2) ({
  ^bb0(%arg7: f32, %arg8: f32):
    %add = arith.addf %arg7, %arg8 : f32
    tt.reduce.return %add : f32
  }) {axis = 0 : i32} : (tensor<16x16xf32>) -> tensor<16xf32>

  tt.return
}
}


// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
tt.func public @select_op(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i1) {
  // CHECK-LABEL: select_op
  %cst = arith.constant dense<0.000000e+00> : tensor<128xf32>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %3 = tt.load %2 : tensor<128x!tt.ptr<f32>>

  // CHECK: %{{.*}} = arith.select %arg2, %{{.*}}, %{{.*}} : tensor<128xf32, #blocked>
  %4 = arith.select %arg2, %cst, %3 : tensor<128xf32>

  %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  tt.store %6, %4 : tensor<128x!tt.ptr<f32>>
  tt.return
}
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
tt.func @arith_splat_bool(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK-LABEL: arith_splat_bool

  // Test arith.constant with splatted bool.
  // CHECK-NEXT: arith.constant dense<true> : tensor<128xi1, #{{.*}}>
  %mask = arith.constant dense<true> : tensor<128xi1>
  tt.return
}
}

// -----

// CHECK-LABEL: gather_op
tt.func @gather_op() {
  %cst = arith.constant dense<1.0> : tensor<128x4xf32>
  %cst_0 = arith.constant dense<1> : tensor<256x4xi32>
  // CHECK: tt.gather %{{.*}}[%{{.*}}] {axis = 0 : i32} : (tensor<128x4xf32, #blocked>, tensor<256x4xi32, #blocked>) -> tensor<256x4xf32, #blocked>
  %0 = tt.gather %cst[%cst_0] {axis = 0 : i32} : (tensor<128x4xf32>, tensor<256x4xi32>) -> tensor<256x4xf32>
  tt.return
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#bar_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// CHECK: [[SLICE_PARENT:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [1, 0]}>

// CHECK: @gather4_layout
tt.func @gather4_layout(%arg0: !tt.tensordesc<1x128xf32>, %arg1: i32, %arg2: !tt.ptr<f32>) {
  %cst = arith.constant dense<1> : tensor<32xi32>
  // CHECK: [[IDX:%.*]] = ttg.convert_layout %cst : tensor<32xi32, #{{.*}}> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = [[SLICE_PARENT]]}>>
  %0 = tt.descriptor_gather %arg0[%cst, %arg1] : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>
  %1 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>>
  tt.store %1, %0 : tensor<32x128x!tt.ptr<f32>>
  tt.return
}

// CHECK: @scatter4_layout
tt.func @scatter4_layout(%arg0: !tt.tensordesc<1x128xf32>, %arg1: i32, %arg2: !tt.ptr<f32>) {
  %cst = arith.constant dense<1> : tensor<32xi32>
  %0 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>>
  %1 = tt.load %0 : tensor<32x128x!tt.ptr<f32>>
  // CHECK: [[IDX:%.*]] = ttg.convert_layout %cst : tensor<32xi32, #{{.*}}> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = [[SLICE_PARENT]]}>>
  tt.descriptor_scatter %arg0[%cst, %arg1], %1 : !tt.tensordesc<1x128xf32>, tensor<32xi32>, i32, tensor<32x128xf32>
  tt.return
}

// -----

// CHECK-LABEL: @ub_poison
tt.func @ub_poison() {
  // CHECK-NEXT: ub.poison : tensor<128x64xf16, #blocked>
  %0 = ub.poison : tensor<128x64xf16>
  tt.return
}

// -----

// CHECK-LABEL: @cf_br
tt.func @cf_br(%ptr: !tt.ptr<i32>) {
  %cst = arith.constant dense<1> : tensor<128xi32>
  // cf.br ^bb1(%{{.+}} : tensor<128xi32, #{{.+}}>)
  cf.br ^bb1(%cst : tensor<128xi32>)
^bb1(%arg0: tensor<128xi32>):
  %ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  tt.store %ptrs, %arg0 : tensor<128x!tt.ptr<i32>>
  tt.return
}

// -----

tt.func @split_op(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
  // CHECK-TWO-CTAS-LABEL: split_op
  // CHECK-TWO-CTAS: tt.split
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x2x!tt.ptr<f32>>
  %1 = tt.load %0 : tensor<64x2x!tt.ptr<f32>>
  %res1, %res2 = tt.split %1 : tensor<64x2xf32> -> tensor<64xf32>
  %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  tt.store %2, %res1 : tensor<64x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: tt.func private @callee
// CHECK-SAME: (%{{.*}}: !tt.ptr<i32>) -> tensor<128xi32, #{{.*}}>
tt.func private @callee(%arg0: !tt.ptr<i32>) -> tensor<128xi32> {
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: tt.return %{{.*}} : tensor<128xi32, #{{.*}}>
  tt.return %0 : tensor<128xi32>
}

// CHECK-LABEL: tt.func @caller
tt.func @caller(%ptr: !tt.ptr<i32>) {
  // CHECK: %{{.*}} = tt.call @callee(%{{.*}}) : (!tt.ptr<i32>) -> tensor<128xi32, #{{.*}}>
  %v = tt.call @callee(%ptr) : (!tt.ptr<i32>) -> tensor<128xi32>
  %ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<128x!tt.ptr<i32>>
  tt.return
}

// -----

// When a callee returns a tensor whose default encoding doesn't match what
// the caller's consumer wants, a ttg.convert_layout should be auto-inserted
// at the call boundary.

// CHECK-LABEL: tt.func private @make_a
// CHECK: tt.return %{{.*}} : tensor<128x32xf16, #[[$BLOCKED:[^,>]+]]>
tt.func private @make_a() -> tensor<128x32xf16> {
  %a = arith.constant dense<1.0> : tensor<128x32xf16>
  tt.return %a : tensor<128x32xf16>
}

// CHECK-LABEL: tt.func @call_into_dot
// CHECK: %[[V:.*]] = tt.call @make_a() : () -> tensor<128x32xf16, #[[$BLOCKED]]>
// CHECK: ttg.convert_layout %[[V]] : tensor<128x32xf16, #[[$BLOCKED]]> -> tensor<128x32xf16, #ttg.dot_op<{{.*}}>>
// CHECK: tt.dot
tt.func @call_into_dot(%b: tensor<32x128xf16>) {
  %a = tt.call @make_a() : () -> tensor<128x32xf16>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %0 = tt.dot %a, %b, %c : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// CHECK-LABEL: @test_coalesce_unknown_alignment
// CHECK: tt.load {{.*}}tensor<128x!tt.ptr<f16>,
// CHECK: tt.load {{.*}}tensor<128x!tt.ptr<f16>,
tt.func @test_coalesce_unknown_alignment(%base: !tt.ptr<f16>) -> tensor<128xf16> {
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<f16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<f16>>, tensor<128xi32>
  %a = tt.load %p : tensor<128x!tt.ptr<f16>>
  %c = tt.load %q : tensor<128x!tt.ptr<f16>>
  %s = arith.addf %a, %c : tensor<128xf16>
  tt.return %s : tensor<128xf16>
}

// CHECK-LABEL: @test_coalesce_duplicate_masks
tt.func @test_coalesce_duplicate_masks(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %n: i32) -> tensor<128xf32> {
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<f16>>, tensor<128xi32>
  %ns = tt.splat %n : i32 -> tensor<128xi32>
  %m0 = arith.cmpi slt, %r, %ns : tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<256x!tt.ptr<f16>,
  // CHECK-NOT: tt.load
  %a = tt.load %p, %m0 : tensor<128x!tt.ptr<f16>>
  %af = arith.extf %a : tensor<128xf16> to tensor<128xf32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<f16>>, tensor<128xi32>
  %ns1 = tt.splat %n : i32 -> tensor<128xi32>
  %m1 = arith.cmpi slt, %r, %ns1 : tensor<128xi32>
  %c = tt.load %q, %m1 : tensor<128x!tt.ptr<f16>>
  %cf = arith.extf %c : tensor<128xf16> to tensor<128xf32>
  %sum = arith.addf %af, %cf : tensor<128xf32>
  // CHECK: tt.return
  tt.return %sum : tensor<128xf32>
}

// CHECK-LABEL: @test_coalesce_quad_unmasked
tt.func @test_coalesce_quad_unmasked(%base: !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> tensor<128xi32> {
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %four = arith.constant dense<4> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %three = arith.constant dense<3> : tensor<128xi32>
  %off = arith.muli %r, %four : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<512x!tt.ptr<i32>,
  // CHECK-NOT: tt.load
  %a = tt.load %p : tensor<128x!tt.ptr<i32>>
  %p1 = tt.addptr %p, %one : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %c = tt.load %p1 : tensor<128x!tt.ptr<i32>>
  %p2 = tt.addptr %p, %two : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %d = tt.load %p2 : tensor<128x!tt.ptr<i32>>
  %p3 = tt.addptr %p, %three : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %e = tt.load %p3 : tensor<128x!tt.ptr<i32>>
  %sum0 = arith.addi %a, %c : tensor<128xi32>
  %sum1 = arith.addi %d, %e : tensor<128xi32>
  %sum = arith.addi %sum0, %sum1 : tensor<128xi32>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi32>
}

// CHECK-LABEL: @test_coalesce_loads_pair
tt.func @test_coalesce_loads_pair(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<256x!tt.ptr<i16>,
  // CHECK-NOT: tt.load
  // CHECK: tt.split
  %a = tt.load %p, %mask, %zero : tensor<128x!tt.ptr<i16>>
  %c = tt.load %q, %mask, %zero  : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_loads_reverse_order
tt.func @test_coalesce_loads_reverse_order(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<256x!tt.ptr<i16>,
  // CHECK-NOT: tt.load
  // CHECK: tt.split
  %a = tt.load %q, %mask, %zero : tensor<128x!tt.ptr<i16>>
  %c = tt.load %p, %mask, %zero  : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_loads_different_other
tt.func @test_coalesce_loads_different_other(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<256x!tt.ptr<i16>,
  // CHECK-NOT: tt.load
  // CHECK: tt.split
  %a = tt.load %p, %mask, %zero : tensor<128x!tt.ptr<i16>>
  %c = tt.load %q, %mask, %fallback  : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_loads_store
tt.func @test_coalesce_loads_store(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %a = tt.load %p, %mask, %zero : tensor<128x!tt.ptr<i16>>
  tt.store %q, %zero, %mask : tensor<128x!tt.ptr<i16>>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %c = tt.load %q, %mask, %zero  : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_loads_unrelated_store
tt.func @test_coalesce_loads_unrelated_store(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %a = tt.load %p, %mask, %zero : tensor<128x!tt.ptr<i16>>
  tt.store %dst, %zeroScalar : !tt.ptr<i16>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %c = tt.load %q, %mask, %zero  : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_loads_atomic
tt.func @test_coalesce_loads_atomic(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %a = tt.load %p, %mask, %zero : tensor<128x!tt.ptr<i16>>
  %old = tt.atomic_rmw add, relaxed, gpu, %counter, %oneScalar : (!tt.ptr<i32>, i32) -> i32
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %c = tt.load %q, %mask, %zero  : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_loads_volatile
tt.func @test_coalesce_loads_volatile(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %a = tt.load %p, %mask, %zero : tensor<128x!tt.ptr<i16>>
  %ignored = tt.load %counter {isVolatile = true} : !tt.ptr<i32>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %c = tt.load %q, %mask, %zero  : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_loads_different_mask
tt.func @test_coalesce_loads_different_mask(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %a = tt.load %p, %mask, %zero : tensor<128x!tt.ptr<i16>>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %c = tt.load %q, %mask1, %zero  : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_loads_cache_mismatch
tt.func @test_coalesce_loads_cache_mismatch(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %a = tt.load %p, %mask, %zero : tensor<128x!tt.ptr<i16>>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %c = tt.load %q, %mask, %zero {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>} : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_loads_dependent_other
tt.func @test_coalesce_loads_dependent_other(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %a = tt.load %p, %mask, %zero : tensor<128x!tt.ptr<i16>>
  %late = arith.addi %a, %zero : tensor<128xi16>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %c = tt.load %q, %mask, %late  : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_loads_region
tt.func @test_coalesce_loads_region(%base: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<128xi1>, %mask1: tensor<128xi1>, %fallback: tensor<128xi16>, %dst: !tt.ptr<i16>, %counter: !tt.ptr<i32>, %cond: i1) -> tensor<128xi16> {
  %oneScalar = arith.constant 1 : i32
  %zeroScalar = arith.constant 0 : i16
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %two = arith.constant dense<2> : tensor<128xi32>
  %one = arith.constant dense<1> : tensor<128xi32>
  %zero = arith.constant dense<0> : tensor<128xi16>
  %off = arith.muli %r, %two : tensor<128xi32>
  %b = tt.splat %base : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
  %p = tt.addptr %b, %off : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  %q = tt.addptr %p, %one : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %a = tt.load %p, %mask, %zero : tensor<128x!tt.ptr<i16>>
  scf.if %cond {
    tt.store %dst, %zeroScalar : !tt.ptr<i16>
  }
  // CHECK: tt.load {{.*}}tensor<128x!tt.ptr<i16>,
  %c = tt.load %q, %mask, %zero  : tensor<128x!tt.ptr<i16>>
  %sum = arith.addi %a, %c : tensor<128xi16>
  // CHECK: tt.return
  tt.return %sum : tensor<128xi16>
}

// CHECK-LABEL: @test_coalesce_fewer_elements_than_threads
// CHECK: tt.load {{.*}}tensor<32x!tt.ptr<f16>,
// CHECK: tt.load {{.*}}tensor<32x!tt.ptr<f16>,
tt.func @test_coalesce_fewer_elements_than_threads(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<32xf16> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %two = arith.constant dense<2> : tensor<32xi32>
  %one = arith.constant dense<1> : tensor<32xi32>
  %off = arith.muli %r, %two : tensor<32xi32>
  %b = tt.splat %base : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
  %p = tt.addptr %b, %off : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
  %q = tt.addptr %p, %one : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
  %a = tt.load %p : tensor<32x!tt.ptr<f16>>
  %c = tt.load %q : tensor<32x!tt.ptr<f16>>
  %s = arith.addf %a, %c : tensor<32xf16>
  tt.return %s : tensor<32xf16>
}
