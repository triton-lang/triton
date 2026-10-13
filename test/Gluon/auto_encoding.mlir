// RUN: triton-opt %s -split-input-file --gluon-resolve-auto-encodings | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @infer_simple() -> tensor<8x16xi32, #blocked> {
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
    // CHECK: [[CST:%.*]] = arith.constant dense<7> : tensor<16xi32, #ttg.slice<{dim = 0, parent = [[BLOCKED]]}>>
    // CHECK: [[SLICE:%.*]] = tt.expand_dims [[CST]] {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = [[BLOCKED]]}>> -> tensor<1x16xi32, [[BLOCKED]]>
    // CHECK: [[BROADCAST:%.*]] = tt.broadcast [[SLICE]] : tensor<1x16xi32, [[BLOCKED]]> -> tensor<8x16xi32, [[BLOCKED]]>
    // CHECK: tt.return [[BROADCAST]] : tensor<8x16xi32, [[BLOCKED]]>
    %x_1d = arith.constant dense<7> : tensor<16xi32, #gluon.auto_encoding>
    %x_slice = tt.expand_dims %x_1d {axis = 0 : i32} : tensor<16xi32, #gluon.auto_encoding> -> tensor<1x16xi32, #gluon.auto_encoding>
    %x_2d = tt.broadcast %x_slice : tensor<1x16xi32, #gluon.auto_encoding> -> tensor<8x16xi32, #gluon.auto_encoding>
    %cvt = gluon.set_auto_layout %x_2d : tensor<8x16xi32, #gluon.auto_encoding> -> tensor<8x16xi32, #blocked>
    tt.return %cvt : tensor<8x16xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @infer_with_convert() -> tensor<16xi32, #blocked1> {
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    // CHECK-DAG: [[BLOCKED1:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    // CHECK: [[CST:%.*]] = arith.constant dense<7> : tensor<16xi32, [[BLOCKED]]>
    // CHECK: [[CVT1:%.*]] = ttg.convert_layout [[CST]] : tensor<16xi32, [[BLOCKED]]> -> tensor<16xi32, [[BLOCKED1]]>
    // CHECK: [[ADD:%.*]] = arith.addi [[CVT1]], [[CVT1]] : tensor<16xi32, [[BLOCKED1]]>
    // CHECK: tt.return [[ADD]] : tensor<16xi32, [[BLOCKED1]]>
    %0 = arith.constant dense<7> : tensor<16xi32, #blocked>
    %cvt1 = ttg.convert_layout %0 : tensor<16xi32, #blocked> -> tensor<16xi32, #gluon.auto_encoding>
    %add = arith.addi %cvt1, %cvt1 : tensor<16xi32, #gluon.auto_encoding>
    %cvt2 = gluon.set_auto_layout %add : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked1>
    tt.return %cvt2 : tensor<16xi32, #blocked1>
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @infer_if(%arg0 : i1) -> tensor<16xi32, #blocked> {
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    // CHECK: [[C1:%.*]] = arith.constant dense<1> : tensor<16xi32, [[BLOCKED]]>
    // CHECK: [[C2:%.*]] = arith.constant dense<2> : tensor<16xi32, [[BLOCKED]]>
    // CHECK: [[IF:%.*]] = scf.if %arg0 -> (tensor<16xi32, [[BLOCKED]]>) {
    // CHECK:   scf.yield [[C1]] : tensor<16xi32, [[BLOCKED]]>
    // CHECK: } else {
    // CHECK:   scf.yield [[C2]] : tensor<16xi32, [[BLOCKED]]>
    // CHECK: }
    // CHECK: tt.return [[IF]] : tensor<16xi32, [[BLOCKED]]>
    %c1 = arith.constant dense<1> : tensor<16xi32, #gluon.auto_encoding>
    %c2 = arith.constant dense<2> : tensor<16xi32, #gluon.auto_encoding>
    %z = scf.if %arg0 -> tensor<16xi32, #gluon.auto_encoding> {
      scf.yield %c1 : tensor<16xi32, #gluon.auto_encoding>
    } else {
      scf.yield %c2 : tensor<16xi32, #gluon.auto_encoding>
    }
    %cvt = gluon.set_auto_layout %z : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked>
    tt.return %cvt : tensor<16xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @infer_for(%arg0: i32) -> tensor<32xi32, #blocked> {
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    // CHECK: [[RANGE:%.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, [[BLOCKED]]>
    // CHECK: [[IF:%.*]] = scf.for {{%.*}} = %c0_i32 to %arg0 step %c1_i32 iter_args([[ITER_ARG:%.*]] = [[RANGE]]) -> (tensor<32xi32, [[BLOCKED]]>) : i32 {
    // CHECK:   [[CST:%.*]] = arith.constant dense<2> : tensor<32xi32, [[BLOCKED]]>
    // CHECK:   [[MUL:%.*]] = arith.muli [[ITER_ARG]], [[CST]] : tensor<32xi32, [[BLOCKED]]>
    // CHECK:   scf.yield [[MUL]] : tensor<32xi32, [[BLOCKED]]>
    // CHECK: }
    // CHECK: tt.return [[IF]] : tensor<32xi32, [[BLOCKED]]>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #gluon.auto_encoding>
    %1 = scf.for %arg1 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg2 = %0) -> (tensor<32xi32, #gluon.auto_encoding>) : i32 {
      %cst = arith.constant dense<2> : tensor<32xi32, #gluon.auto_encoding>
      %2 = arith.muli %arg2, %cst : tensor<32xi32, #gluon.auto_encoding>
      scf.yield %2 : tensor<32xi32, #gluon.auto_encoding>
    }
    %cvt = gluon.set_auto_layout %1 : tensor<32xi32, #gluon.auto_encoding> -> tensor<32xi32, #blocked>
    tt.return %cvt : tensor<32xi32, #blocked>
  }
}


// -----


#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @infer_make_range() -> tensor<16xi32, #blocked> {
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    // CHECK: [[CST:%.*]] = arith.constant 0 : i32
    // CHECK: [[SPLAT: %.*]] = tt.splat [[CST]] : i32 -> tensor<16xi32, [[BLOCKED]]>
    // CHECK: tt.return [[RANGE]] : tensor<16xi32, [[BLOCKED]]>
    %cst = arith.constant 0 : i32
    %0 = tt.splat %cst : i32 -> tensor<16xi32, #gluon.auto_encoding>
    %cvt = gluon.set_auto_layout %0 : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked>
    tt.return %cvt : tensor<16xi32, #blocked>
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {ttg.maxnreg = 128 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @infer_with_downstream_ops() -> tensor<128x128xi32, #blocked> {
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
    // CHECK: [[RANGE:%.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = [[BLOCKED]]}>>
    // CHECK: [[EXPAND:%.*]] = tt.expand_dims [[RANGE]] {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = [[BLOCKED]]}>> -> tensor<1x128xi32, [[BLOCKED]]>
    // CHECK: [[BROADCAST:%.*]] = tt.broadcast [[EXPAND]] : tensor<1x128xi32, [[BLOCKED]]> -> tensor<128x128xi32, [[BLOCKED]]>
    // CHECK: tt.return [[BROADCAST]] : tensor<128x128xi32, [[BLOCKED]]>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #gluon.auto_encoding>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<128xi32, #gluon.auto_encoding> -> tensor<1x128xi32, #gluon.auto_encoding>
    %2 = gluon.set_auto_layout %1 : tensor<1x128xi32, #gluon.auto_encoding> -> tensor<1x128xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    tt.return %3 : tensor<128x128xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_tmem_col_slice_load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<64x128xi32, #blocked> {
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
    // CHECK-DAG: [[LINEAR:#.*]] = #ttg.linear
    // CHECK: [[RANGE:%.*]] = tt.make_range {end = 8192 : i32, start = 0 : i32} : tensor<8192xi32, [[LINEAR]]>
    // CHECK: [[RESHAPE:%.*]] = tt.reshape [[RANGE]] : tensor<8192xi32, [[LINEAR]]> -> tensor<64x128xi32, [[BLOCKED]]>
    // CHECK: tt.return [[RESHAPE]] : tensor<64x128xi32, [[BLOCKED]]>
    %0 = tt.make_range {end = 8192 : i32, start = 0 : i32} : tensor<8192xi32, #gluon.auto_encoding>
    %1 = tt.reshape %0 : tensor<8192xi32, #gluon.auto_encoding> -> tensor<64x128xi32, #gluon.auto_encoding>
    %2 = gluon.set_auto_layout %1 : tensor<64x128xi32, #gluon.auto_encoding> -> tensor<64x128xi32, #blocked>
    tt.return %2 : tensor<64x128xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @infer_if_yield_propagation
  tt.func public @infer_if_yield_propagation(%cond: i1) -> tensor<16xi32, #blocked> {
    // The scf.if has two results. Result #0 is resolved from outside (via
    // set_auto_layout on %z#0). Result #1 can only be resolved via forward
    // propagation from the yield operands inside the if body.
    //
    // CHECK: %[[IF:.*]]:2 = scf.if {{.*}} -> (tensor<16xi32, #{{.*}}>, tensor<16xi32, #{{.*}}>)
    // CHECK-NOT: auto_encoding
    %c1 = arith.constant dense<1> : tensor<16xi32, #gluon.auto_encoding>
    %c2 = arith.constant dense<2> : tensor<16xi32, #gluon.auto_encoding>
    %z:2 = scf.if %cond -> (tensor<16xi32, #gluon.auto_encoding>, tensor<16xi32, #gluon.auto_encoding>) {
      %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #gluon.auto_encoding>
      %resolved = gluon.set_auto_layout %range : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked1>
      scf.yield %c1, %range : tensor<16xi32, #gluon.auto_encoding>, tensor<16xi32, #gluon.auto_encoding>
    } else {
      %cst = arith.constant dense<0> : tensor<16xi32, #gluon.auto_encoding>
      %resolved2 = gluon.set_auto_layout %cst : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked1>
      scf.yield %c2, %cst : tensor<16xi32, #gluon.auto_encoding>, tensor<16xi32, #gluon.auto_encoding>
    }
    %out = gluon.set_auto_layout %z#0 : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked>
    tt.return %out : tensor<16xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @infer_reduce_to_scalar
  // CHECK-NOT: auto_encoding
  // CHECK: "tt.reduce"
  // CHECK: tt.return
  tt.func public @infer_reduce_to_scalar() -> i32 {
    %0 = arith.constant dense<1> : tensor<16xi32, #gluon.auto_encoding>
    %1 = gluon.set_auto_layout %0 : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked>
    %2 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %3 = arith.addi %lhs, %rhs : i32
      tt.reduce.return %3 : i32
    }) : (tensor<16xi32, #gluon.auto_encoding>) -> i32
    tt.return %2 : i32
  }

  // The scalar results cannot propagate a layout between the two operands.
  // CHECK-LABEL: @infer_reduce_operands_to_scalar
  // CHECK: arith.constant dense<1> : tensor<16xi32, [[ENC:#.*]]>
  // CHECK: arith.constant {{.*}} : tensor<16xf32, [[ENC]]>
  // CHECK: "tt.reduce"
  // CHECK-NOT: auto_encoding
  // CHECK: tt.return
  tt.func public @infer_reduce_operands_to_scalar() -> (i32, f32) {
    %ints = arith.constant dense<1> : tensor<16xi32, #gluon.auto_encoding>
    %floats = arith.constant dense<1.0> : tensor<16xf32, #gluon.auto_encoding>
    %fixed = gluon.set_auto_layout %ints : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked>
    %result:2 = "tt.reduce"(%ints, %floats) <{axis = 0 : i32}> ({
    ^bb0(%lhs_i: i32, %lhs_f: f32, %rhs_i: i32, %rhs_f: f32):
      %sum_i = arith.addi %lhs_i, %rhs_i : i32
      %sum_f = arith.addf %lhs_f, %rhs_f : f32
      tt.reduce.return %sum_i, %sum_f : i32, f32
    }) : (tensor<16xi32, #gluon.auto_encoding>, tensor<16xf32, #gluon.auto_encoding>) -> (i32, f32)
    tt.return %result#0, %result#1 : i32, f32
  }
}

// -----

// The reshape can provisionally infer a linear encoding for the cast result.
// The explicit slice seed must refine it so both sides of the cast agree.
// CHECK-LABEL: @refine_reshape_layout
// CHECK: %[[POSITIONS:.*]] = tt.make_range {{.*}} : tensor<8xi32, #ttg.slice<{dim = 1, parent = [[BLOCKED:#.*]]}>>
// CHECK: %[[CAST:.*]] = arith.sitofp %[[POSITIONS]] : tensor<8xi32, #ttg.slice<{dim = 1, parent = [[BLOCKED]]}>> to tensor<8xf32, #ttg.slice<{dim = 1, parent = [[BLOCKED]]}>>
// CHECK: tt.reshape %[[CAST]]
// CHECK-NOT: auto_encoding
// CHECK: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @refine_reshape_layout() -> (tensor<8xi32, #slice>, tensor<8x1xf32, #blocked>) {
    %positions = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #gluon.auto_encoding>
    %cast = arith.sitofp %positions : tensor<8xi32, #gluon.auto_encoding> to tensor<8xf32, #gluon.auto_encoding>
    %expanded = tt.reshape %cast : tensor<8xf32, #gluon.auto_encoding> -> tensor<8x1xf32, #gluon.auto_encoding>
    %fixed_positions = gluon.set_auto_layout %positions : tensor<8xi32, #gluon.auto_encoding> -> tensor<8xi32, #slice>
    %fixed_expanded = gluon.set_auto_layout %expanded : tensor<8x1xf32, #gluon.auto_encoding> -> tensor<8x1xf32, #blocked>
    tt.return %fixed_positions, %fixed_expanded : tensor<8xi32, #slice>, tensor<8x1xf32, #blocked>
  }
}

// -----

// Join permits several result register orders. Keep the order inferred
// backwards from the dot operand instead of replacing it with the default.
// CHECK-LABEL: @preserve_join_layout
// CHECK-NOT: auto_encoding
// CHECK: %[[LHS:.*]] = tt.trans {{.*}} -> tensor<4x128x16xbf16, #ttg.dot_op<
// CHECK-NEXT: tt.return %[[LHS]]
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1, 1], instrShape = [1, 16, 8]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#auto = #gluon.auto_encoding
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @preserve_join_layout(%arg: bf16) -> tensor<4x128x16xbf16, #dot> {
    %input = tt.splat %arg : bf16 -> tensor<4x8x128xbf16, #auto>
    %padding = arith.constant dense<0.0> : tensor<4x8x128xbf16, #auto>
    %joined = tt.join %input, %padding : tensor<4x8x128xbf16, #auto> -> tensor<4x8x128x2xbf16, #auto>
    %transposed = tt.trans %joined {order = array<i32: 0, 3, 1, 2>} : tensor<4x8x128x2xbf16, #auto> -> tensor<4x2x8x128xbf16, #auto>
    %reshaped = tt.reshape %transposed : tensor<4x2x8x128xbf16, #auto> -> tensor<4x16x128xbf16, #auto>
    %lhs = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<4x16x128xbf16, #auto> -> tensor<4x128x16xbf16, #auto>
    %result = gluon.set_auto_layout %lhs : tensor<4x128x16xbf16, #auto> -> tensor<4x128x16xbf16, #dot>
    tt.return %result : tensor<4x128x16xbf16, #dot>
  }
}

// -----

// Join operands must reconcile to identical types, even for equivalent layouts.
// CHECK-LABEL: @equivalent_join_operands
// CHECK-NOT: auto_encoding
// CHECK: tt.return
#q = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 4], order = [0, 1, 2, 3]}>
#j = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 2], threadsPerWarp = [1, 1, 1, 32, 1], warpsPerCTA = [1, 1, 1, 4, 1], order = [4, 0, 2, 1, 3]}>
#auto = #gluon.auto_encoding
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @equivalent_join_operands(%ptr: !tt.ptr<f32>) -> (tensor<1x1x1x128xf32, #q>, tensor<1x1x1x128x2xf32, #j>) {
    %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<1x1x1x128x!tt.ptr<f32>, #auto>
    %x = tt.load %p : tensor<1x1x1x128x!tt.ptr<f32>, #auto>
    %a = tt.trans %x {order = array<i32: 1, 0, 2, 3>} : tensor<1x1x1x128xf32, #auto> -> tensor<1x1x1x128xf32, #auto>
    %b = tt.trans %x {order = array<i32: 0, 2, 1, 3>} : tensor<1x1x1x128xf32, #auto> -> tensor<1x1x1x128xf32, #auto>
    %joined = tt.join %a, %b : tensor<1x1x1x128xf32, #auto> -> tensor<1x1x1x128x2xf32, #auto>
    %anchor = gluon.set_auto_layout %x : tensor<1x1x1x128xf32, #auto> -> tensor<1x1x1x128xf32, #q>
    %out = ttg.convert_layout %joined : tensor<1x1x1x128x2xf32, #auto> -> tensor<1x1x1x128x2xf32, #j>
    tt.return %anchor, %out : tensor<1x1x1x128xf32, #q>, tensor<1x1x1x128x2xf32, #j>
  }
}

// -----

// Split results must reconcile to identical types.
// CHECK-LABEL: @split_result_reconciliation
// CHECK-NOT: auto_encoding
// CHECK: tt.return
#parent = #ttg.blocked<{sizePerThread = [1, 2, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#slice = #ttg.slice<{dim = 2, parent = #parent}>
#seed = #ttg.slice<{dim = 1, parent = #slice}>
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#auto = #gluon.auto_encoding
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @split_result_reconciliation(%xptr: !tt.ptr<f32>, %yptr: !tt.ptr<f32>) -> (tensor<128x2xf32, #blocked>, tensor<128xf32, #seed>, tensor<128x2xf32, #blocked>) {
    %xp = tt.splat %xptr : !tt.ptr<f32> -> tensor<128x2x2x!tt.ptr<f32>, #auto>
    %x = tt.load %xp : tensor<128x2x2x!tt.ptr<f32>, #auto>
    %parts:2 = tt.split %x : tensor<128x2x2xf32, #auto> -> tensor<128x2xf32, #auto>
    %yp = tt.splat %yptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #auto>
    %y = tt.load %yp : tensor<128x!tt.ptr<f32>, #auto>
    %joined = tt.join %y, %y : tensor<128xf32, #auto> -> tensor<128x2xf32, #auto>
    %sum = arith.addf %joined, %parts#0 : tensor<128x2xf32, #auto>
    %part_anchor = gluon.set_auto_layout %parts#1 : tensor<128x2xf32, #auto> -> tensor<128x2xf32, #blocked>
    %seed_anchor = gluon.set_auto_layout %y : tensor<128xf32, #auto> -> tensor<128xf32, #seed>
    %out = ttg.convert_layout %sum : tensor<128x2xf32, #auto> -> tensor<128x2xf32, #blocked>
    tt.return %part_anchor, %seed_anchor, %out : tensor<128x2xf32, #blocked>, tensor<128xf32, #seed>, tensor<128x2xf32, #blocked>
  }
}

// -----

// Backward split inference chooses a register order, but the transpose fixes
// the source layout independently of that choice.
// CHECK: [[$SOURCE:#.*]] = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-LABEL: @preserve_split_source_layout
// CHECK: %[[INPUT:.*]] = tt.splat {{.*}} -> tensor<128x2xf32, [[$SOURCE]]>
// CHECK: tt.split %[[INPUT]]
// CHECK: tt.trans %[[INPUT]]
// CHECK-NOT: auto_encoding
// CHECK: tt.return
#transposed = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#part = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#auto = #gluon.auto_encoding
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @preserve_split_source_layout(%arg: f32) -> (tensor<128xf32, #part>, tensor<2x128xf32, #transposed>) {
    %input = tt.splat %arg : f32 -> tensor<128x2xf32, #auto>
    %parts:2 = tt.split %input : tensor<128x2xf32, #auto> -> tensor<128xf32, #auto>
    %transposed = tt.trans %input {order = array<i32: 1, 0>} : tensor<128x2xf32, #auto> -> tensor<2x128xf32, #auto>
    %fixed_part = gluon.set_auto_layout %parts#0 : tensor<128xf32, #auto> -> tensor<128xf32, #part>
    %fixed_transposed = gluon.set_auto_layout %transposed : tensor<2x128xf32, #auto> -> tensor<2x128xf32, #transposed>
    tt.return %fixed_part, %fixed_transposed : tensor<128xf32, #part>, tensor<2x128xf32, #transposed>
  }
}

// -----

// Backward inference from the split gives the join result distance 5. A round
// trip through the join operands proposes another register order at distance 7.
// CHECK-LABEL: @prefer_fewer_layout_choices
// CHECK: tt.join
// CHECK: %[[PART:.*]], %{{.*}} = tt.split
// CHECK-NEXT: tt.return %[[PART]]
#part = #ttg.linear<{register = [[0, 8, 0], [0, 0, 4], [0, 16, 0], [0, 32, 0], [0, 64, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 2, 0], [0, 4, 0]], warp = [[1, 0, 0], [2, 0, 0]], block = []}>
#auto = #gluon.auto_encoding
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @prefer_fewer_layout_choices(%arg: bf16) -> tensor<4x128x8xbf16, #part> {
    %input = tt.splat %arg : bf16 -> tensor<4x8x128xbf16, #auto>
    %padding = arith.constant dense<0.0> : tensor<4x8x128xbf16, #auto>
    %joined = tt.join %input, %padding : tensor<4x8x128xbf16, #auto> -> tensor<4x8x128x2xbf16, #auto>
    %transposed = tt.trans %joined {order = array<i32: 0, 3, 1, 2>} : tensor<4x8x128x2xbf16, #auto> -> tensor<4x2x8x128xbf16, #auto>
    %reshaped = tt.reshape %transposed : tensor<4x2x8x128xbf16, #auto> -> tensor<4x16x128xbf16, #auto>
    %lhs = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<4x16x128xbf16, #auto> -> tensor<4x128x16xbf16, #auto>
    %paired = tt.reshape %lhs : tensor<4x128x16xbf16, #auto> -> tensor<4x128x8x2xbf16, #auto>
    %parts:2 = tt.split %paired : tensor<4x128x8x2xbf16, #auto> -> tensor<4x128x8xbf16, #auto>
    %result = gluon.set_auto_layout %parts#0 : tensor<4x128x8xbf16, #auto> -> tensor<4x128x8xbf16, #part>
    tt.return %result : tensor<4x128x8xbf16, #part>
  }
}

// -----

// Broadcast source and result layouts can be fixed independently when every
// thread already holds the values it needs.
// CHECK-DAG: [[$SOURCE:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: [[$RESULT:#.*]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: @preserve_broadcast_layouts
// CHECK: %[[INPUT:.*]] = tt.splat {{.*}} -> tensor<1x32xf32, [[$SOURCE]]>
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[INPUT]] : tensor<1x32xf32, [[$SOURCE]]> -> tensor<8x32xf32, [[$RESULT]]>
// CHECK-NEXT: tt.return %[[INPUT]], %[[BROADCAST]]
#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#result = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#auto = #gluon.auto_encoding
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @preserve_broadcast_layouts(%arg: f32) -> (tensor<1x32xf32, #source>, tensor<8x32xf32, #result>) {
    %input = tt.splat %arg : f32 -> tensor<1x32xf32, #auto>
    %broadcast = tt.broadcast %input : tensor<1x32xf32, #auto> -> tensor<8x32xf32, #auto>
    %fixed_input = gluon.set_auto_layout %input : tensor<1x32xf32, #auto> -> tensor<1x32xf32, #source>
    %fixed_result = gluon.set_auto_layout %broadcast : tensor<8x32xf32, #auto> -> tensor<8x32xf32, #result>
    tt.return %fixed_input, %fixed_result : tensor<1x32xf32, #source>, tensor<8x32xf32, #result>
  }
}
