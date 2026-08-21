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

#index = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The index and result infer the same rank-two encoding. The unseeded
  // rank-one basis independently infers its required warp-local encoding.
  // CHECK-DAG: [[$INDEX:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-DAG: [[$BASIS:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
  // CHECK-LABEL: @infer_linear_apply_independent_basis
  // CHECK: [[INDEX_VALUE:%.*]] = arith.constant dense<5> : tensor<8x8xi32, [[$INDEX]]>
  // CHECK: [[BASIS_VALUE:%.*]] = arith.constant dense<7> : tensor<32xi32, [[$BASIS]]>
  // CHECK: [[RESULT:%.*]] = tt.linear_apply [[INDEX_VALUE]], [[BASIS_VALUE]] : tensor<8x8xi32, [[$INDEX]]>, tensor<32xi32, [[$BASIS]]> -> tensor<8x8xi32, [[$INDEX]]>
  // CHECK: tt.return [[RESULT]] : tensor<8x8xi32, [[$INDEX]]>
  tt.func public @infer_linear_apply_independent_basis() -> tensor<8x8xi32, #index> {
    %index = arith.constant dense<5> : tensor<8x8xi32, #gluon.auto_encoding>
    %bases = arith.constant dense<7> : tensor<32xi32, #gluon.auto_encoding>
    %result = tt.linear_apply %index, %bases : tensor<8x8xi32, #gluon.auto_encoding>, tensor<32xi32, #gluon.auto_encoding> -> tensor<8x8xi32, #gluon.auto_encoding>
    %resolved_result = gluon.set_auto_layout %result : tensor<8x8xi32, #gluon.auto_encoding> -> tensor<8x8xi32, #index>
    tt.return %resolved_result : tensor<8x8xi32, #index>
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #parent}>
#index = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // An explicit slice encoding equivalent to the canonical blocked encoding
  // must be preserved instead of producing conflicting inference seeds.
  // CHECK-DAG: [[$PARENT:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-DAG: [[$INDEX_SLICE:#.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
  // CHECK-LABEL: @infer_linear_apply_explicit_slice_basis
  // CHECK: [[INDEX_VALUE:%.*]] = arith.constant dense<5> : tensor<128xi32, [[$INDEX_SLICE]]>
  // CHECK: [[BASIS_VALUE:%.*]] = arith.constant dense<7> : tensor<32xi32, #ttg.slice<{dim = 0, parent = [[$PARENT]]}>>
  // CHECK: [[RESULT:%.*]] = tt.linear_apply [[INDEX_VALUE]], [[BASIS_VALUE]] : tensor<128xi32, [[$INDEX_SLICE]]>, tensor<32xi32, #ttg.slice<{dim = 0, parent = [[$PARENT]]}>> -> tensor<128xi32, [[$INDEX_SLICE]]>
  // CHECK: tt.return [[RESULT]] : tensor<128xi32, [[$INDEX_SLICE]]>
  tt.func public @infer_linear_apply_explicit_slice_basis() -> tensor<128xi32, #index> {
    %index = arith.constant dense<5> : tensor<128xi32, #index>
    %bases = arith.constant dense<7> : tensor<32xi32, #gluon.auto_encoding>
    %resolved_bases = gluon.set_auto_layout %bases : tensor<32xi32, #gluon.auto_encoding> -> tensor<32xi32, #slice>
    %result = tt.linear_apply %index, %bases : tensor<128xi32, #index>, tensor<32xi32, #gluon.auto_encoding> -> tensor<128xi32, #index>
    tt.return %result : tensor<128xi32, #index>
  }
}

// -----

#index = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#basis = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // An explicit noncanonical seed takes priority over the canonical default.
  // CHECK-DAG: [[$INDEX_SPT4:#.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
  // CHECK-DAG: [[$BASIS_SPT4:#.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
  // CHECK-LABEL: @infer_linear_apply_explicit_spt4_basis
  // CHECK: [[INDEX_VALUE:%.*]] = arith.constant dense<5> : tensor<128xi32, [[$INDEX_SPT4]]>
  // CHECK: [[BASIS_VALUE:%.*]] = arith.constant dense<7> : tensor<32xi32, [[$BASIS_SPT4]]>
  // CHECK: [[RESULT:%.*]] = tt.linear_apply [[INDEX_VALUE]], [[BASIS_VALUE]] : tensor<128xi32, [[$INDEX_SPT4]]>, tensor<32xi32, [[$BASIS_SPT4]]> -> tensor<128xi32, [[$INDEX_SPT4]]>
  // CHECK: tt.return [[RESULT]] : tensor<128xi32, [[$INDEX_SPT4]]>
  tt.func public @infer_linear_apply_explicit_spt4_basis() -> tensor<128xi32, #index> {
    %index = arith.constant dense<5> : tensor<128xi32, #index>
    %bases = arith.constant dense<7> : tensor<32xi32, #gluon.auto_encoding>
    %resolved_bases = gluon.set_auto_layout %bases : tensor<32xi32, #gluon.auto_encoding> -> tensor<32xi32, #basis>
    %result = tt.linear_apply %index, %bases : tensor<128xi32, #index>, tensor<32xi32, #gluon.auto_encoding> -> tensor<128xi32, #index>
    tt.return %result : tensor<128xi32, #index>
  }
}

// -----

#index = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#basis = #ttg.slice<{dim = 0, parent = #parent}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Explicit CTA-local cross-warp basis distributions are also preserved.
  // CHECK-DAG: [[$CROSS_PARENT:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-DAG: [[$CROSS_INDEX:#.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
  // CHECK-LABEL: @infer_linear_apply_explicit_cross_warp_basis
  // CHECK: [[INDEX_VALUE:%.*]] = arith.constant dense<5> : tensor<128xi32, [[$CROSS_INDEX]]>
  // CHECK: [[BASIS_VALUE:%.*]] = arith.constant dense<7> : tensor<32xi32, #ttg.slice<{dim = 0, parent = [[$CROSS_PARENT]]}>>
  // CHECK: [[RESULT:%.*]] = tt.linear_apply [[INDEX_VALUE]], [[BASIS_VALUE]] : tensor<128xi32, [[$CROSS_INDEX]]>, tensor<32xi32, #ttg.slice<{dim = 0, parent = [[$CROSS_PARENT]]}>> -> tensor<128xi32, [[$CROSS_INDEX]]>
  // CHECK: tt.return [[RESULT]] : tensor<128xi32, [[$CROSS_INDEX]]>
  tt.func public @infer_linear_apply_explicit_cross_warp_basis() -> tensor<128xi32, #index> {
    %index = arith.constant dense<5> : tensor<128xi32, #index>
    %bases = arith.constant dense<7> : tensor<32xi32, #gluon.auto_encoding>
    %resolved_bases = gluon.set_auto_layout %bases : tensor<32xi32, #gluon.auto_encoding> -> tensor<32xi32, #basis>
    %result = tt.linear_apply %index, %bases : tensor<128xi32, #index>, tensor<32xi32, #gluon.auto_encoding> -> tensor<128xi32, #index>
    tt.return %result : tensor<128xi32, #index>
  }
}

// -----

#index = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // On a wave64 target, the basis has one value per lane, with lanes 32-63
  // replicating lanes 0-31 because the basis tensor has only 32 elements.
  // CHECK-DAG: [[$INDEX64:#.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
  // CHECK-DAG: [[$BASIS64:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
  // CHECK-LABEL: @infer_linear_apply_wave64_basis
  // CHECK: [[INDEX_VALUE:%.*]] = arith.constant dense<5> : tensor<128xi32, [[$INDEX64]]>
  // CHECK: [[BASIS_VALUE:%.*]] = arith.constant dense<7> : tensor<32xi32, [[$BASIS64]]>
  // CHECK: [[RESULT:%.*]] = tt.linear_apply [[INDEX_VALUE]], [[BASIS_VALUE]] : tensor<128xi32, [[$INDEX64]]>, tensor<32xi32, [[$BASIS64]]> -> tensor<128xi32, [[$INDEX64]]>
  // CHECK: tt.return [[RESULT]] : tensor<128xi32, [[$INDEX64]]>
  tt.func public @infer_linear_apply_wave64_basis() -> tensor<128xi32, #index> {
    %index = arith.constant dense<5> : tensor<128xi32, #gluon.auto_encoding>
    %bases = arith.constant dense<7> : tensor<32xi32, #gluon.auto_encoding>
    %result = tt.linear_apply %index, %bases : tensor<128xi32, #gluon.auto_encoding>, tensor<32xi32, #gluon.auto_encoding> -> tensor<128xi32, #gluon.auto_encoding>
    %resolved = gluon.set_auto_layout %result : tensor<128xi32, #gluon.auto_encoding> -> tensor<128xi32, #index>
    tt.return %resolved : tensor<128xi32, #index>
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
}
