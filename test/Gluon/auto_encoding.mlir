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
