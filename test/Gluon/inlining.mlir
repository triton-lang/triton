// RUN: triton-opt %s --gluon-inline | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @set_encoding(%arg0 : tensor<16xi32, #gluon.auto_encoding>) -> tensor<16xi32, #blocked> {
    %cvt = gluon.set_auto_layout %arg0 : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked>
    tt.return %cvt : tensor<16xi32, #blocked>
  }

  tt.func public @infer_make_range() -> tensor<16xi32, #blocked> {
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    // CHECK: [[CST:%.*]] = arith.constant dense<0> : tensor<16xi32, #gluon.auto_encoding>
    // CHECK: [[SET:%.*]] = gluon.set_auto_layout [[CST]] : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, [[BLOCKED]]>
    // CHECK: tt.return [[SET]] : tensor<16xi32, [[BLOCKED]]>
    %cst = arith.constant dense<0> : tensor<16xi32, #gluon.auto_encoding>
    %0 = tt.call @"set_encoding"(%cst) : (tensor<16xi32, #gluon.auto_encoding>) -> tensor<16xi32, #blocked>
    tt.return %0 : tensor<16xi32, #blocked>
  }
}
