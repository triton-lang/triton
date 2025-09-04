// RUN: triton-opt %s -split-input-file --gluon-resolve-auto-encodings --verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @infer_conflict() -> (tensor<16xi32, #blocked>, tensor<16xi32, #blocked1>) {
    // expected-error-re @+1 {{found conflicting encodings for value:{{.*}}  #ttg.blocked<{sizePerThread = [1]{{.*}}and{{.*}}  #ttg.blocked<{sizePerThread = [2]}}
    %0 = arith.constant dense<7> : tensor<16xi32, #gluon.auto_encoding>
    %cvt1 = gluon.set_auto_layout %0 : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked>
    %cvt2 = gluon.set_auto_layout %0 : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked1>
    tt.return %cvt1, %cvt2 : tensor<16xi32, #blocked>, tensor<16xi32, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @infer_no_seed(%arg0 : !tt.ptr<i32>) {
    // expected-error @+1 {{Failed to infer return type}}
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #gluon.auto_encoding>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #gluon.auto_encoding>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<i32>, #gluon.auto_encoding>, tensor<32xi32, #gluon.auto_encoding>
    tt.store %2, %0 : tensor<32x!tt.ptr<i32>, #gluon.auto_encoding>
    tt.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{Functions taking auto encoding must be fully inlined}}
  tt.func public @function_argument(%arg0 : tensor<32xi32, #gluon.auto_encoding>) {
    tt.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{Functions returning auto encoding must be fully inlined}}
  tt.func public @function_return() -> tensor<32xi32, #gluon.auto_encoding> {
    %0 = arith.constant dense<0> : tensor<32xi32, #gluon.auto_encoding>
    tt.return %0 : tensor<32xi32, #gluon.auto_encoding>
  }
}
