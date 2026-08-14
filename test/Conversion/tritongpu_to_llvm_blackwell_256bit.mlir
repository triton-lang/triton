// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=88' -cse | FileCheck --check-prefix=BW256 %s
// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=103 ptx-version=88' -cse | FileCheck --check-prefix=SM103 %s
// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' -cse | FileCheck --check-prefix=PRE_BW %s

// Test 256-bit global load with 8x f32 (v8.b32) on Blackwell
#blocked_8xf32 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // BW256-LABEL: global_load_v8_b32
  // BW256: ld.global.v8.b32
  // PRE_BW-LABEL: global_load_v8_b32
  // PRE_BW-NOT: ld.global.v8.b32
  // PRE_BW: ld.global.v4.b32
  tt.func @global_load_v8_b32(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked_8xf32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked_8xf32>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked_8xf32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked_8xf32>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked_8xf32>, tensor<256xi32, #blocked_8xf32>
    %7 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked_8xf32>
    tt.return
  }
}

// -----

// Test 256-bit global store with 8x f32 (v8.b32) on Blackwell
#blocked_8xf32 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // BW256-LABEL: global_store_v8_b32
  // BW256: st.global.v8.b32
  // PRE_BW-LABEL: global_store_v8_b32
  // PRE_BW-NOT: st.global.v8.b32
  // PRE_BW: st.global.v4.b32
  tt.func @global_store_v8_b32(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<1.0> : tensor<256xf32, #blocked_8xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked_8xf32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked_8xf32>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked_8xf32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked_8xf32>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked_8xf32>, tensor<256xi32, #blocked_8xf32>
    tt.store %6, %cst : tensor<256x!tt.ptr<f32>, #blocked_8xf32>
    tt.return
  }
}

// -----

// Test 256-bit global load with 4x f64 (v4.b64) on Blackwell
#blocked_4xf64 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // BW256-LABEL: global_load_v4_b64
  // BW256: ld.global.v4.b64
  // PRE_BW-LABEL: global_load_v4_b64
  // PRE_BW-NOT: ld.global.v4.b64
  // PRE_BW: ld.global.v2.b64
  tt.func @global_load_v4_b64(%arg0: !tt.ptr<f64> {tt.divisibility = 32 : i32}) {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_4xf64>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked_4xf64>
    %4 = arith.addi %3, %2 : tensor<128xi32, #blocked_4xf64>
    %5 = tt.splat %arg0 : !tt.ptr<f64> -> tensor<128x!tt.ptr<f64>, #blocked_4xf64>
    %6 = tt.addptr %5, %4 : tensor<128x!tt.ptr<f64>, #blocked_4xf64>, tensor<128xi32, #blocked_4xf64>
    %7 = tt.load %6 : tensor<128x!tt.ptr<f64>, #blocked_4xf64>
    tt.return
  }
}

// -----

#blocked_redux_103 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:103", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // SM103-LABEL: @maxnum_reduction_sm103
  // SM103: nvvm.redux.sync fmax
  tt.func public @maxnum_reduction_sm103(%arg0: tensor<1x1024xf32, #blocked_redux_103>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %1 : f32
    }) {allocation.offset = 0 : i32} : (tensor<1x1024xf32, #blocked_redux_103>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_redux_103}>>
    tt.return
  }
}
