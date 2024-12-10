// RUN: not triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=90

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @load_ops(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<true> : tensor<128xi1, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked>
    %1 = tt.load %0 : tensor<128x!tt.ptr<f32>, #blocked>
    %2 = tt.load %0, %cst : tensor<128x!tt.ptr<f32>, #blocked>
    %3 = tt.load %0, %cst, %cst_0 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %0, %1 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %0, %2 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %0, %3 memSemantic = release memSyncScope = cta : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
