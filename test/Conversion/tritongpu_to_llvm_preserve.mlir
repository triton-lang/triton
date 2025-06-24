// RUN: triton-opt %s -convert-triton-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @test_preserve_to_llvm
  tt.func public @test_preserve_to_llvm(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %3 = tt.load %2 : tensor<256x!tt.ptr<f32>, #blocked>

    // CHECK: call void asm sideeffect "", "r,~{memory}"
    tt.preserve %3 : tensor<256xf32, #blocked>

    tt.return
  }
}
