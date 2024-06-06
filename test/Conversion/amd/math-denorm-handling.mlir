// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx942 ftz=True" --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx942 ftz=False" --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=LLVM_NO_FTZ


#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp2(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.amdgcn.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = math.exp2 %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp2(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = math.exp %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}
