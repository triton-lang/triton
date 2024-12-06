// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx942 ftz=True" --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx942 ftz=False" --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=LLVM_NO_FTZ


#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp2(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.amdgcn.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = math.exp2 %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = math.exp %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_log(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.amdgcn.log.f32
    // LLVM_NO_FTZ: llvm.log.f32
    %0 = math.log %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_f32(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.amdgcn.sqrt.f32
    // LLVM_NO_FTZ: llvm.sqrt.f32
    %0 = math.sqrt %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_f64(%arg0: tensor<64xf64, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.amdgcn.sqrt.f64
    // LLVM_NO_FTZ: llvm.sqrt.f64
    %0 = math.sqrt %arg0 : tensor<64xf64, #blocked>
    tt.return
  }
}
