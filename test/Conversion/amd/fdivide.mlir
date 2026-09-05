// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s

// The default f32 division (`/` operator, `arith.divf` without fast-math
// flags) must stay IEEE-compliant (`llvm.fdiv`). Replacing it with the
// approximate sequence broke numerics upstream in PyTorch and was reverted,
// see https://github.com/pytorch/pytorch/issues/154215.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fdiv_f32_default(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: test_fdiv_f32_default
    // CHECK: llvm.fdiv
    // CHECK-NOT: llvm.amdgcn.div.scale.f32
    // CHECK-NOT: llvm.amdgcn.rcp.f32
    %0 = arith.divf %arg0, %arg1 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

// `tl.fdiv(x, y)` (ieee_rounding=False) is emitted with `fastmath<arcp>` and
// lowers to the approximate div.scale + rcp + Newton-Raphson + fixup sequence.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fdiv_f32_fast(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: test_fdiv_f32_fast
    // CHECK: llvm.amdgcn.div.scale.f32
    // CHECK: llvm.amdgcn.div.scale.f32
    // CHECK: llvm.amdgcn.rcp.f32
    // CHECK: llvm.fmul
    // CHECK: llvm.intr.fma
    // CHECK: llvm.intr.fma
    // CHECK: llvm.amdgcn.div.fmas.f32
    // CHECK: llvm.amdgcn.div.fixup.f32
    // CHECK-NOT: llvm.fdiv
    %0 = arith.divf %arg0, %arg1 fastmath<arcp> : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

// Non-f32 division stays precise even when the fast-math flag is present.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fdiv_f64_fast(%arg0: tensor<64xf64, #blocked>, %arg1: tensor<64xf64, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: test_fdiv_f64_fast
    // CHECK: llvm.fdiv
    // CHECK-NOT: llvm.amdgcn.div.scale.f32
    %0 = arith.divf %arg0, %arg1 fastmath<arcp> : tensor<64xf64, #blocked>
    tt.return
  }
}

// -----

// `tl.div_rn` / `tl.fdiv(..., ieee_rounding=True)` (`tt.precise_divf`) always
// lower to IEEE-compliant division.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_div_rn(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: test_div_rn
    // CHECK: llvm.fdiv
    // CHECK-NOT: llvm.amdgcn.rcp.f32
    %0 = tt.precise_divf %arg0, %arg1 : tensor<64xf32, #blocked>
    tt.return
  }
}
