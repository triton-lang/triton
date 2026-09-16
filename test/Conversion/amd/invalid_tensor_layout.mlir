// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm='gfx-arch=gfx942' -verify-diagnostics

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @local_load_missing_layout() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    // expected-error @+1 {{'ttg.local_load' op requires a distributed layout on tensor type 'tensor<32xf32>'}}
    %1 = ttg.local_load %0 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // expected-error @+1 {{'tt.func' op requires a distributed layout on tensor type 'tensor<64xi32>'}}
  tt.func public @func_arg_missing_layout(%arg0: tensor<64xi32>) -> tensor<64xi32> {
    tt.return %arg0 : tensor<64xi32>
  }
}
