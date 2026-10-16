// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --verify-diagnostics

// Pointers are only known to be 16-byte aligned.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unproven_alignment(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %splat, %range : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
    // expected-error @+1 {{cannot prove that each lane group accesses a contiguous, 128-byte aligned region: along dimension 0 the pointers have contiguity 512 (need 32) and divisibility 16 bytes (need 128)}}
    %v = amdg.cooperative_atomic_load relaxed, gpu, %ptrs : tensor<512x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Pointers are aligned but not contiguous: every lane reads the same row.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @not_contiguous(%base: !tt.ptr<f32> {tt.divisibility = 128 : i32}) {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %c32 = arith.constant dense<32> : tensor<512xi32, #blocked>
    %offs = arith.muli %range, %c32 : tensor<512xi32, #blocked>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %splat, %offs : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
    // expected-error @+1 {{cannot prove that each lane group accesses a contiguous, 128-byte aligned region}}
    %v = amdg.cooperative_atomic_load relaxed, gpu, %ptrs : tensor<512x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
