// RUN: env TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=1 triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=16" | FileCheck %s --check-prefixes ON16
// RUN: env TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=1 triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=32" | FileCheck %s --check-prefixes ON32
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=16" | FileCheck %s --check-prefixes OFF16
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=32" | FileCheck %s --check-prefixes OFF32

// Chain-dot detection across one level of scf.if.
//
// isChainDotHead/isChainDotTail require every op between the two dots to sit in
// the same MLIR region, so nesting the second dot inside an scf.if defeats
// detection and planWarps() falls back to a generic layout. With
// TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF set, the detectors look through exactly
// one level of scf.if and the chain-dot layout is chosen as if the branch were
// not there.
//
// This lives in its own file rather than extending
// accelerate-amd-matmul-chain-dot.mlir because the behaviour under test is
// selected by an environment variable, and that file's cases are shared by RUN
// lines that must not have it set. `env` on the RUN line is also the only way
// to deliver it: test/lit.cfg.py forwards just HOME/INCLUDE/LIB/TMP/TEMP, so an
// ambient TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF never reaches triton-opt. That
// same scrubbing is what makes the OFF arms below reliably unset.
//
// BLOCK_M is 64, not 128: at 128 the generic layout and the chain-dot layout are
// both [4, 1] and the test could not tell them apart. The shape is otherwise
// mfma_chain_dot_BM64 from the sibling file, with the second dot wrapped.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

// Detected: both dots take the chain-dot layout, matching the unwrapped case.
// ON16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// ON32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
// ON32{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>

// Not detected: the head dot loses [4, 1] under mfma32, and the tail dot falls
// back to a generic [1, 4] under both instruction sizes.
// OFF16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// OFF16{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16, 16], isTransposed = true}>
// OFF32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
// OFF32{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [32, 32, 8], isTransposed = true}>

// ON16-LABEL: chain_dot_scf_if_BM64
// ON32-LABEL: chain_dot_scf_if_BM64
// OFF16-LABEL: chain_dot_scf_if_BM64
// OFF32-LABEL: chain_dot_scf_if_BM64

// The head dot, outside the scf.if.
// ON16: tt.dot {{.*}} -> tensor<64x16xf32, #mma>
// ON32: tt.dot {{.*}} -> tensor<64x16xf32, #mma>
// OFF16: tt.dot {{.*}} -> tensor<64x16xf32, #mma>
// OFF32: tt.dot {{.*}} -> tensor<64x16xf32, #mma>

// The tail dot, one region below.
// ON16: tt.dot {{.*}} -> tensor<64x128xf32, #mma>
// ON32: tt.dot {{.*}} -> tensor<64x128xf32, #mma1>
// OFF16: tt.dot {{.*}} -> tensor<64x128xf32, #mma1>
// OFF32: tt.dot {{.*}} -> tensor<64x128xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @chain_dot_scf_if_BM64(
      %q: tensor<64x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %v: tensor<16x128xf16, #dotOp1>,
      %cond: i1,
      %o_ptr: tensor<64x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<64x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<64x16xf32, #blocked>
    %o = scf.if %cond -> (tensor<64x128xf32, #blocked>) {
      %qk_f16 = arith.truncf %qk : tensor<64x16xf32, #blocked> to tensor<64x16xf16, #blocked>
      %p = ttg.convert_layout %qk_f16 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #dotOp0>
      %d = tt.dot %p, %v, %cst1 : tensor<64x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<64x128xf32, #blocked>
      scf.yield %d : tensor<64x128xf32, #blocked>
    } else {
      scf.yield %cst1 : tensor<64x128xf32, #blocked>
    }
    tt.store %o_ptr, %o : tensor<64x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
