// RUN: triton-opt %s -split-input-file -tritonamdgpu-lds-prefetch=num-insts=4 -canonicalize | FileCheck %s --dump-input-context=50
// RUN: triton-opt %s -split-input-file -tritonamdgpu-lds-prefetch -canonicalize | FileCheck %s --check-prefix=DEFAULT --dump-input-context=50

// ============================================================================
// Test 1: slice_k_only — gfx942, MFMA v3, !transA, !transB, async
// A [32, 128] f16, B [128, 32] f16, C/D [32, 32] f32
// prefetchWidth = (32, 32, 64) → K sliced (2 K-slices)
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: slice_k_only
// Prologue: subslice A[0,0] -> [32,64], subslice B[0,0] -> [64,32]
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<32x64xf16
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<64x32xf16
// CHECK: scf.for
// In-loop: remainder subslices at K offset 64
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 64] {{.*}} -> !ttg.memdesc<32x64xf16
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[64, 0] {{.*}} -> !ttg.memdesc<64x32xf16
// Two dots (2 K-slices)
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @slice_k_only(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<32x128xf16, #blocked>,
      %b_tensor: tensor<128x32xf16, #blocked>) -> tensor<32x32xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<32x128xf16, #blocked>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem>
    %tok_init = ttg.async_wait {num = 0 : i32}
    %loop:4 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init, %awt = %tok_init)
      -> (!ttg.memdesc<32x128xf16, #shared, #smem>, !ttg.memdesc<128x32xf16, #shared1, #smem>, tensor<32x32xf32, #mma>, !ttg.async.token) {
      %a_op = ttg.local_load %a token %awt : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b token %awt : !ttg.memdesc<128x32xf16, #shared1, #smem> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<32x128xf16, #blocked>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem>
      %next_tok = ttg.async_wait {num = 0 : i32}
      scf.yield %next_a, %next_b, %d, %next_tok : !ttg.memdesc<32x128xf16, #shared, #smem>, !ttg.memdesc<128x32xf16, #shared1, #smem>, tensor<32x32xf32, #mma>, !ttg.async.token
    }
    tt.return %loop#2 : tensor<32x32xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 2: slice_m_only — gfx942, MFMA v3, !transA, !transB, async
// A [256, 16] f16, B [16, 32] f16, C/D [256, 32] f32
// prefetchWidth = (128, 32, 16) → M sliced (2 M-slices)
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: slice_m_only
// Prologue: subslice A[0,0] -> [128,16], subslice B[0,0] -> [16,32]
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<128x16xf16
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<16x32xf16
// CHECK: scf.for
// In-loop: remainder A at M offset 128
// CHECK: ttg.memdesc_subslice {{.*}}[128, 0] {{.*}} -> !ttg.memdesc<128x16xf16
// Two dots (2 M-slices); B is not sliced so reused
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @slice_m_only(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<256x16xf16, #blocked>,
      %b_tensor: tensor<16x32xf16, #blocked>) -> tensor<256x32xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<256x32xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<256x16xf16, #blocked>) -> !ttg.memdesc<256x16xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<16x32xf16, #blocked>) -> !ttg.memdesc<16x32xf16, #shared1, #smem>
    %tok_init = ttg.async_wait {num = 0 : i32}
    %loop:4 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init, %awt = %tok_init)
      -> (!ttg.memdesc<256x16xf16, #shared, #smem>, !ttg.memdesc<16x32xf16, #shared1, #smem>, tensor<256x32xf32, #mma>, !ttg.async.token) {
      %a_op = ttg.local_load %a token %awt : !ttg.memdesc<256x16xf16, #shared, #smem> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b token %awt : !ttg.memdesc<16x32xf16, #shared1, #smem> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x32xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<256x16xf16, #blocked>) -> !ttg.memdesc<256x16xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<16x32xf16, #blocked>) -> !ttg.memdesc<16x32xf16, #shared1, #smem>
      %next_tok = ttg.async_wait {num = 0 : i32}
      scf.yield %next_a, %next_b, %d, %next_tok : !ttg.memdesc<256x16xf16, #shared, #smem>, !ttg.memdesc<16x32xf16, #shared1, #smem>, tensor<256x32xf32, #mma>, !ttg.async.token
    }
    tt.return %loop#2 : tensor<256x32xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 3: slice_n_only — gfx942, MFMA v3, !transA, !transB, async
// A [32, 16] f16, B [16, 256] f16, C/D [32, 256] f32
// prefetchWidth = (32, 128, 16) → N sliced (2 N-slices)
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: slice_n_only
// Prologue subslices
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<32x16xf16
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<16x128xf16
// CHECK: scf.for
// In-loop: remainder B at N offset 128
// CHECK: ttg.memdesc_subslice {{.*}}[0, 128] {{.*}} -> !ttg.memdesc<16x128xf16
// Two dots (2 N-slices); A is not sliced so reused
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @slice_n_only(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<32x16xf16, #blocked>,
      %b_tensor: tensor<16x256xf16, #blocked>) -> tensor<32x256xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<32x256xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<32x16xf16, #blocked>) -> !ttg.memdesc<32x16xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<16x256xf16, #blocked>) -> !ttg.memdesc<16x256xf16, #shared1, #smem>
    %tok_init = ttg.async_wait {num = 0 : i32}
    %loop:4 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init, %awt = %tok_init)
      -> (!ttg.memdesc<32x16xf16, #shared, #smem>, !ttg.memdesc<16x256xf16, #shared1, #smem>, tensor<32x256xf32, #mma>, !ttg.async.token) {
      %a_op = ttg.local_load %a token %awt : !ttg.memdesc<32x16xf16, #shared, #smem> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b token %awt : !ttg.memdesc<16x256xf16, #shared1, #smem> -> tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x256xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<32x16xf16, #blocked>) -> !ttg.memdesc<32x16xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<16x256xf16, #blocked>) -> !ttg.memdesc<16x256xf16, #shared1, #smem>
      %next_tok = ttg.async_wait {num = 0 : i32}
      scf.yield %next_a, %next_b, %d, %next_tok : !ttg.memdesc<32x16xf16, #shared, #smem>, !ttg.memdesc<16x256xf16, #shared1, #smem>, tensor<32x256xf32, #mma>, !ttg.async.token
    }
    tt.return %loop#2 : tensor<32x256xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 4: slice_m_and_n — gfx942, MFMA v3, !transA, !transB, async
// A [256, 16] f16, B [16, 64] f16, C/D [256, 64] f32
// prefetchWidth = (128, 32, 16) → M sliced (128<256), N sliced (32<64)
// 2 M-slices x 2 N-slices = 4 dots total
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: slice_m_and_n
// Prologue subslices for prefetch
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<128x16xf16
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<16x32xf16
// CHECK: scf.for
// Four dots (2 M-slices x 2 N-slices x 1 K-slice)
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @slice_m_and_n(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<256x16xf16, #blocked>,
      %b_tensor: tensor<16x64xf16, #blocked>) -> tensor<256x64xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<256x16xf16, #blocked>) -> !ttg.memdesc<256x16xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<16x64xf16, #blocked>) -> !ttg.memdesc<16x64xf16, #shared1, #smem>
    %tok_init = ttg.async_wait {num = 0 : i32}
    %loop:4 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init, %awt = %tok_init)
      -> (!ttg.memdesc<256x16xf16, #shared, #smem>, !ttg.memdesc<16x64xf16, #shared1, #smem>, tensor<256x64xf32, #mma>, !ttg.async.token) {
      %a_op = ttg.local_load %a token %awt : !ttg.memdesc<256x16xf16, #shared, #smem> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b token %awt : !ttg.memdesc<16x64xf16, #shared1, #smem> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x64xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<256x16xf16, #blocked>) -> !ttg.memdesc<256x16xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<16x64xf16, #blocked>) -> !ttg.memdesc<16x64xf16, #shared1, #smem>
      %next_tok = ttg.async_wait {num = 0 : i32}
      scf.yield %next_a, %next_b, %d, %next_tok : !ttg.memdesc<256x16xf16, #shared, #smem>, !ttg.memdesc<16x64xf16, #shared1, #smem>, tensor<256x64xf32, #mma>, !ttg.async.token
    }
    tt.return %loop#2 : tensor<256x64xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 5: slice_n_and_k — gfx942, MFMA v3, transA=true, !transB, async
// A [128, 128] f16, B [128, 128] f16, C/D [128, 128] f32
// transA=true (A shared order=[0,1]): m=4, k=4 initially → prefetchWidth = (128, 32, 64)
// M unsliced (128=128), N sliced (32<128)=4 slices, K sliced (64<128)=2 slices
// 1 M x 4 N x 2 K = 8 dots
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared_at = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: slice_n_and_k
// Prologue: prefetch subslices exist
// CHECK: ttg.memdesc_subslice
// CHECK: ttg.local_load
// CHECK: scf.for
// 8 dots (1M x 4N x 2K)
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @slice_n_and_k(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<128x128xf16, #blocked>,
      %b_tensor: tensor<128x128xf16, #blocked>) -> tensor<128x128xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_at, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_b, #smem>
    %tok_init = ttg.async_wait {num = 0 : i32}
    %loop:4 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init, %awt = %tok_init)
      -> (!ttg.memdesc<128x128xf16, #shared_at, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, tensor<128x128xf32, #mma>, !ttg.async.token) {
      %a_op = ttg.local_load %a token %awt : !ttg.memdesc<128x128xf16, #shared_at, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b token %awt : !ttg.memdesc<128x128xf16, #shared_b, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_at, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_b, #smem>
      %next_tok = ttg.async_wait {num = 0 : i32}
      scf.yield %next_a, %next_b, %d, %next_tok : !ttg.memdesc<128x128xf16, #shared_at, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, tensor<128x128xf32, #mma>, !ttg.async.token
    }
    tt.return %loop#2 : tensor<128x128xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 6: slice_m_and_k — gfx942, MFMA v3, transA=true, !transB, async
// A [256, 128] f16, B [128, 32] f16, C/D [256, 32] f32
// prefetchWidth = (128, 32, 64) → M sliced (128<256), K sliced (64<128)
// 2 M x 1 N x 2 K = 4 dots
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared_at = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: slice_m_and_k
// Prologue subslices
// CHECK: ttg.memdesc_subslice
// CHECK: scf.for
// 4 dots (2M x 1N x 2K)
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @slice_m_and_k(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<256x128xf16, #blocked>,
      %b_tensor: tensor<128x32xf16, #blocked>) -> tensor<256x32xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<256x32xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared_at, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared_b, #smem>
    %tok_init = ttg.async_wait {num = 0 : i32}
    %loop:4 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init, %awt = %tok_init)
      -> (!ttg.memdesc<256x128xf16, #shared_at, #smem>, !ttg.memdesc<128x32xf16, #shared_b, #smem>, tensor<256x32xf32, #mma>, !ttg.async.token) {
      %a_op = ttg.local_load %a token %awt : !ttg.memdesc<256x128xf16, #shared_at, #smem> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b token %awt : !ttg.memdesc<128x32xf16, #shared_b, #smem> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x32xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared_at, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared_b, #smem>
      %next_tok = ttg.async_wait {num = 0 : i32}
      scf.yield %next_a, %next_b, %d, %next_tok : !ttg.memdesc<256x128xf16, #shared_at, #smem>, !ttg.memdesc<128x32xf16, #shared_b, #smem>, tensor<256x32xf32, #mma>, !ttg.async.token
    }
    tt.return %loop#2 : tensor<256x32xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 7: slice_m_n_k — gfx950, MFMA v3, transA=true, !transB, async
// A [256, 128] f16, B [128, 128] f16, C/D [256, 128] f32
// prefetchWidth = (128, 32, 64) → all three sliced
// 2 M x 4 N x 2 K = 16 dots
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared_at = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: slice_m_n_k
// Prologue
// CHECK: ttg.memdesc_subslice
// CHECK: scf.for
// 16 dots (2M x 4N x 2K)
// CHECK-COUNT-16: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @slice_m_n_k(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<256x128xf16, #blocked>,
      %b_tensor: tensor<128x128xf16, #blocked>) -> tensor<256x128xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared_at, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_b, #smem>
    %tok_init = ttg.async_wait {num = 0 : i32}
    %loop:4 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init, %awt = %tok_init)
      -> (!ttg.memdesc<256x128xf16, #shared_at, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, tensor<256x128xf32, #mma>, !ttg.async.token) {
      %a_op = ttg.local_load %a token %awt : !ttg.memdesc<256x128xf16, #shared_at, #smem> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b token %awt : !ttg.memdesc<128x128xf16, #shared_b, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x128xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared_at, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_b, #smem>
      %next_tok = ttg.async_wait {num = 0 : i32}
      scf.yield %next_a, %next_b, %d, %next_tok : !ttg.memdesc<256x128xf16, #shared_at, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, tensor<256x128xf32, #mma>, !ttg.async.token
    }
    tt.return %loop#2 : tensor<256x128xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 8: no_slicing_possible — gfx942, MFMA v3, !transA, !transB, async
// A [32, 16] f16, B [16, 32] f16, C/D [32, 32] f32
// prefetchWidth = (32, 32, 16) = full tile. No slicing.
// Pass still adds prologue + prefetch iter_args but single dot (no split/join).
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: no_slicing_possible
// Prologue: subslice covers full tile (same as original shape)
// CHECK: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<32x16xf16
// CHECK: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<16x32xf16
// CHECK: scf.for
// Only one dot (no slicing, no split/join)
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @no_slicing_possible(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<32x16xf16, #blocked>,
      %b_tensor: tensor<16x32xf16, #blocked>) -> tensor<32x32xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<32x16xf16, #blocked>) -> !ttg.memdesc<32x16xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<16x32xf16, #blocked>) -> !ttg.memdesc<16x32xf16, #shared1, #smem>
    %tok_init = ttg.async_wait {num = 0 : i32}
    %loop:4 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init, %awt = %tok_init)
      -> (!ttg.memdesc<32x16xf16, #shared, #smem>, !ttg.memdesc<16x32xf16, #shared1, #smem>, tensor<32x32xf32, #mma>, !ttg.async.token) {
      %a_op = ttg.local_load %a token %awt : !ttg.memdesc<32x16xf16, #shared, #smem> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b token %awt : !ttg.memdesc<16x32xf16, #shared1, #smem> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<32x16xf16, #blocked>) -> !ttg.memdesc<32x16xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<16x32xf16, #blocked>) -> !ttg.memdesc<16x32xf16, #shared1, #smem>
      %next_tok = ttg.async_wait {num = 0 : i32}
      scf.yield %next_a, %next_b, %d, %next_tok : !ttg.memdesc<32x16xf16, #shared, #smem>, !ttg.memdesc<16x32xf16, #shared1, #smem>, tensor<32x32xf32, #mma>, !ttg.async.token
    }
    tt.return %loop#2 : tensor<32x32xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 9: two_dots_in_loop_fails — gfx942, MFMA v3
// Two tt.dot ops in loop body → initialize() rejects (dotsInFor.size() > 1)
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: two_dots_in_loop_fails
// CHECK: scf.for
// CHECK-NOT: ttg.memdesc_subslice
// Exactly two dots remain (unchanged)
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @two_dots_in_loop_fails(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<32x16xf16, #blocked>,
      %b1_tensor: tensor<16x32xf16, #blocked>,
      %b2_tensor: tensor<16x32xf16, #blocked>) -> tensor<32x32xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<32x16xf16, #blocked>) -> !ttg.memdesc<32x16xf16, #shared, #smem>
    %b1_init = ttg.local_alloc %b1_tensor : (tensor<16x32xf16, #blocked>) -> !ttg.memdesc<16x32xf16, #shared1, #smem>
    %b2_init = ttg.local_alloc %b2_tensor : (tensor<16x32xf16, #blocked>) -> !ttg.memdesc<16x32xf16, #shared1, #smem>
    %loop:5 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b1 = %b1_init, %b2 = %b2_init, %c1 = %c_init, %c2 = %c_init)
      -> (!ttg.memdesc<32x16xf16, #shared, #smem>, !ttg.memdesc<16x32xf16, #shared1, #smem>, !ttg.memdesc<16x32xf16, #shared1, #smem>, tensor<32x32xf32, #mma>, tensor<32x32xf32, #mma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<32x16xf16, #shared, #smem> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b1_op = ttg.local_load %b1 : !ttg.memdesc<16x32xf16, #shared1, #smem> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %b2_op = ttg.local_load %b2 : !ttg.memdesc<16x32xf16, #shared1, #smem> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d1 = tt.dot %a_op, %b1_op, %c1 : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
      %d2 = tt.dot %a_op, %b2_op, %c2 : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<32x16xf16, #blocked>) -> !ttg.memdesc<32x16xf16, #shared, #smem>
      %next_b1 = ttg.local_alloc %b1_tensor : (tensor<16x32xf16, #blocked>) -> !ttg.memdesc<16x32xf16, #shared1, #smem>
      %next_b2 = ttg.local_alloc %b2_tensor : (tensor<16x32xf16, #blocked>) -> !ttg.memdesc<16x32xf16, #shared1, #smem>
      scf.yield %next_a, %next_b1, %next_b2, %d1, %d2 : !ttg.memdesc<32x16xf16, #shared, #smem>, !ttg.memdesc<16x32xf16, #shared1, #smem>, !ttg.memdesc<16x32xf16, #shared1, #smem>, tensor<32x32xf32, #mma>, tensor<32x32xf32, #mma>
    }
    tt.return %loop#3 : tensor<32x32xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 10: dot_scaled_with_scales_fails — gfx950, MFMA v4
// DotScaledOp with scale operands → initialize() rejects
// ============================================================================
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]], warp = [[0, 0], [32, 0]], block = []}>

// CHECK-LABEL: dot_scaled_with_scales_fails
// CHECK: scf.for
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.dot_scaled
// CHECK-NOT: tt.dot_scaled
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @dot_scaled_with_scales_fails(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<64x64xf8E4M3FN, #blocked>,
      %b_tensor: tensor<64x16xf8E4M3FN, #blocked>,
      %a_scale: tensor<64x2xi8, #linear>,
      %b_scale: tensor<16x2xi8, #linear>) -> tensor<64x16xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<64x16xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x16xf8E4M3FN, #shared1, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>, !ttg.memdesc<64x16xf8E4M3FN, #shared1, #smem>, tensor<64x16xf32, #mma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem> -> tensor<64x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<64x16xf8E4M3FN, #shared1, #smem> -> tensor<64x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %d = tt.dot_scaled %a_op scale %a_scale, %b_op scale %b_scale, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<64x2xi8, #linear> * tensor<64x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<16x2xi8, #linear> -> tensor<64x16xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<64x16xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x16xf8E4M3FN, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>, !ttg.memdesc<64x16xf8E4M3FN, #shared1, #smem>, tensor<64x16xf32, #mma>
    }
    tt.return %loop#2 : tensor<64x16xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 11: async_token_at_top_not_loop_carried_fails — gfx942, MFMA v3
// Memdescs are NOT loop iter_args (created inside loop via memdesc_index).
// getIncomingOp returns null → dot not prefetched → loop unchanged.
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: async_token_at_top_not_loop_carried_fails
// CHECK: scf.for
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @async_token_at_top_not_loop_carried_fails(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<32x16xf16, #blocked>,
      %b_tensor: tensor<16x32xf16, #blocked>,
      %a_alloc: !ttg.memdesc<2x32x16xf16, #shared, #smem, mutable>,
      %b_alloc: !ttg.memdesc<2x16x32xf16, #shared1, #smem, mutable>) -> tensor<32x32xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c0 = arith.constant 0 : i32
    %loop:2 = scf.for %iv = %lb to %ub step %step
      iter_args(%c = %c_init, %idx = %c0)
      -> (tensor<32x32xf32, #mma>, i32) {
      %a_desc = ttg.memdesc_index %a_alloc[%idx] : !ttg.memdesc<2x32x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable>
      %b_desc = ttg.memdesc_index %b_alloc[%idx] : !ttg.memdesc<2x16x32xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared1, #smem, mutable>
      %a_op = ttg.local_load %a_desc : !ttg.memdesc<32x16xf16, #shared, #smem, mutable> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b_desc : !ttg.memdesc<16x32xf16, #shared1, #smem, mutable> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
      %c1 = arith.constant 1 : i32
      %next_idx = arith.addi %idx, %c1 : i32
      scf.yield %d, %next_idx : tensor<32x32xf32, #mma>, i32
    }
    tt.return %loop#0 : tensor<32x32xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 12: without_async_wait_mnk_slicing — gfx942, MFMA v3, transA=true, sync
// A [256, 128] f16, B [128, 128] f16, C/D [256, 128] f32
// prefetchWidth = (128, 32, 64) → all three sliced, sync path (no token)
// 2 M x 4 N x 2 K = 16 dots
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared_at = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: without_async_wait_mnk_slicing
// Prologue
// CHECK: ttg.memdesc_subslice
// CHECK: ttg.local_load
// CHECK: scf.for
// 16 dots (2M x 4N x 2K)
// CHECK-COUNT-16: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @without_async_wait_mnk_slicing(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<256x128xf16, #blocked>,
      %b_tensor: tensor<128x128xf16, #blocked>) -> tensor<256x128xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared_at, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_b, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<256x128xf16, #shared_at, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, tensor<256x128xf32, #mma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<256x128xf16, #shared_at, #smem> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<128x128xf16, #shared_b, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x128xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared_at, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_b, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<256x128xf16, #shared_at, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, tensor<256x128xf32, #mma>
    }
    tt.return %loop#2 : tensor<256x128xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 13: fp8 — gfx950, MFMA v4, !transA, !transB, sync
// A [64, 256] f8E4M3FN, B [256, 16] f8E4M3FN, C/D [64, 16] f32
// prefetchWidth = (64, 16, 128) → K sliced (2 K-slices)
// ============================================================================
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: fp8_k_sliced
// Prologue
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<64x128xf8E4M3FN
// CHECK-DAG: ttg.memdesc_subslice {{.*}}[0, 0] {{.*}} -> !ttg.memdesc<128x16xf8E4M3FN
// CHECK: scf.for
// Two dots (2 K-slices)
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @fp8_k_sliced(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<64x256xf8E4M3FN, #blocked>,
      %b_tensor: tensor<256x16xf8E4M3FN, #blocked>) -> tensor<64x16xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<64x256xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x256xf8E4M3FN, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<256x16xf8E4M3FN, #blocked>) -> !ttg.memdesc<256x16xf8E4M3FN, #shared1, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<64x256xf8E4M3FN, #shared, #smem>, !ttg.memdesc<256x16xf8E4M3FN, #shared1, #smem>, tensor<64x16xf32, #mma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<64x256xf8E4M3FN, #shared, #smem> -> tensor<64x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<256x16xf8E4M3FN, #shared1, #smem> -> tensor<256x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<64x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<256x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<64x16xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<64x256xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x256xf8E4M3FN, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<256x16xf8E4M3FN, #blocked>) -> !ttg.memdesc<256x16xf8E4M3FN, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<64x256xf8E4M3FN, #shared, #smem>, !ttg.memdesc<256x16xf8E4M3FN, #shared1, #smem>, tensor<64x16xf32, #mma>
    }
    tt.return %loop#2 : tensor<64x16xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 14: dot_scaled_no_scales — gfx950, MFMA v4, !transA, !transB, sync
// Same shapes as fp8 test but using tt.dot_scaled without scale operands.
// A [64, 256] f8E4M3FN, B [256, 16] f8E4M3FN, C/D [64, 16] f32
// prefetchWidth = (64, 16, 128) → K sliced
// ============================================================================
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: dot_scaled_no_scales
// Prologue
// CHECK: ttg.memdesc_subslice
// CHECK: scf.for
// Two dot_scaled ops (2 K-slices)
// CHECK: tt.dot_scaled
// CHECK: tt.dot_scaled
// CHECK-NOT: tt.dot_scaled
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @dot_scaled_no_scales(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<64x256xf8E4M3FN, #blocked>,
      %b_tensor: tensor<256x16xf8E4M3FN, #blocked>) -> tensor<64x16xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<64x256xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x256xf8E4M3FN, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<256x16xf8E4M3FN, #blocked>) -> !ttg.memdesc<256x16xf8E4M3FN, #shared1, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<64x256xf8E4M3FN, #shared, #smem>, !ttg.memdesc<256x16xf8E4M3FN, #shared1, #smem>, tensor<64x16xf32, #mma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<64x256xf8E4M3FN, #shared, #smem> -> tensor<64x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<256x16xf8E4M3FN, #shared1, #smem> -> tensor<256x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %d = tt.dot_scaled %a_op, %b_op, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<256x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<64x16xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<64x256xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x256xf8E4M3FN, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<256x16xf8E4M3FN, #blocked>) -> !ttg.memdesc<256x16xf8E4M3FN, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<64x256xf8E4M3FN, #shared, #smem>, !ttg.memdesc<256x16xf8E4M3FN, #shared1, #smem>, tensor<64x16xf32, #mma>
    }
    tt.return %loop#2 : tensor<64x16xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 15: wmma — gfx1250, WMMA v3, !transA, !transB, sync
// A [128, 64] f16, B [64, 128] f16, C/D [128, 128] f32
// warpsPerCTA=[2,2], baseM=32, baseN=32, baseK=32, numInsts=8
// prefetchWidth = (128, 64, 32) → N sliced (64<128), K sliced (32<64)
// 1 M x 2 N x 2 K = 4 dots
// ============================================================================
#wmma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: wmma_nk_sliced
// Prologue
// CHECK: ttg.memdesc_subslice
// CHECK: scf.for
// 8 dots (1M x 4N x 2K)
// CHECK-COUNT-8: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_nk_sliced(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<128x64xf16, #blocked>,
      %b_tensor: tensor<64x128xf16, #blocked>) -> tensor<128x128xf32, #wmma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #wmma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, tensor<128x128xf32, #wmma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma, kWidth = 8}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<64x128xf16, #shared1, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 8}>> -> tensor<128x128xf32, #wmma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, tensor<128x128xf32, #wmma>
    }
    tt.return %loop#2 : tensor<128x128xf32, #wmma>
  }
}

// -----

// ============================================================================
// Test 16: fp_to_fp_between_load_and_dot_mnk — gfx942, MFMA v3, transA=true, sync
// A [256, 128] f8E5M2 in shared → local_load → fp_to_fp f16 → dot
// B [128, 128] f16 in shared → local_load → dot
// prefetchWidth = (128, 32, 64) → all three sliced, 16 dots
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared_at = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: fp_to_fp_between_load_and_dot_mnk
// Prologue: subslice + local_load + fp_to_fp
// CHECK: ttg.memdesc_subslice
// CHECK: ttg.local_load
// CHECK: tt.fp_to_fp
// CHECK: scf.for
// In-loop: sliced local_loads and fp_to_fp feeding dots
// CHECK: tt.fp_to_fp
// CHECK: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @fp_to_fp_between_load_and_dot_mnk(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<256x128xf8E5M2, #blocked>,
      %b_tensor: tensor<128x128xf16, #blocked>) -> tensor<256x128xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<256x128xf8E5M2, #blocked>) -> !ttg.memdesc<256x128xf8E5M2, #shared_at, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_b, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<256x128xf8E5M2, #shared_at, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, tensor<256x128xf32, #mma>) {
      %a_f8 = ttg.local_load %a : !ttg.memdesc<256x128xf8E5M2, #shared_at, #smem> -> tensor<256x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %a_op = tt.fp_to_fp %a_f8 : tensor<256x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<128x128xf16, #shared_b, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x128xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<256x128xf8E5M2, #blocked>) -> !ttg.memdesc<256x128xf8E5M2, #shared_at, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_b, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<256x128xf8E5M2, #shared_at, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, tensor<256x128xf32, #mma>
    }
    tt.return %loop#2 : tensor<256x128xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 17: fp_to_fp_between_load_and_dot_async — gfx942, MFMA v3, transA=true, async
// Same as test 16 but with async wait tokens (loop-carried).
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared_at = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: fp_to_fp_between_load_and_dot_async
// Prologue: subslice + local_load with token + fp_to_fp
// CHECK: ttg.memdesc_subslice
// CHECK: ttg.local_load
// CHECK: tt.fp_to_fp
// CHECK: scf.for
// In-loop: sliced fp_to_fp feeding dots
// CHECK: tt.fp_to_fp
// CHECK: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @fp_to_fp_between_load_and_dot_async(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<256x128xf8E5M2, #blocked>,
      %b_tensor: tensor<128x128xf16, #blocked>) -> tensor<256x128xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<256x128xf8E5M2, #blocked>) -> !ttg.memdesc<256x128xf8E5M2, #shared_at, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_b, #smem>
    %tok_init = ttg.async_wait {num = 0 : i32}
    %loop:4 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init, %awt = %tok_init)
      -> (!ttg.memdesc<256x128xf8E5M2, #shared_at, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, tensor<256x128xf32, #mma>, !ttg.async.token) {
      %a_f8 = ttg.local_load %a token %awt : !ttg.memdesc<256x128xf8E5M2, #shared_at, #smem> -> tensor<256x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %a_op = tt.fp_to_fp %a_f8 : tensor<256x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b token %awt : !ttg.memdesc<128x128xf16, #shared_b, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x128xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<256x128xf8E5M2, #blocked>) -> !ttg.memdesc<256x128xf8E5M2, #shared_at, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared_b, #smem>
      %next_tok = ttg.async_wait {num = 0 : i32}
      scf.yield %next_a, %next_b, %d, %next_tok : !ttg.memdesc<256x128xf8E5M2, #shared_at, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, tensor<256x128xf32, #mma>, !ttg.async.token
    }
    tt.return %loop#2 : tensor<256x128xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 18: default_numinsts_mfma_f16 — exercises computeDefaultNumInsts (num-insts=0)
// gfx942, MFMA v3, f16, !transA, !transB. A [32, 256], B [256, 32], C/D [32, 32].
// warpsPerCTA=[2,2] -> maxM = maxN = 1 (no M/N slicing), maxK = 256/16 = 16.
// instrShape 16x16x16 = 4096 FMAs; gfx942 f16 throughput 256/cyc -> 16 cyc/MFMA;
// with 64-cyc LDS latency, numInsts = pow2ceil(2*(64/16)) = 8. K slice spans 8
// MFMAs -> 16/8 = 2 K-slices (2 dots).
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// DEFAULT-LABEL: default_numinsts_mfma_f16
// DEFAULT: scf.for
// DEFAULT-COUNT-2: tt.dot
// DEFAULT-NOT: tt.dot
// DEFAULT: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @default_numinsts_mfma_f16(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<32x256xf16, #blocked>,
      %b_tensor: tensor<256x32xf16, #blocked>) -> tensor<32x32xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<32x256xf16, #blocked>) -> !ttg.memdesc<32x256xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<256x32xf16, #blocked>) -> !ttg.memdesc<256x32xf16, #shared1, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<32x256xf16, #shared, #smem>, !ttg.memdesc<256x32xf16, #shared1, #smem>, tensor<32x32xf32, #mma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<32x256xf16, #shared, #smem> -> tensor<32x256xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<256x32xf16, #shared1, #smem> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<32x256xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<256x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<32x256xf16, #blocked>) -> !ttg.memdesc<32x256xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<256x32xf16, #blocked>) -> !ttg.memdesc<256x32xf16, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<32x256xf16, #shared, #smem>, !ttg.memdesc<256x32xf16, #shared1, #smem>, tensor<32x32xf32, #mma>
    }
    tt.return %loop#2 : tensor<32x32xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 19: default_numinsts_mfma_fp8 — exercises computeDefaultNumInsts (num-insts=0)
// gfx950, MFMA v4, f8E4M3FN, !transA, !transB. A [64, 1024], B [1024, 16], C/D [64, 16].
// warpsPerCTA=[4,1] -> maxM = maxN = 1, maxK = 1024/32 = 32.
// instrShape 16x16x32 = 8192 FMAs; gfx950 fp8 throughput 1024/cyc -> 8 cyc/MFMA;
// with 64-cyc LDS latency, numInsts = pow2ceil(2*(64/8)) = 16. K slice spans 16
// MFMAs -> 32/16 = 2 K-slices (2 dots). The higher fp8 matrix-unit throughput
// (vs f16) needs more instructions in flight to hide the same latency.
// ============================================================================
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// DEFAULT-LABEL: default_numinsts_mfma_fp8
// DEFAULT: scf.for
// DEFAULT-COUNT-2: tt.dot
// DEFAULT-NOT: tt.dot
// DEFAULT: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @default_numinsts_mfma_fp8(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<64x1024xf8E4M3FN, #blocked>,
      %b_tensor: tensor<1024x16xf8E4M3FN, #blocked>) -> tensor<64x16xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<64x1024xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x1024xf8E4M3FN, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<1024x16xf8E4M3FN, #blocked>) -> !ttg.memdesc<1024x16xf8E4M3FN, #shared1, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<64x1024xf8E4M3FN, #shared, #smem>, !ttg.memdesc<1024x16xf8E4M3FN, #shared1, #smem>, tensor<64x16xf32, #mma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<64x1024xf8E4M3FN, #shared, #smem> -> tensor<64x1024xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<1024x16xf8E4M3FN, #shared1, #smem> -> tensor<1024x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<64x1024xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<1024x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<64x16xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<64x1024xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x1024xf8E4M3FN, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<1024x16xf8E4M3FN, #blocked>) -> !ttg.memdesc<1024x16xf8E4M3FN, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<64x1024xf8E4M3FN, #shared, #smem>, !ttg.memdesc<1024x16xf8E4M3FN, #shared1, #smem>, tensor<64x16xf32, #mma>
    }
    tt.return %loop#2 : tensor<64x16xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 20: default_numinsts_wmma_f16 — exercises computeDefaultNumInsts (num-insts=0)
// gfx1250, WMMA v3, f16, !transA, !transB. A [32, 2048], B [2048, 32], C/D [32, 32].
// warpsPerCTA=[2,2] -> maxM = maxN = 1, maxK = 2048/32 = 64.
// instrShape 16x16x32 = 8192 FMAs; gfx1250 f16 throughput 1024/cyc -> 8 cyc/WMMA;
// with RDNA's higher 96-cyc LDS latency, numInsts = pow2ceil(2*(96/8)) = 32. K
// slice spans 32 WMMAs -> 64/32 = 2 K-slices (2 dots). The larger numInsts (vs
// the gfx950 fp8 case at the same throughput) reflects the arch's higher LDS
// latency.
// ============================================================================
#wmma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// DEFAULT-LABEL: default_numinsts_wmma_f16
// DEFAULT: scf.for
// DEFAULT-COUNT-2: tt.dot
// DEFAULT-NOT: tt.dot
// DEFAULT: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @default_numinsts_wmma_f16(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<32x2048xf16, #blocked>,
      %b_tensor: tensor<2048x32xf16, #blocked>) -> tensor<32x32xf32, #wmma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #wmma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<32x2048xf16, #blocked>) -> !ttg.memdesc<32x2048xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<2048x32xf16, #blocked>) -> !ttg.memdesc<2048x32xf16, #shared1, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<32x2048xf16, #shared, #smem>, !ttg.memdesc<2048x32xf16, #shared1, #smem>, tensor<32x32xf32, #wmma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<32x2048xf16, #shared, #smem> -> tensor<32x2048xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma, kWidth = 8}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<2048x32xf16, #shared1, #smem> -> tensor<2048x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<32x2048xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma, kWidth = 8}>> * tensor<2048x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 8}>> -> tensor<32x32xf32, #wmma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<32x2048xf16, #blocked>) -> !ttg.memdesc<32x2048xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<2048x32xf16, #blocked>) -> !ttg.memdesc<2048x32xf16, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<32x2048xf16, #shared, #smem>, !ttg.memdesc<2048x32xf16, #shared1, #smem>, tensor<32x32xf32, #wmma>
    }
    tt.return %loop#2 : tensor<32x32xf32, #wmma>
  }
}

// -----

// ============================================================================
// Test 21: promote_async_memdesc_index — gfx942, MFMA v3, async, NOT loop-carried
// The local_load sources (memdesc_index on a ring buffer) and the async_wait
// token are in-body expressions, not loop iter_args. The pass promotes them:
// it clones the memdesc_index/async_wait chain into the prologue (induction var
// -> lower bound) and into the yield (induction var advanced), so the loop is
// still prefetched. This is the positive counterpart to Test 11.
// A [32, 128] f16, B [128, 32] f16 -> K width 64 -> 2 K-slices (2 dots).
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: promote_async_memdesc_index
// Prologue materializes the promoted memdesc_index + async_wait + subslice.
// CHECK: ttg.memdesc_index
// CHECK: ttg.async_wait
// CHECK: ttg.memdesc_subslice {{.*}} -> !ttg.memdesc<32x64xf16
// CHECK: ttg.local_load
// CHECK: scf.for
// Two dots (2 K-slices) prove the promote path prefetched the loop.
// CHECK: tt.dot
// CHECK: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @promote_async_memdesc_index(%lb: index, %ub: index, %step: index,
      %a_alloc: !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>,
      %b_alloc: !ttg.memdesc<2x128x32xf16, #shared1, #smem, mutable>) -> tensor<32x32xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c0 = arith.constant 0 : i32
    %loop:2 = scf.for %iv = %lb to %ub step %step
      iter_args(%c = %c_init, %idx = %c0)
      -> (tensor<32x32xf32, #mma>, i32) {
      %tok = ttg.async_wait {num = 0 : i32}
      %a_desc = ttg.memdesc_index %a_alloc[%idx] : !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
      %b_desc = ttg.memdesc_index %b_alloc[%idx] : !ttg.memdesc<2x128x32xf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %a_op = ttg.local_load %a_desc token %tok : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b_desc token %tok : !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
      %c1 = arith.constant 1 : i32
      %next_idx = arith.addi %idx, %c1 : i32
      scf.yield %d, %next_idx : tensor<32x32xf32, #mma>, i32
    }
    tt.return %loop#0 : tensor<32x32xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 22: cond_barrier_skips_pingpong — gfx942, MFMA v3
// A loop preceded by amdg.cond_barrier signals that PingPong scheduling already
// owns the loop, so the pass skips it (hasPrecedingCondBarrier). The loop must
// be left unchanged (no subslices, single dot).
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: cond_barrier_skips_pingpong
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @cond_barrier_skips_pingpong(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<32x128xf16, #blocked>,
      %b_tensor: tensor<128x32xf16, #blocked>) -> tensor<32x32xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %true = arith.constant true
    %a_init = ttg.local_alloc %a_tensor : (tensor<32x128xf16, #blocked>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem>
    amdg.cond_barrier %true
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<32x128xf16, #shared, #smem>, !ttg.memdesc<128x32xf16, #shared1, #smem>, tensor<32x32xf32, #mma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<128x32xf16, #shared1, #smem> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<32x128xf16, #blocked>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<32x128xf16, #shared, #smem>, !ttg.memdesc<128x32xf16, #shared1, #smem>, tensor<32x32xf32, #mma>
    }
    tt.return %loop#2 : tensor<32x32xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 23: non_unit_tiles_per_warp — gfx950, MFMA v4, tilesPerWarp = [2, 2]
// The slice-size math assumes the canonical instrShape * warpsPerCTA grid, so
// MFMA layouts with non-unit tilesPerWarp are skipped
// (computePrefetchWidthForDotType returns false). Loop left unchanged.
// ============================================================================
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], tilesPerWarp = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: non_unit_tiles_per_warp
// CHECK: scf.for
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @non_unit_tiles_per_warp(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<32x128xf8E4M3FN, #blocked>,
      %b_tensor: tensor<128x128xf8E4M3FN, #blocked>) -> tensor<32x128xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<32x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<32x128xf8E4M3FN, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<128x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<32x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, tensor<32x128xf32, #mma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<32x128xf8E4M3FN, #shared, #smem> -> tensor<32x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<32x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<32x128xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<32x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<32x128xf8E4M3FN, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<128x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<32x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, tensor<32x128xf32, #mma>
    }
    tt.return %loop#2 : tensor<32x128xf32, #mma>
  }
}

// -----

// ============================================================================
// Test 24: warp_swizzled_wmma — gfx1250, WMMA v3, non-permutation ctaLayout
// WMMA stores warp distribution as a general linear layout. When ctaLayout is
// not a permutation matrix (warp-swizzled, e.g. to resolve partition conflicts)
// warpsPerCTA is not well-defined, so the pass bails
// (isPermutationMatrixLayout check). Loop left unchanged.
// ============================================================================
#wmma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 1]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK-LABEL: warp_swizzled_wmma
// CHECK: scf.for
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @warp_swizzled_wmma(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<128x64xf16, #blocked>,
      %b_tensor: tensor<64x128xf16, #blocked>) -> tensor<128x128xf32, #wmma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #wmma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, tensor<128x128xf32, #wmma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma, kWidth = 8}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<64x128xf16, #shared1, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 8}>> -> tensor<128x128xf32, #wmma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, tensor<128x128xf32, #wmma>
    }
    tt.return %loop#2 : tensor<128x128xf32, #wmma>
  }
}

// -----

// ============================================================================
// Test 25: batched_dot — gfx942, MFMA v3, 3-D (batched) operands
// The slicing logic assumes 2-D operands, so batched dots (rank != 2) are
// skipped in initialize(). Loop left unchanged.
// ============================================================================
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [2, 1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 2, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 16, 4], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>

// CHECK-LABEL: batched_dot
// CHECK: scf.for
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @batched_dot(%lb: index, %ub: index, %step: index,
      %a_tensor: tensor<2x32x128xf16, #blocked>,
      %b_tensor: tensor<2x128x32xf16, #blocked>) -> tensor<2x32x32xf32, #mma> {
    %c_init = arith.constant dense<0.000000e+00> : tensor<2x32x32xf32, #mma>
    %a_init = ttg.local_alloc %a_tensor : (tensor<2x32x128xf16, #blocked>) -> !ttg.memdesc<2x32x128xf16, #shared, #smem>
    %b_init = ttg.local_alloc %b_tensor : (tensor<2x128x32xf16, #blocked>) -> !ttg.memdesc<2x128x32xf16, #shared1, #smem>
    %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a = %a_init, %b = %b_init, %c = %c_init)
      -> (!ttg.memdesc<2x32x128xf16, #shared, #smem>, !ttg.memdesc<2x128x32xf16, #shared1, #smem>, tensor<2x32x32xf32, #mma>) {
      %a_op = ttg.local_load %a : !ttg.memdesc<2x32x128xf16, #shared, #smem> -> tensor<2x32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_op = ttg.local_load %b : !ttg.memdesc<2x128x32xf16, #shared1, #smem> -> tensor<2x128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_op, %b_op, %c : tensor<2x32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<2x128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<2x32x32xf32, #mma>
      %next_a = ttg.local_alloc %a_tensor : (tensor<2x32x128xf16, #blocked>) -> !ttg.memdesc<2x32x128xf16, #shared, #smem>
      %next_b = ttg.local_alloc %b_tensor : (tensor<2x128x32xf16, #blocked>) -> !ttg.memdesc<2x128x32xf16, #shared1, #smem>
      scf.yield %next_a, %next_b, %d : !ttg.memdesc<2x32x128xf16, #shared, #smem>, !ttg.memdesc<2x128x32xf16, #shared1, #smem>, tensor<2x32x32xf32, #mma>
    }
    tt.return %loop#2 : tensor<2x32x32xf32, #mma>
  }
}
