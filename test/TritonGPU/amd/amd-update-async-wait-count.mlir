// RUN: triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=gfx-arch=gfx1250 | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=gfx-arch=gfx1250 --convert-scf-to-cf --allocate-shared-memory --triton-amdgpu-membar=gfx-arch=gfx1250 | FileCheck %s --check-prefix=SCOPE

// Simple case without any branching

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.padded_shared<[4:+4] {order = [1, 0], shape=[16, 256]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: simple_waitcnt
  // SCOPE-LABEL: @simple_waitcnt(
  tt.func public @simple_waitcnt(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) -> tensor<128x16xf16, #blocked> {
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 4 direct to lds instructions (padding interval limits vec to 4)
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2

    // Wait on token %1: 4 instructions outstanding (second async_copy emits 4)
    // CHECK: amdg.async_wait {{.*}} {num_inst = 4
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // Wait on token %3: 0 instructions outstanding
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %10 = ttg.async_wait %3 {num = 0 : i32}
    // SCOPE: amdg.async_wait
    // SCOPE: amdg.async_wait
    // SCOPE-SAME: ttg.amdg.completion_scope = "shared"
    // SCOPE-NEXT: {{.*}} = tt.load
    %observed = tt.load %arg3 : tensor<128x16x!tt.ptr<f16>, #blocked>
    tt.return %observed : tensor<128x16xf16, #blocked>
  }
}

// -----

// Simple case with amdg.buffer_load_to_local

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.padded_shared<[16:+4] {order = [1, 0], shape = [16, 256]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: simple_buffer_load_to_local_waitcnt
  tt.func public @simple_buffer_load_to_local_waitcnt(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: tensor<128x16xi32, #blocked> {tt.contiguity = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: tensor<16x256xi32, #blocked1> {tt.contiguity = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg5: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>) {
    // Emits 1 direct to lds instruction
    %0 = amdg.buffer_load_to_local %arg0[%arg1] into %arg4 : !tt.ptr<f16>[tensor<128x16xi32, #blocked>]  -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = amdg.buffer_load_to_local %arg2[%arg3] into %arg5 : !tt.ptr<f16>[tensor<16x256xi32, #blocked1>]  -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    // Wait on token %1: 2 instructions outstanding (second buffer_load emits 2)
    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %4 = ttg.async_wait %1 {num = 0 : i32}
    // Wait on token %3: 0 instructions outstanding
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %5 = ttg.async_wait %3 {num = 0 : i32}
    tt.return
  }
}

// -----

// Same as simple_waitcnt but swapped async_waits

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: simple_waitcnt_reversed
  tt.func public @simple_waitcnt_reversed(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2

    // Wait on token %3: 0 instructions outstanding (nothing after %3)
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %9 = ttg.async_wait %3 {num = 0 : i32}
    // Wait on token %1: 2 instructions outstanding (second async_copy emits 2)
    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %10 = ttg.async_wait %1 {num = 0 : i32}
    tt.return
  }
}

// -----

// We should ignore tt.loads when counting

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: simple_waitcnt_with_tt_load
  tt.func public @simple_waitcnt_with_tt_load(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2

    %4 = tt.load %arg3 : tensor<128x16x!tt.ptr<f16>, #blocked>

    // Wait on token %1: 2 instructions outstanding (tt.load ignored, second async_copy emits 2)
    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // Wait on token %3: 0 instructions outstanding
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %10 = ttg.async_wait %3 {num = 0 : i32}
    tt.return
  }
}

// -----

// Simple loop without any interleaving loads so we expect waitcnt 0

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL wait_in_for_loop
  tt.func public @wait_in_for_loop(%arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %8:2 = scf.for %arg14 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg15 = %1, %arg16 = %3) -> (!ttg.async.token, !ttg.async.token)  : i32 {
      // min over tokens: %arg16 has 0 outstanding → overall 0
      // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 0
      %10 = ttg.async_wait %arg15, %arg16 {num = 2 : i32}
      %11 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %12 = ttg.async_commit_group tokens %11
      %13 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
      %14 = ttg.async_commit_group tokens %13
      scf.yield %12, %14: !ttg.async.token, !ttg.async.token
    }
    // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 0
    %9 = ttg.async_wait %8#0, %8#1 {num = 0 : i32}
    tt.return
  }
}

// -----

// Double buffering for 2 loads where the first one will emit 2 instructions and the second 1 instruction so we expect waitcnt 3 inside the loop and 0 in the epilogue

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL double_buffering
  tt.func public @double_buffering(%arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %7 = ttg.async_commit_group tokens %6
    %8:4 = scf.for %arg14 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg15 = %1, %arg16 = %5, %arg17 = %3, %arg18 = %7) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      // 2 loads per buffer: first emits 1, second emits 2 → 3 per iteration
      // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 3
      %10 = ttg.async_wait %arg15, %arg17 {num = 2 : i32}
      %11 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %12 = ttg.async_commit_group tokens %11
      %13 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
      %14 = ttg.async_commit_group tokens %13
      scf.yield %arg16, %12, %arg18, %14 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 0
    %9 = ttg.async_wait %8#0, %8#1, %8#2, %8#3 {num = 0 : i32}
    tt.return
  }
}
// -----

// Double buffering with async_wait inside scf.if

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: double_buffering_wait_in_if
  tt.func public @double_buffering_wait_in_if(%cond: i1, %arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %7 = ttg.async_commit_group tokens %6
    %8:4 = scf.for %arg14 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg15 = %1, %arg16 = %5, %arg17 = %3, %arg18 = %7) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
      %103 = scf.if %cond -> (!ttg.async.token) {
        // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 3
        %token1 = ttg.async_wait %arg15, %arg17 {num = 2 : i32}
        scf.yield %token1 : !ttg.async.token
      } else {
        // CHECK: amdg.async_wait {{.*}} {num_inst = 5
        %token2 = ttg.async_wait %arg15 {num = 1 : i32}
        scf.yield %token2 : !ttg.async.token
      }
      %11 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %12 = ttg.async_commit_group tokens %11
      %13 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
      %14 = ttg.async_commit_group tokens %13
      scf.yield %arg16, %12, %arg18, %14 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 0
    %9 = ttg.async_wait %8#0, %8#1, %8#2, %8#3 {num = 0 : i32}
    tt.return
  }
}

// -----

// Double buffering with async_wait and additional async_loads inside the scf.if

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: doube_buffering_wait_loads_in_if
  tt.func public @doube_buffering_wait_loads_in_if(%cond: i1, %arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %7 = ttg.async_commit_group tokens %6
    %8:4 = scf.for %arg14 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg15 = %1, %arg16 = %5, %arg17 = %3, %arg18 = %7) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %103 = scf.if %cond -> (!ttg.async.token) {
        %cond_load = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
        %cond_load_commit = ttg.async_commit_group tokens %cond_load
        // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 5
        %token1 = ttg.async_wait %arg15, %arg17 {num = 2 : i32}
        scf.yield %token1 : !ttg.async.token
      } else {
        scf.yield %arg15 : !ttg.async.token
      }
      %11 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %12 = ttg.async_commit_group tokens %11
      %13 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
      %14 = ttg.async_commit_group tokens %13
      scf.yield %arg16, %12, %arg18, %14 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 0
    %9 = ttg.async_wait %8#0, %8#1, %8#2, %8#3 {num = 0 : i32}
    tt.return
  }
}

// -----

// Double buffering with different number of async_copies inside scf.if then and else block. Check that we take the lower number from both blocks

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: double_buffering_uneven_then_else
  tt.func public @double_buffering_uneven_then_else(%cond: i1, %arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %7 = ttg.async_commit_group tokens %6
    %8:4 = scf.for %arg14 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg15 = %1, %arg16 = %5, %arg17 = %3, %arg18 = %7) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 3
      %token1 = ttg.async_wait %arg15, %arg17 {num = 2 : i32}

      %103 = scf.if %cond -> (!ttg.async.token) {
        %11 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
        %110 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
        %12 = ttg.async_commit_group tokens %11, %110
        scf.yield %12 : !ttg.async.token
      } else {
        %11 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
        %12 = ttg.async_commit_group tokens %11
        scf.yield %12 : !ttg.async.token
      }
      %13 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
      %14 = ttg.async_commit_group tokens %13
      scf.yield %arg16, %103, %arg18, %14 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 0
    %9 = ttg.async_wait %8#0, %8#1, %8#2, %8#3 {num = 0 : i32}
    tt.return
  }
}

// -----

// Test for dynamic loop in def chain

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: dynamic_loop_in_def_chain
  tt.func public @dynamic_loop_in_def_chain(%arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 1 direct to lds instruction
    %6 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %7 = ttg.async_commit_group tokens %6
    // Dynamic iteration count
    %30 = scf.for %arg21 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg30 = %6) -> (!ttg.async.token) : i32 {
      // CHECK: amdg.async_wait {{.*}} {num_inst = 0
      %31 = ttg.async_wait %arg30 {num = 1 : i32}
      // Emits 1 direct to lds instruction
      %32 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %33 = ttg.async_commit_group tokens %32
      scf.yield %33 : !ttg.async.token
    }
    // CHECK: amdg.async_wait {{.*}} {num_inst = 1
    %10 = ttg.async_wait %1 {num = 1 : i32}
    tt.return
  }
}

// -----

// Test loop in def chain with constant iteration count

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: constant_loop_in_def_chain
  tt.func public @constant_loop_in_def_chain(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 1 direct to lds instruction
    %6 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %7 = ttg.async_commit_group tokens %6
    %30 = scf.for %arg21 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg30 = %6) -> (!ttg.async.token) : i32 {
      // CHECK: amdg.async_wait {{.*}} {num_inst = 0
      %31 = ttg.async_wait %arg30 {num = 1 : i32}
      // Emits 1 direct to lds instruction
      %32 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %33 = ttg.async_commit_group tokens %32
      scf.yield %33 : !ttg.async.token
    }
    // CHECK: amdg.async_wait {{.*}} {num_inst = 5
    %10 = ttg.async_wait %1 {num = 1 : i32}
    tt.return
  }
}

// -----

// Simple case with TDM

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: simple_tdm_waitcnt
  tt.func public @simple_tdm_waitcnt(%memDesc: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %tensorDesc: !tt.tensordesc<128x16xf16>, %mask: i32
  ) {
    %c0_i32 = arith.constant 0 : i32

    // Each async_tdm_copy only emits a single instruction (-> counts 1)
    %1 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc : !tt.tensordesc<128x16xf16> -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable>
    %2 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc : !tt.tensordesc<128x16xf16> -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable>

    // Do not wait on the second tdm => waitcnt 1
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 1
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %w1 = amdg.async_tdm_wait %1 {num = 0 : i32}
    // No async_copies in between => waitcnt 0
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 0
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %w2 = amdg.async_tdm_wait %2 {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: contiguity_hint_waitcnt
  tt.func public @contiguity_hint_waitcnt(%arg1: !ttg.memdesc<64x32xf16, #shared, #smem, mutable>, %arg2: tensor<64x32x!tt.ptr<f16>, #blocked>) {
    // Dummy commit group to wait on both loads
    %0 = ttg.async_commit_group

    // 16 loads since vec=1 and no hints
    %1 = ttg.async_copy_global_to_local %arg2, %arg1 : tensor<64x32x!tt.ptr<f16>, #blocked> -> <64x32xf16, #shared, #smem, mutable>
    %2 = ttg.async_commit_group tokens %1

    // Contig hint bumps contig to 8 -> 2 loads
    %3 = ttg.async_copy_global_to_local %arg2, %arg1 {contiguity = 8 : i32} : tensor<64x32x!tt.ptr<f16>, #blocked> -> <64x32xf16, #shared, #smem, mutable>
    %4 = ttg.async_commit_group tokens %3

    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %5 = ttg.async_wait %2 {num = 0 : i32}
    // CHECK: amdg.async_wait {{.*}} {num_inst = 18
    %6 = ttg.async_wait %0 {num = 0 : i32}

    tt.return
  }
}

// -----

// Test async_copy_local_to_global on CDNA5

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: simple_local_to_global_waitcnt
  // SCOPE-LABEL: @simple_local_to_global_waitcnt(
  tt.func public @simple_local_to_global_waitcnt(%arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, %arg2: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) -> tensor<32x32xf32, #blocked> {
    // Emits 2 async store instructions (256 bits per thread, split into 2x128-bit stores)
    %0 = amdg.async_copy_local_to_global %arg1, %arg2 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 async store instructions
    %2 = amdg.async_copy_local_to_global %arg1, %arg2 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %3 = ttg.async_commit_group tokens %2

    // Wait on token %1: 2 store instructions outstanding
    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    // CHECK-SAME: ttg.amdg.completion_scope = "all"
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // Wait on token %3: 0 outstanding
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    // CHECK-SAME: ttg.amdg.completion_scope = "all"
    %10 = ttg.async_wait %3 {num = 0 : i32}
    // SCOPE: amdg.async_wait
    // SCOPE: amdg.async_wait
    // SCOPE-SAME: ttg.amdg.completion_scope = "all"
    // SCOPE-NEXT: ttg.barrier local
    // SCOPE-NEXT: {{.*}} = tt.load
    %observed = tt.load %arg2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return %observed : tensor<32x32xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: bf16_local_to_global_split_waitcnt
  tt.func public @bf16_local_to_global_split_waitcnt(%arg1: !ttg.memdesc<4x32xbf16, #shared, #smem, mutable>, %arg2: tensor<4x32x!tt.ptr<bf16>, #blocked>) {
    // Emits 2 async store intrinsics (16-bit store split into two 8-bit stores)
    %0 = amdg.async_copy_local_to_global %arg1, %arg2 : !ttg.memdesc<4x32xbf16, #shared, #smem, mutable> -> tensor<4x32x!tt.ptr<bf16>, #blocked>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 async store intrinsics
    %2 = amdg.async_copy_local_to_global %arg1, %arg2 : !ttg.memdesc<4x32xbf16, #shared, #smem, mutable> -> tensor<4x32x!tt.ptr<bf16>, #blocked>
    %3 = ttg.async_commit_group tokens %2

    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %10 = ttg.async_wait %3 {num = 0 : i32}
    tt.return
  }
}

// -----

// Test mixing async_copy_global_to_local and async_copy_local_to_global on CDNA5

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: mix_global_to_local_and_local_to_global
  tt.func public @mix_global_to_local_and_local_to_global(%arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, %arg2: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    // Emits 2 async load instructions
    %0 = ttg.async_copy_global_to_local %arg2, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 async store instructions
    %2 = amdg.async_copy_local_to_global %arg1, %arg2 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %3 = ttg.async_commit_group tokens %2

    // Wait on token %1: 2 store instructions outstanding
    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // Wait on token %3: 0 outstanding
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %10 = ttg.async_wait %3 {num = 0 : i32}
    tt.return
  }
}

// -----

// Test mixing async_copy and async_tdm_copy

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: mix_async_copy_and_async_tdm_copy
  tt.func public @mix_async_copy_and_async_tdm_copy(%memDesc: !ttg.memdesc<128x8xf16, #shared, #smem, mutable>, %tensorDesc: !tt.tensordesc<128x8xf16>, %mask: i32, %ptr: tensor<128x8x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}
  ) {
    %c0_i32 = arith.constant 0 : i32

    // Each async_tdm_copy only emits a single instruction (-> counts 1)
    %1 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc : !tt.tensordesc<128x8xf16> -> !ttg.memdesc<128x8xf16, #shared, #smem, mutable>

    %2 = ttg.async_copy_global_to_local %ptr, %memDesc : tensor<128x8x!tt.ptr<f16>, #blocked> -> <128x8xf16, #shared, #smem, mutable>
    %21 = ttg.async_commit_group tokens %2

    %3 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc : !tt.tensordesc<128x8xf16> -> !ttg.memdesc<128x8xf16, #shared, #smem, mutable>

    %4 = ttg.async_copy_global_to_local %ptr, %memDesc : tensor<128x8x!tt.ptr<f16>, #blocked> -> <128x8xf16, #shared, #smem, mutable>
    %5 = ttg.async_copy_global_to_local %ptr, %memDesc : tensor<128x8x!tt.ptr<f16>, #blocked> -> <128x8xf16, #shared, #smem, mutable>
    %51 = ttg.async_commit_group tokens %4, %5

    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 1
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %tw1 = amdg.async_tdm_wait %1 {num = 0 : i32}

    // Wait on token %21: 2 async_copy instructions outstanding
    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %cw1 = ttg.async_wait %21 {num = 0 : i32}

    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 0
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %w2 = amdg.async_tdm_wait %3 {num = 0 : i32}

    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %cw2 = ttg.async_wait %51 {num = 0 : i32}
    tt.return
  }
}

// -----

// Test scf.if without else region in def chain

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: scf_if_without_else
  tt.func public @scf_if_without_else(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %cond: i1) {
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0

    // For scf.if without else region, the else path contributes 0 instructions;
    // so the minimum across both paths is 0.
    scf.if %cond {
      // Emits 1 direct to lds instruction inside the if
      %inner = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %inner_commit = ttg.async_commit_group tokens %inner
    }

    // Wait on token %1: if path contributes 1, else (skip) contributes 0 → min = 0
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %10 = ttg.async_wait %1 {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#idx_i32_parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#idx_i16_parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_gather_scatter_multiple_instructions
  tt.func public @tdm_gather_scatter_multiple_instructions(
    %memDesc: !ttg.memdesc<256x128xf16, #shared, #smem, mutable>,
    %tensorDesc: !tt.tensordesc<64x128xf16>,
    %row_indices_i32: tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_i32_parent}>>,
    %row_indices_i16: tensor<256xi16, #ttg.slice<{dim = 0, parent = #idx_i16_parent}>>,
    %pred: i32
  ) {
    %c0_i32 = arith.constant 0 : i32

    %memDesc_i32 = ttg.memdesc_subslice %memDesc [0, 0] : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable, 256x128>

    // Gather with i32 indices: sizePerThread=16, 4 warps, maxPerInstr=8 => 2 instructions
    %token1 = amdg.async_tdm_gather %tensorDesc[%row_indices_i32] to %memDesc_i32 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_i32_parent}>>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable, 256x128> -> !tt.tensordesc<64x128xf16>
    // Scatter with i32 indices: 2 instructions
    %token2 = amdg.async_tdm_scatter %tensorDesc[%row_indices_i32] from %memDesc_i32 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_i32_parent}>>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable, 256x128> -> !tt.tensordesc<64x128xf16>
    // Gather with i16 indices: sizePerThread=64, 4 warps, maxPerInstr=16 => 4 instructions
    %token3 = amdg.async_tdm_gather %tensorDesc[%row_indices_i16] to %memDesc : tensor<256xi16, #ttg.slice<{dim = 0, parent = #idx_i16_parent}>>, !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // Scatter with i16 indices: 4 instructions
    %token4 = amdg.async_tdm_scatter %tensorDesc[%row_indices_i16] from %memDesc : tensor<256xi16, #ttg.slice<{dim = 0, parent = #idx_i16_parent}>>, !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>

    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 0
    // CHECK-SAME: ttg.amdg.completion_scope = "all"
    %w1 = amdg.async_tdm_wait %token4 {num = 0 : i32}
    // The full wait above retired all stores, including these older tokens.
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 4
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %w2 = amdg.async_tdm_wait %token3 {num = 0 : i32}
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 8
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %w3 = amdg.async_tdm_wait %token2 {num = 0 : i32}
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 10
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %w4 = amdg.async_tdm_wait %token1 {num = 0 : i32}

    tt.return
  }
}

// -----

// Test bug fix: With warp = [[0], [0]] the warp dimension has free variables,
// so the load should contribute 0 instructions — non-canonical warps skip the
// load entirely.

#linear_warp_free = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16], [32]], warp = [[0], [0]], block = []}>
#shared_simple = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: warp_free_variable_returns_zero
  tt.func public @warp_free_variable_returns_zero(
      %ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %offsets: tensor<64xi32, #linear_warp_free>,
      %dest: !ttg.memdesc<64xi32, #shared_simple, #smem, mutable>) {
    %0 = amdg.buffer_load_to_local %ptr[%offsets] into %dest : !tt.ptr<i32>[tensor<64xi32, #linear_warp_free>]  -> <64xi32, #shared_simple, #smem, mutable>
    %1 = ttg.async_commit_group

    // Warp free variable means 0 instructions emitted for this warp config
    // CHECK: amdg.async_wait {num_inst = 0
    %2 = ttg.async_wait {num = 1 : i32}
    tt.return
  }
}

// -----

// Test bug fix: register zero bases should not inflate instruction count.

#linear_reg_zero = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [128]], block = []}>
#shared_simple2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: register_zero_bases_not_inflated
  tt.func public @register_zero_bases_not_inflated(
      %ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %offsets: tensor<256xi32, #linear_reg_zero>,
      %dest: !ttg.memdesc<256xi32, #shared_simple2, #smem, mutable>) {
    %0 = amdg.buffer_load_to_local %ptr[%offsets] into %dest : !tt.ptr<i32>[tensor<256xi32, #linear_reg_zero>]  -> <256xi32, #shared_simple2, #smem, mutable>
    %1 = ttg.async_commit_group
    // Register zero bases should not inflate count: 1 instruction
    // CHECK: amdg.async_wait {num_inst = 1
    %2 = ttg.async_wait {num = 1 : i32}
    tt.return
  }
}

// -----

// Test TDM with partitioned shared encoding where each copy emits multiple instructions

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 4, partitionDim = 0, partitionLayout = #shared_inner}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_partitioned_shared_waitcnt
  tt.func public @tdm_partitioned_shared_waitcnt(
    %memDesc: !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>,
    %tensorDesc: !tt.tensordesc<128x16xf16>,
    %mask: i32
  ) {
    %c0_i32 = arith.constant 0 : i32

    // numLogicalPieces = numPartitions * numGroups = 2 * 4 = 8
    // warpsAlongPartition = gcd(numWarps=4, numLogicalPieces=8) = 4
    // Each async_tdm_copy emits divideCeil(8, 4) = 2 instructions
    %1 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc : !tt.tensordesc<128x16xf16> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    %2 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc : !tt.tensordesc<128x16xf16> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>

    // Skip second copy (2 instructions) => count = 2
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 2
    %w1 = amdg.async_tdm_wait %1 {num = 0 : i32}
    // Nothing in between => count = 0
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 0
    %w2 = amdg.async_tdm_wait %2 {num = 0 : i32}
    tt.return
  }
}

// -----

#copy = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#idx = #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @completion_scope_gather_only(
  // SCOPE-LABEL: @completion_scope_gather_only(
  tt.func @completion_scope_gather_only(%mem: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, %desc: !tt.tensordesc<64x128xf16>, %indices: tensor<64xi32, #idx>, %ptr: tensor<64x128x!tt.ptr<f16>, #copy>) -> tensor<64x128xf16, #copy> {
    %g = amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // CHECK: amdg.async_tdm_intrinsic_wait
    // CHECK-SAME: ttg.amdg.completion_scope = "shared"
    %w = amdg.async_tdm_wait %g {num = 0 : i32}
    // SCOPE: amdg.async_tdm_intrinsic_wait
    // SCOPE-SAME: ttg.amdg.completion_scope = "shared"
    // SCOPE-NEXT: {{.*}} = tt.load
    %observed = tt.load %ptr : tensor<64x128x!tt.ptr<f16>, #copy>
    tt.return %observed : tensor<64x128xf16, #copy>
  }

  // CHECK-LABEL: @completion_scope_load_loop_store(
  tt.func @completion_scope_load_loop_store(
      %mem: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x128xf16>,
      %indices: tensor<64xi32, #idx>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // Partial waits in a load-only phase do not require global ordering.
    // CHECK: amdg.async_tdm_intrinsic_wait {count = 1 : i32, ttg.amdg.completion_scope = "shared"}
    amdg.async_tdm_intrinsic_wait {count = 1 : i32}
    scf.for %i = %c0 to %n step %c1 : i32 {
      amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
      // CHECK: amdg.async_tdm_intrinsic_wait {count = 1 : i32, ttg.amdg.completion_scope = "shared"}
      amdg.async_tdm_intrinsic_wait {count = 1 : i32}
    }
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}}ttg.amdg.completion_scope = "shared"
    amdg.async_tdm_wait {num = 0 : i32}
    amdg.async_tdm_scatter %desc[%indices] from %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // Stores require All scope until a full wait completes them.
    // CHECK: amdg.async_tdm_intrinsic_wait {count = 0 : i32, ttg.amdg.completion_scope = "all"
    amdg.async_tdm_wait {num = 0 : i32}
    // The retired store no longer affects scope, even without a new load.
    // CHECK: amdg.async_tdm_intrinsic_wait {count = 1 : i32, ttg.amdg.completion_scope = "shared"}
    amdg.async_tdm_intrinsic_wait {count = 1 : i32}
    tt.return
  }

  // CHECK-LABEL: @completion_scope_partial_mixed(
  tt.func @completion_scope_partial_mixed(
      %mem: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x128xf16>,
      %indices: tensor<64xi32, #idx>) {
    amdg.async_tdm_scatter %desc[%indices] from %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // A partial wait may leave a store pending, even after later loads.
    // CHECK: amdg.async_tdm_intrinsic_wait {count = 1 : i32, ttg.amdg.completion_scope = "all"}
    amdg.async_tdm_intrinsic_wait {count = 1 : i32}
    %load = amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // Waiting on the load also completes earlier stores in the same queue.
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}}ttg.amdg.completion_scope = "all"
    amdg.async_tdm_wait %load {num = 0 : i32}
    amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // CHECK: amdg.async_tdm_intrinsic_wait {count = 0 : i32, ttg.amdg.completion_scope = "shared"}
    amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @completion_scope_independent_drains(
  tt.func @completion_scope_independent_drains(
      %mem: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x128xf16>,
      %indices: tensor<64xi32, #idx>,
      %ptr: tensor<64x128x!tt.ptr<f16>, #copy>) {
    %store = amdg.async_copy_local_to_global %mem, %ptr : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128x!tt.ptr<f16>, #copy>
    %storeGroup = ttg.async_commit_group tokens %store
    amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // A TDM wait neither inherits nor clears ordinary pending stores.
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}}ttg.amdg.completion_scope = "shared"
    amdg.async_tdm_wait {num = 0 : i32}
    // CHECK: amdg.async_wait {{.*}}ttg.amdg.completion_scope = "all"
    ttg.async_wait %storeGroup {num = 0 : i32}

    amdg.async_tdm_scatter %desc[%indices] from %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    %copy = ttg.async_copy_global_to_local %ptr, %mem : tensor<64x128x!tt.ptr<f16>, #copy> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %copy
    // The same independence holds with the queues reversed, across a barrier.
    ttg.barrier local
    // CHECK: amdg.async_wait {{.*}}ttg.amdg.completion_scope = "shared"
    ttg.async_wait %group {num = 0 : i32}
    // Draining ordinary copies must not discard the TDM store.
    amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}}ttg.amdg.completion_scope = "all"
    amdg.async_tdm_wait {num = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @completion_scope_branch_join(
  tt.func @completion_scope_branch_join(
      %mem: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x128xf16>,
      %indices: tensor<64xi32, #idx>,
      %cond: i1) {
    scf.if %cond {
      amdg.async_tdm_scatter %desc[%indices] from %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    } else {
      amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    }
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}}ttg.amdg.completion_scope = "all"
    amdg.async_tdm_wait {num = 0 : i32}
    amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}}ttg.amdg.completion_scope = "shared"
    amdg.async_tdm_wait {num = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @completion_scope_for_backedge(
  tt.func @completion_scope_for_backedge(
      %mem: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x128xf16>,
      %indices: tensor<64xi32, #idx>,
      %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
      // The previous iteration's scatter reaches this wait.
      // CHECK: amdg.async_tdm_intrinsic_wait {{.*}}ttg.amdg.completion_scope = "all"
      amdg.async_tdm_wait {num = 0 : i32}
      amdg.async_tdm_scatter %desc[%indices] from %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    }
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}}ttg.amdg.completion_scope = "all"
    amdg.async_tdm_wait {num = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @completion_scope_unsupported_region(
  tt.func @completion_scope_unsupported_region(
      %mem: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x128xf16>,
      %indices: tensor<64xi32, #idx>) {
    amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // Warp-specialized regions may run concurrently; retain the fallback.
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      // CHECK: amdg.async_tdm_intrinsic_wait {count = 0 : i32, ttg.amdg.completion_scope = "all"}
      amdg.async_tdm_intrinsic_wait {count = 0 : i32}
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      // CHECK: amdg.async_tdm_intrinsic_wait {count = 0 : i32, ttg.amdg.completion_scope = "all"}
      amdg.async_tdm_intrinsic_wait {count = 0 : i32}
      ttg.warp_return
    } : () -> ()
    amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // CHECK: amdg.async_tdm_intrinsic_wait {{.*}}ttg.amdg.completion_scope = "all"
    amdg.async_tdm_wait {num = 0 : i32}
    tt.return
  }
}

// -----

#copy = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#idx = #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @completion_scope_callee(
  tt.func private @completion_scope_callee(%mem: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, %desc: !tt.tensordesc<64x128xf16>, %indices: tensor<64xi32, #idx>, %ptr: tensor<64x128x!tt.ptr<f16>, #copy>) {
    %g = amdg.async_tdm_gather %desc[%indices] to %mem : tensor<64xi32, #idx>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // CHECK: amdg.async_tdm_intrinsic_wait
    // CHECK-SAME: ttg.amdg.completion_scope = "all"
    %tw = amdg.async_tdm_wait %g {num = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @completion_scope_caller(
  tt.func @completion_scope_caller(%mem: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, %desc: !tt.tensordesc<64x128xf16>, %indices: tensor<64xi32, #idx>, %ptr: tensor<64x128x!tt.ptr<f16>, #copy>) {
    tt.call @completion_scope_callee(%mem, %desc, %indices, %ptr) : (!ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !tt.tensordesc<64x128xf16>, tensor<64xi32, #idx>, tensor<64x128x!tt.ptr<f16>, #copy>) -> ()
    %c = ttg.async_copy_global_to_local %ptr, %mem : tensor<64x128x!tt.ptr<f16>, #copy> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %c
    // CHECK: amdg.async_wait
    // CHECK-SAME: ttg.amdg.completion_scope = "all"
    %cw = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
}
