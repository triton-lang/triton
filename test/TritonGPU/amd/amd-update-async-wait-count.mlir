// RUN: triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=arch-generation-name=gfx950 | FileCheck %s

// Simple case without any branching

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: simple_waitcnt
  tt.func public @simple_waitcnt(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1>) {
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2

    // Do not wait on the second async_copy => waitcnt 2
    // CHECK: ttg.async_wait {{.*}} {num = 2
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // No async_copies in between => waitcnt 0
    // CHECK: ttg.async_wait {{.*}} {num = 0
    %10 = ttg.async_wait %3 {num = 0 : i32}
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
  tt.func public @simple_waitcnt_reversed(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1>) {
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2

    // Do not wait on the second async_copy => waitcnt 2
    // CHECK: ttg.async_wait {{.*}} {num = 0
    %9 = ttg.async_wait %3 {num = 0 : i32}
    // No async_copies in between => waitcnt 0
    // CHECK: ttg.async_wait {{.*}} {num = 2
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
  tt.func public @simple_waitcnt_with_tt_load(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1>) {
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2

    %4 = tt.load %arg3 : tensor<128x16x!tt.ptr<f16>, #blocked>

    // CHECK: ttg.async_wait {{.*}} {num = 2
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // CHECK: ttg.async_wait {{.*}} {num = 0
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
  tt.func public @wait_in_for_loop(%arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %8:2 = scf.for %arg14 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg15 = %1, %arg16 = %3) -> (!ttg.async.token, !ttg.async.token)  : i32 {
      // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 0
      %10 = ttg.async_wait %arg15, %arg16 {num = 2 : i32}
      %11 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %12 = ttg.async_commit_group tokens %11
      %13 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
      %14 = ttg.async_commit_group tokens %13
      scf.yield %12, %14: !ttg.async.token, !ttg.async.token
    }
    // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 0
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
  tt.func public @double_buffering(%arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1>) {
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
      // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 3
      %10 = ttg.async_wait %arg15, %arg17 {num = 2 : i32}
      %11 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %12 = ttg.async_commit_group tokens %11
      %13 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
      %14 = ttg.async_commit_group tokens %13
      scf.yield %arg16, %12, %arg18, %14 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 0
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
  tt.func public @double_buffering_wait_in_if(%cond: i1, %arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1>) {
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
        // We wait on both tokens so we interleave with one iteration => 3
        // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 3
        %token1 = ttg.async_wait %arg15, %arg17 {num = 2 : i32}
        scf.yield %token1 : !ttg.async.token
      } else {
        // We only wait on the token of the first load so we can interleave one more load => 3 + 2
        // CHECK: ttg.async_wait {{.*}} {num = 5
        %token2 = ttg.async_wait %arg15 {num = 1 : i32}
        scf.yield %token2 : !ttg.async.token
      }
      %11 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %12 = ttg.async_commit_group tokens %11
      %13 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
      %14 = ttg.async_commit_group tokens %13
      scf.yield %arg16, %12, %arg18, %14 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 0
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
  tt.func public @doube_buffering_wait_loads_in_if(%cond: i1, %arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1>) {
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
        // We wait on both tokens (3) and additionally we should count the load inside our block (+2) => 5
        // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 5
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
    // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 0
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
  tt.func public @double_buffering_uneven_then_else(%cond: i1, %arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1>) {
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
      // The then block contains 3 instructions and the else 1 so we expect the count to be 3 (1 + 2) because there are also 2 instructions outside the scf.if in the loop body
      // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 3
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
    // CHECK: ttg.async_wait {{.*}}, {{.*}} {num = 0
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
  tt.func public @dynamic_loop_in_def_chain(%arg0: i32, %arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 1 direct to lds instruction
    %6 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %7 = ttg.async_commit_group tokens %6
    // Dynamic iteration count so we should not count its body
    %30 = scf.for %arg21 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg30 = %6) -> (!ttg.async.token) : i32 {
      // CHECK: ttg.async_wait {{.*}} {num = 0
      %31 = ttg.async_wait %arg30 {num = 1 : i32}
      // Emits 1 direct to lds instruction
      %32 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %33 = ttg.async_commit_group tokens %32
      scf.yield %33 : !ttg.async.token
    }
    // CHECK: ttg.async_wait {{.*}} {num = 1
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
  tt.func public @constant_loop_in_def_chain(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 1 direct to lds instruction
    %6 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %7 = ttg.async_commit_group tokens %6
    // Loop with 4 iterations => 4 instructions
    %30 = scf.for %arg21 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg30 = %6) -> (!ttg.async.token) : i32 {
      // CHECK: ttg.async_wait {{.*}} {num = 0
      %31 = ttg.async_wait %arg30 {num = 1 : i32}
      // Emits 1 direct to lds instruction
      %32 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %33 = ttg.async_commit_group tokens %32
      scf.yield %33 : !ttg.async.token
    }
    // CHECK: ttg.async_wait {{.*}} {num = 5
    %10 = ttg.async_wait %1 {num = 1 : i32}
    tt.return
  }
}
