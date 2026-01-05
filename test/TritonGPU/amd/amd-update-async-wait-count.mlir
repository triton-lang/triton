// RUN: triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=arch-generation-name=gfx950 | FileCheck %s

// Simple case without any branching

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.padded_shared<[4:+4] {order = [1, 0], shape=[16, 256]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: simple_waitcnt
  tt.func public @simple_waitcnt(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    // Emits 1 direct to lds instruction
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2

    // Do not wait on the second async_copy => waitcnt 2
    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // No async_copies in between => waitcnt 0
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %10 = ttg.async_wait %3 {num = 0 : i32}
    tt.return
  }
}

// -----

// Simple case with amdg.buffer_load_to_local

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.padded_shared<[4:+4] {order = [1, 0], shape = [16, 256]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: simple_buffer_load_to_local_waitcnt
  tt.func public @simple_buffer_load_to_local_waitcnt(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: tensor<128x16xi32, #blocked> {tt.contiguity = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: tensor<16x256xi32, #blocked1> {tt.contiguity = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg5: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>) {
    // Emits 1 direct to lds instruction
    %0 = amdg.buffer_load_to_local %arg0[%arg1] into %arg4 : <f16>[tensor<128x16xi32, #blocked>]  -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 direct to lds instructions
    %2 = amdg.buffer_load_to_local %arg2[%arg3] into %arg5 : <f16>[tensor<16x256xi32, #blocked1>]  -> <16x256xf16, #shared1, #smem, mutable>
    // Do not wait on the second buffer_load_to_local => waitcnt 2
    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_wait %1 {num = 0 : i32}
    // No buffer_load_to_local in between => waitcnt 0
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
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

    // Do not wait on the second async_copy => waitcnt 2
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %9 = ttg.async_wait %3 {num = 0 : i32}
    // No async_copies in between => waitcnt 0
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

    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %9 = ttg.async_wait %1 {num = 0 : i32}
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
        // We wait on both tokens so we interleave with one iteration => 3
        // CHECK: amdg.async_wait {{.*}}, {{.*}} {num_inst = 3
        %token1 = ttg.async_wait %arg15, %arg17 {num = 2 : i32}
        scf.yield %token1 : !ttg.async.token
      } else {
        // We only wait on the token of the first load so we can interleave one more load => 3 + 2
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
        // We wait on both tokens (3) and additionally we should count the load inside our block (+2) => 5
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
      // The then block contains 3 instructions and the else 1 so we expect the count to be 3 (1 + 2) because there are also 2 instructions outside the scf.if in the loop body
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
    // Dynamic iteration count so we should not count its body
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
    // Loop with 4 iterations => 4 instructions
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

// Test async_copy_local_to_global on GFX1250

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: simple_local_to_global_waitcnt
  tt.func public @simple_local_to_global_waitcnt(%arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, %arg2: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    // Emits 2 async store instructions (256 bits per thread, split into 2x128-bit stores)
    %0 = amdg.async_copy_local_to_global %arg1, %arg2 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %1 = ttg.async_commit_group tokens %0
    // Emits 2 async store instructions
    %2 = amdg.async_copy_local_to_global %arg1, %arg2 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %3 = ttg.async_commit_group tokens %2

    // Do not wait on the second async_copy => waitcnt 2
    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // No async_copies in between => waitcnt 0
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %10 = ttg.async_wait %3 {num = 0 : i32}
    tt.return
  }
}

// -----

// Test mixing async_copy_global_to_local and async_copy_local_to_global on GFX1250

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

    // Do not wait on the store => waitcnt 2
    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // No async_copies in between => waitcnt 0
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %10 = ttg.async_wait %3 {num = 0 : i32}
    tt.return
  }
}

// -----

// Test mixing async_copy and async_tdm_copy

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mix_async_copy_and_async_tdm_copy
  tt.func public @mix_async_copy_and_async_tdm_copy(%memDesc: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %tensorDesc: !tt.tensordesc<tensor<128x16xf16>>, %mask: i1, %ptr: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}
  ) {
    %c0_i32 = arith.constant 0 : i32

    // Each async_tdm_copy only emits a single instruction (-> counts 1)
    %1 = amdg.async_tdm_copy_global_to_local %tensorDesc[%c0_i32, %c0_i32] into %memDesc, %mask : !tt.tensordesc<tensor<128x16xf16>> -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable>

    %2 = ttg.async_copy_global_to_local %ptr, %memDesc : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %21 = ttg.async_commit_group tokens %2

    %3 = amdg.async_tdm_copy_global_to_local %tensorDesc[%c0_i32, %c0_i32] into %memDesc, %mask : !tt.tensordesc<tensor<128x16xf16>> -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable>

    %4 = ttg.async_copy_global_to_local %ptr, %memDesc : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_copy_global_to_local %ptr, %memDesc : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %51 = ttg.async_commit_group tokens %4, %5

    // Check that we do not take other TDM loads into account (they use a different HW counter)

    // CHECK: amdg.async_wait {{.*}} {num_inst = 2
    %cw1 = ttg.async_wait %21 {num = 0 : i32}

    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
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

    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %10 = ttg.async_wait %1 {num = 0 : i32}
    tt.return
  }
}
