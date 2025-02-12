// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_atomic_cas_i64
  tt.func public @buffer_atomic_cas_i64(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %val = arith.constant dense<2> : tensor<512xi64, #blocked>
    %cmp = arith.constant dense<0> : tensor<512xi64, #blocked>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<i64>, i32

    // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK: llvm.fence syncscope("agent") release
    // CHECK: %[[dst:.*]] = rocdl.raw.ptr.buffer.atomic.cmpswap %{{.*}}, %{{.*}}, %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i64
    // CHECK: %[[dst:.*]] = rocdl.raw.ptr.buffer.atomic.cmpswap %{{.*}}, %{{.*}}, %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i64
    // // CHECK: llvm.fence syncscope("agent") acquire

    %4 = amdgpu.buffer_atomic_cas acq_rel, gpu, %val, %cmp, %3[%2] : tensor<512xi64, #blocked>
    %5 = tt.addptr %arg1, %1 : !tt.ptr<i64>, i32
    amdgpu.buffer_store %4, %5[%2] : tensor<512xi64, #blocked>
    tt.return
  }
}
