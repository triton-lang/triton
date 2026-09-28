// RUN: not triton-opt %s --nvgpu-warp-specialization=num-stages=2 -o /dev/null 2>&1 | FileCheck %s

// CHECK: warp specialization cannot fall back from unsupported atomic RMW in warp-specialized function with preexisting async_task_id attributes

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @preexisting_unsupported_atomic(%ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c1 step %c1 : i32 {
      %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %c1 {async_task_id = array<i32: 1>} : (!tt.ptr<i32>, i32) -> i32
      scf.yield
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
