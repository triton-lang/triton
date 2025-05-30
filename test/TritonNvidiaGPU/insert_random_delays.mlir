// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-insert-random-delays | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @insert_delay_after() {
    // CHECK: ttg.async_wait
    // CHECK: ttng.random_delay
    ttg.async_wait {num = 4: i32}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @insert_delay_before() {
    // CHECK: ttng.random_delay
    // CHECK: ttng.fence_async_shared
    ttng.fence_async_shared {bCluster = false, ttg.partition = 0 : i32}
    tt.return
  }
}
