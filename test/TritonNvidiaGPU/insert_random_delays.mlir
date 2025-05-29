// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-insert-random-delays | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @insert_delay_after() {
    ttg.async_wait {num = 4: i32}
    // CHECK: ttng.random_delay
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @insert_delay_before() {
    // CHECK: ttng.random_delay
    ttng.fence_async_shared {bCluster = false, ttg.partition = 0 : i32}
    tt.return
  }
}
