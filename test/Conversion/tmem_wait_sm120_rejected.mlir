// RUN: triton-opt %s -split-input-file \
// RUN:   --convert-triton-gpu-to-llvm='compute-capability=90' -verify-diagnostics -o /dev/null
// RUN: triton-opt %s -split-input-file \
// RUN:   --convert-triton-gpu-to-llvm='compute-capability=120' -verify-diagnostics -o /dev/null

// ttng.tmem_wait must fail with a clear error on targets without tensor
// memory instead of reaching NVPTX instruction selection (see #11747).
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_wait_rejected() {
    // expected-error @below {{'ttng.tmem_wait' op requires a target with tcgen05 (tensor memory) support}}
    // expected-error @below {{failed to legalize operation 'ttng.tmem_wait'}}
    ttng.tmem_wait store
    tt.return
  }
}
