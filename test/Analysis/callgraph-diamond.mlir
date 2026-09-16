// RUN: triton-opt %s --tritongpu-allocate-warp-groups 2>&1 | FileCheck %s

// A diamond is not a cycle: the shared callee is walked again after the first
// path through it is popped, without tripping the guard. Companion to
// callgraph-cycle.mlir (#11726).
// CHECK-LABEL: tt.func @main
// CHECK: tt.call @a
// CHECK: tt.call @b
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @main() {
    tt.call @a() : () -> ()
    tt.call @b() : () -> ()
    tt.return
  }
  tt.func @a() {
    tt.call @c() : () -> ()
    tt.return
  }
  tt.func @b() {
    tt.call @c() : () -> ()
    tt.return
  }
  tt.func @c() {
    tt.return
  }
}
