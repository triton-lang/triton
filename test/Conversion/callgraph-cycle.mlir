// RUN: not --crash triton-opt %s --tritongpu-allocate-warp-groups 2>&1 | FileCheck %s

// A cyclic call graph must trip the cycle guard instead of recursing until
// the stack gives out: doWalk marks the recursion path in `visited` (#11726).
// The guard lives in triton::CallGraph's walk, which
// --tritongpu-allocate-warp-groups exercises on every module.
// CHECK: LLVM ERROR: Cycle detected in call graph
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @main() {
    tt.call @a() : () -> ()
    tt.return
  }
  tt.func @a() {
    tt.call @b() : () -> ()
    tt.return
  }
  tt.func @b() {
    tt.call @a() : () -> ()
    tt.return
  }
}
