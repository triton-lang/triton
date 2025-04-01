// RUN: triton-opt --split-input-file --test-print-scope-id-allocation %s 2>&1 | FileCheck %s

module {
  // CHECK-LABEL: one_scope
  tt.func @one_scope() {
    // CHECK: scope id = 0
    // CHECK-NEXT: scope id = 0
    proton.record start "name0"
    proton.record end "name0"
    tt.return
  }

  // CHECK-LABEL: two_scopes
  tt.func @two_scopes() {
    // CHECK: scope id = 1
    // CHECK-NEXT: scope id = 1
    // CHECK: scope id = 2
    // CHECK-NEXT: scope id = 2
    proton.record start "name0"
    proton.record end "name0"
    proton.record start "name1"
    proton.record end "name1"
    tt.return
  }

  // CHECK-LABEL: two_scopes_overlap
  tt.func @two_scopes_overlap() {
    // CHECK: scope id = 3
    // CHECK-NEXT: scope id = 4
    // CHECK-NEXT: scope id = 3
    // CHECK-NEXT: scope id = 4
    proton.record start "name0"
    proton.record start "name1"
    proton.record end "name0"
    proton.record end "name1"
    tt.return
  }

  // CHECK-LABEL: control_flow
  tt.func @control_flow(%cond: i1) {
    // CHECK: scope id = 5
    // CHECK-NEXT: scope id = 6
    // CHECK-NEXT: scope id = 6
    // CHECK-NEXT: scope id = 5
    proton.record start "name0"
    scf.if %cond {
      proton.record start "name1"
      proton.record end "name1"
    }
    proton.record end "name0"
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: inner
  tt.func @inner() {
    // CHECK: scope id = 0
    // CHECK-NEXT: scope id = 0
    proton.record start "name0"
    proton.record end "name0"
    tt.return
  }

  // CHECK-LABEL: outer
  tt.func @outer() {
    // CHECK: scope id = 1
    proton.record start "name0"
    tt.call @inner() : () -> ()
    // CHECK-NEXT: scope id = 1
    proton.record end "name0"
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: duplicate
  tt.func @duplicate() {
    // CHECK: scope id = 0
    // CHECK-NEXT: scope id = 0
    // CHECK-NEXT: scope id = 1
    // CHECK-NEXT: scope id = 1
    proton.record start "name0"
    proton.record end "name0"
    proton.record start "name0"
    proton.record end "name0"
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: condition
  tt.func @condition(%cond: i1) {
    // CHECK: scope id = 0
    // CHECK-NEXT: scope id = 0
    proton.record start "name0"
    proton.record end "name0"
    scf.if %cond {
      // CHECK-NEXT: scope id = 1
      // CHECK-NEXT: scope id = 1
      proton.record start "name0"
      proton.record end "name0"
    }
    tt.return
  }
}
