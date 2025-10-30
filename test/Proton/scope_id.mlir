// RUN: triton-opt --split-input-file --test-print-scope-id-allocation -verify-diagnostics -o /dev/null %s

module {
  // expected-remark @below {{one_scope}}
  tt.func @one_scope() {
    // expected-remark @below {{scope id = 0}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 0}}
    proton.record end "name0"
    tt.return
  }

  // expected-remark @below {{two_scopes}}
  tt.func @two_scopes() {
    // expected-remark @below {{scope id = 1}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 1}}
    proton.record end "name0"
    // expected-remark @below {{scope id = 2}}
    proton.record start "name1"
    // expected-remark @below {{scope id = 2}}
    proton.record end "name1"
    tt.return
  }

  // expected-remark @below {{two_scopes_overlap}}
  tt.func @two_scopes_overlap() {
    // expected-remark @below {{scope id = 3}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 4}}
    proton.record start "name1"
    // expected-remark @below {{scope id = 3}}
    proton.record end "name0"
    // expected-remark @below {{scope id = 4}}
    proton.record end "name1"
    tt.return
  }
}

// -----

module {
  // expected-remark @below {{inner}}
  tt.func @inner() {
    // expected-remark @below {{scope id = 0}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 0}}
    proton.record end "name0"
    tt.return
  }

  // expected-remark @below {{outer}}
  tt.func @outer() {
    // expected-remark @below {{scope id = 1}}
    proton.record start "name0"
    tt.call @inner() : () -> ()
    // expected-remark @below {{scope id = 1}}
    proton.record end "name0"
    tt.return
  }
}

// -----

module {
  // expected-remark @below {{duplicate}}
  tt.func @duplicate() {
    // expected-remark @below {{scope id = 0}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 0}}
    proton.record end "name0"
    // expected-remark @below {{scope id = 1}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 1}}
    proton.record end "name0"
    tt.return
  }
}

// -----

module {
  // expected-remark @below {{cf_branch}}
  tt.func @cf_branch(%cond: i1) {
  ^bb0(%arg0: i1):
    // expected-remark @below {{scope id = 0}}
    proton.record start "name0"
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    // expected-remark @below {{scope id = 0}}
    proton.record end "name0"
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    // expected-remark @below {{scope id = 0}}
    proton.record end "name0"
    cf.br ^bb3
  ^bb3:  // preds: ^bb1, ^bb2
    tt.return
  }
}

// -----

module {
  // expected-remark @below {{cf_reordered}}
  tt.func @cf_reordered() {
  ^entry:
    cf.br ^start
  ^exit:
    // expected-remark @below {{scope id = 0}}
    proton.record end "name0"
    tt.return
  ^start:
    // expected-remark @below {{scope id = 0}}
    proton.record start "name0"
    cf.br ^exit
  }
}

// -----

module {
  tt.func @cf_mismatch(%cond: i1) {
  ^bb0(%arg0: i1):
    proton.record start "name0"
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    proton.record end "name0"
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    cf.br ^bb3
  ^bb3:  // preds: ^bb1, ^bb2
    // expected-error @below {{inconsistent proton scope stack across predecessors, expected [name0] but found []}}
    tt.return
  }
}

// -----

module {
  // expected-remark @below {{warp_specialize_balanced}}
  tt.func @warp_specialize_balanced() {
    // expected-remark @below {{scope id = 0}}
    proton.record start "outer"
    ttg.warp_specialize()
    default {
      // expected-remark @below {{scope id = 1}}
      proton.record start "default"
      // expected-remark @below {{scope id = 1}}
      proton.record end "default"
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      // expected-remark @below {{scope id = 2}}
      proton.record start "partition"
      // expected-remark @below {{scope id = 2}}
      proton.record end "partition"
      ttg.warp_return
    } : () -> ()
    // expected-remark @below {{scope id = 0}}
    proton.record end "outer"
    tt.return
  }
}

// -----

module {
  tt.func @warp_specialize_mismatch() {
    proton.record start "outer"
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      proton.record end "outer"
      ttg.warp_return
    } : () -> ()
    // expected-error @below {{inconsistent proton scope stack across predecessors, expected [outer] but found []}}
    tt.return
  }
}
