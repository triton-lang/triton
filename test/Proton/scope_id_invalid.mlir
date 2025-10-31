// RUN: triton-opt --split-input-file --test-print-scope-id-allocation -verify-diagnostics=error -o /dev/null %s

module {
  // expected-error@below {{Scope name 'name0' is not properly closed (missing start record)}}
  tt.func @cf_branch(%cond: i1) {
    proton.record start "name0"
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:  // pred: ^entry
    proton.record end "name0"
    cf.br ^bb3
  ^bb2:  // pred: ^entry
    proton.record end "name0"
    cf.br ^bb3
  ^bb3:  // preds: ^bb1, ^bb2
    tt.return
  }
}

// -----

module {
  tt.func @cf_mismatch(%cond: i1) {
  ^entry:
    proton.record start "name0"
    cf.cond_br %cond, ^then, ^else
  ^then:  // pred: ^entry
    proton.record end "name0"
    cf.br ^merge
  ^else:  // pred: ^entry
    cf.br ^merge
  ^merge:  // preds: ^then, ^else
    // expected-error @below {{inconsistent proton scope stack across predecessors, expected [] but found [name0]}}
    tt.return
  }
}
