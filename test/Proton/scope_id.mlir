// RUN: triton-opt --split-input-file --test-print-scope-id-allocation -verify-diagnostics=only-expected -o /dev/null %s

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
  // expected-remark @below {{cf_single_branch}}
  tt.func @cf_single_branch(%cond: i1) {
    // expected-remark @below {{scope id = 0}}
    proton.record start "name0"
    cf.cond_br %cond, ^then, ^else
  ^then:  // pred: ^entry
    // expected-remark @below {{scope id = 0}}
    proton.record end "name0"
    cf.br ^merge
  ^else:  // pred: ^entry
    cf.br ^merge
  ^merge:  // preds: ^then, ^else
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
  tt.func @cf_loop_closed() {
  ^entry:
    %c0 = arith.constant 0 : index
    cf.br ^loop(%c0 : index)
  ^exit:
    tt.return
  ^loop(%iv: index):
    proton.record start "loop_body"
    %c1 = arith.constant 1 : index
    %next = arith.addi %iv, %c1
    %c2 = arith.constant 2 : index
    %cond = arith.cmpi ult, %next, %c2
    proton.record end "loop_body"
    cf.cond_br %cond, ^loop(%next : index), ^exit
  }
}

// -----

module {
  tt.func @cf_loop_closed_two_blocks() {
  ^entry:
    %c0 = arith.constant 0 : index
    cf.br ^loop(%c0 : index)
  ^exit:
    tt.return
  ^loop(%iv: index):
    proton.record start "loop_body"
    %c1 = arith.constant 1 : index
    %next = arith.addi %iv, %c1
    cf.br ^loop_body(%next : index)
  ^loop_body(%iv_next: index):
    %c2 = arith.constant 2 : index
    %cond = arith.cmpi ult, %iv_next, %c2
    proton.record end "loop_body"
    cf.cond_br %cond, ^loop(%iv_next : index), ^exit
  }
}

// -----

module {
  // expected-remark @below {{cf_liveness_error}}
  tt.func @cf_liveness_error(%cond: i1) {
    // expected-remark @below {{scope id = 0}}
    proton.record start "name0"
    cf.cond_br %cond, ^then, ^else
  ^then:  // pred: ^entry
    // expected-remark @below {{scope id = 0}}
    proton.record end "name0"
    cf.br ^merge
  ^else:  // pred: ^entry
    // expected-remark @below {{scope id = 0}}
    proton.record end "name0"
    cf.br ^merge
  ^merge:  // preds: ^then, ^else
    tt.return
  }
}

// -----

module {
  // expected-error @below {{The scope name 'unclosed' is started without being closed}}
  tt.func @cf_unclosed() {
    proton.record start "unclosed"
  }
}

// -----

module {
  // expected-error @below {{The scope name 'dangling' is closed without being opened}}
  tt.func @cf_dangling_end() {
    proton.record end "dangling"
    tt.return
  }
}

// -----

module {
  // expected-error @below {{The scope name 'ghost' is started without being closed}}
  // expected-error @below {{The scope name 'ghost' is closed without being opened}}
  tt.func @cf_branch_unclosed_dangling(%cond: i1) {
    cf.cond_br %cond, ^then, ^else
  ^then:  // pred: ^entry
    proton.record start "ghost"
    cf.br ^merge
  ^else:  // pred: ^entry
    proton.record end "ghost"
    cf.br ^merge
  ^merge:  // preds: ^then, ^else
    tt.return
  }
}

// -----

module {
  // expected-error @below {{The scope name 'ghost' is started without being closed}}
  tt.func @cf_merge_unclosed(%cond: i1) {
    cf.cond_br %cond, ^then, ^else
    proton.record start "ghost"
  ^then:  // pred: ^entry
    proton.record stop "ghost"
    cf.br ^merge
  ^else:  // pred: ^entry
    proton.record start "ghost"
    cf.br ^merge
  ^merge:  // preds: ^then, ^else
    proton.record end "ghost"
    tt.return
  }
}

// -----

module {
  // expected-error @below {{The scope name 'loop' is started without being closed}}
  tt.func @cf_loop_unclosed() {
  ^entry:
    %c0 = arith.constant 0 : index
    cf.br ^loop(%c0 : index)
  ^exit:
    tt.return
  ^loop(%iv: index):
    proton.record start "loop"
    %c1 = arith.constant 1 : index
    %next = arith.addi %iv, %c1
    %c2 = arith.constant 2 : index
    %cond = arith.cmpi ult, %next, %c2
    cf.cond_br %cond, ^loop(%next : index), ^exit
  }
}

// -----

module {
  // expected-error @below {{The scope name 'loop' has end record that dominates its start record}}
  tt.func @cf_loop_end_before_start() {
  ^entry:
    %c0 = arith.constant 0 : index
    cf.br ^loop(%c0 : index)
  ^exit:
    tt.return
  ^loop(%iv: index):
    proton.record end "loop"
    %c1 = arith.constant 1 : index
    %next = arith.addi %iv, %c1
    %c2 = arith.constant 2 : index
    %cond = arith.cmpi ult, %next, %c2
    proton.record start "loop"
    cf.cond_br %cond, ^loop(%next : index), ^exit
  }
}
