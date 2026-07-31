// RUN: triton-opt --split-input-file --test-print-scope-id-allocation -verify-diagnostics=only-expected -o /dev/null %s

module {
  // expected-remark @below {{one_scope}}
  tt.func @one_scope() {
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "name0"
    tt.return
  }

  // expected-remark @below {{two_scopes}}
  tt.func @two_scopes() {
    // expected-remark @below {{scope id = 1}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 1}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "name0"
    // expected-remark @below {{scope id = 2}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name1"
    // expected-remark @below {{scope id = 2}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "name1"
    tt.return
  }

  // expected-remark @below {{two_scopes_overlap}}
  tt.func @two_scopes_overlap() {
    // expected-remark @below {{scope id = 3}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 4}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name1"
    // expected-remark @below {{scope id = 3}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "name0"
    // expected-remark @below {{scope id = 4}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "name1"
    tt.return
  }

  // expected-remark @below {{nested_scopes}}
  tt.func @nested_scopes() {
    // expected-remark @below {{scope id = 5}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 6}}
    // expected-remark @below {{scope parent id = 5}}
    proton.record start "name1"
    // expected-remark @below {{scope id = 6}}
    // expected-remark @below {{scope parent id = 5}}
    proton.record end "name1"
    // expected-remark @below {{scope id = 5}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "name0"
    tt.return
  }
}

// -----

module {
  // expected-remark @below {{inner}}
  tt.func @inner() {
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "name0"
    tt.return
  }

  // expected-remark @below {{outer}}
  tt.func @outer() {
    // expected-remark @below {{scope id = 1}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name0"
    tt.call @inner() : () -> ()
    // expected-remark @below {{scope id = 1}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "name0"
    tt.return
  }
}

// -----

module {
  // expected-remark @below {{duplicate}}
  tt.func @duplicate() {
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "name0"
    // expected-remark @below {{scope id = 1}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name0"
    // expected-remark @below {{scope id = 1}}
    // expected-remark @below {{scope parent id = -1}}
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
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "name0"
    tt.return
  ^start:
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name0"
    cf.br ^exit
  }
}

// -----

module {
  // expected-remark @below {{scf_cond}}
  tt.func @scf_cond(%cond: i1) {
    scf.if %cond {
      // expected-remark @below {{scope id = 0}}
      // expected-remark @below {{scope parent id = -1}}
      proton.record start "if_only"
    }
    // expected-remark @below {{scope id = 0}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "if_only"
    tt.return
  }
}

// -----

module {
  tt.func @scf_loop() {
    %c0 = arith.constant 0 : index
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "loop"
    scf.for %i = %c0 to %c0 step %c0 {
      // expected-remark @below {{scope id = 1}}
      // expected-remark @below {{scope parent id = 0}}
      proton.record start "loop_body"
      proton.record end "loop_body"
    }
    proton.record end "loop"
    tt.return
  }
}

// -----

module {
  tt.func @scf_loop_if(%cond: i1) {
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c0 step %c0 {
      scf.if %cond {
        // expected-remark @below {{scope id = 0}}
        // expected-remark @below {{scope parent id = -1}}
        proton.record start "loop_if"
      }
      scf.if %cond {
        // expected-remark @below {{scope id = 0}}
        // expected-remark @below {{scope parent id = -1}}
        proton.record end "loop_if"
      }
    }
    tt.return
  }
}

// -----

module {
  // expected-remark @below {{cf_single_branch}}
  tt.func @cf_single_branch(%cond: i1) {
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "name0"
    cf.cond_br %cond, ^then, ^else
  ^then:  // pred: ^entry
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
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
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "outer"
    ttg.warp_specialize()
    default {
      // expected-remark @below {{scope id = 1}}
      // expected-remark @below {{scope parent id = 0}}
      proton.record start "default"
      // expected-remark @below {{scope id = 1}}
      // expected-remark @below {{scope parent id = 0}}
      proton.record end "default"
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      // expected-remark @below {{scope id = 2}}
      // expected-remark @below {{scope parent id = 0}}
      proton.record start "partition"
      // expected-remark @below {{scope id = 2}}
      // expected-remark @below {{scope parent id = 0}}
      proton.record end "partition"
      ttg.warp_return
    } : () -> ()
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "outer"
    tt.return
  }
}

// -----

module {
  // expected-remark @below {{cf_loop_closed}}
  tt.func @cf_loop_closed() {
  ^entry:
    %c0 = arith.constant 0 : index
    cf.br ^loop(%c0 : index)
  ^exit:
    tt.return
  ^loop(%iv: index):
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "loop_body"
    %c1 = arith.constant 1 : index
    %next = arith.addi %iv, %c1 : index
    %c2 = arith.constant 2 : index
    %cond = arith.cmpi ult, %next, %c2: index
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "loop_body"
    cf.cond_br %cond, ^loop(%next : index), ^exit
  }
}

// -----

module {
  // expected-remark @below {{cf_loop_closed_two_blocks}}
  tt.func @cf_loop_closed_two_blocks() {
  ^entry:
    %c0 = arith.constant 0 : index
    cf.br ^loop(%c0 : index)
  ^exit:
    tt.return
  ^loop(%iv: index):
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "loop_body"
    %c1 = arith.constant 1 : index
    %next = arith.addi %iv, %c1 : index
    cf.br ^loop_body(%next : index)
  ^loop_body(%iv_next: index):
    %c2 = arith.constant 2 : index
    %cond = arith.cmpi ult, %iv_next, %c2: index
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "loop_body"
    cf.cond_br %cond, ^loop(%iv_next : index), ^exit
  }
}

// -----

module {
  tt.func @cf_unclosed() {
    // expected-error @below {{The scope name 'unclosed' is not properly closed (missing end record)}}
    proton.record start "unclosed"
    tt.return
  }
}

// -----

module {
  tt.func @cf_dangling_end() {
    // expected-error @below {{The scope name 'dangling' is closed without being opened}}
    proton.record end "dangling"
    tt.return
  }
}

// -----

module {
  tt.func @cf_liveness_error(%cond: i1) {
    proton.record start "name0"
    cf.cond_br %cond, ^then, ^else
  ^then:  // pred: ^entry
    proton.record end "name0"
    cf.br ^merge
  ^else:  // pred: ^entry
    // expected-error @below {{The scope name 'name0' is not properly closed (missing start record)}}
    proton.record end "name0"
    cf.br ^merge
  ^merge:  // preds: ^then, ^else
    tt.return
  }
}

// -----

module {
  tt.func @cf_branch_unclosed_dangling(%cond: i1) {
    cf.cond_br %cond, ^then, ^else
  ^then:  // pred: ^entry
    proton.record start "ghost"
    cf.br ^merge
  ^else:  // pred: ^entry
    // expected-error @below {{The scope name 'ghost' is closed without being opened}}
    proton.record end "ghost"
    cf.br ^merge
  ^merge:  // preds: ^then, ^else
    tt.return
  }
}

// -----

module {
  tt.func @cf_merge_unclosed(%cond: i1) {
    cf.br ^start(%cond : i1)
  ^start(%cond_arg: i1):
    proton.record start "ghost"
    cf.cond_br %cond_arg, ^then, ^else
  ^then:  // pred: ^start
    proton.record end "ghost"
    cf.br ^merge
  ^else:  // pred: ^start
    proton.record start "ghost"
    cf.br ^merge
  ^merge:  // preds: ^then, ^else
    proton.record end "ghost"
    tt.return
  }
}

// -----

module {
  tt.func @cf_loop_unclosed() {
    %c0 = arith.constant 0 : index
    cf.br ^loop(%c0 : index)
  ^exit:
    tt.return
  ^loop(%iv: index):
    // expected-error @below {{The scope name 'loop' is started without being closed}}
    proton.record start "loop"
    %c1 = arith.constant 1 : index
    %next = arith.addi %iv, %c1 : index
    %c2 = arith.constant 2 : index
    %cond = arith.cmpi ult, %next, %c2: index
    cf.cond_br %cond, ^loop(%next : index), ^exit
  }
}

// -----

module {
  tt.func @cf_loop_end_before_start() {
    %c0 = arith.constant 0 : index
    cf.br ^loop(%c0 : index)
  ^exit:
    tt.return
  ^loop(%iv: index):
    // expected-error @below {{The scope name 'loop' has end record that dominates its start record}}
    proton.record end "loop"
    %c1 = arith.constant 1 : index
    %next = arith.addi %iv, %c1 : index
    %c2 = arith.constant 2 : index
    %cond = arith.cmpi ult, %next, %c2: index
    proton.record start "loop"
    cf.cond_br %cond, ^loop(%next : index), ^exit
  }
}

// -----

module {
  // expected-remark @below {{async_scope}}
  tt.func @async_scope() {
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record start "outer"
    // expected-remark @below {{scope id = 1}}
    // expected-remark @below {{scope parent id = -1}}
    %token = proton.allocate_async_token "async" : i32
    proton.async_record start %token : i32
    proton.async_record end %token : i32
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    proton.record end "outer"
    tt.return
  }
}

// -----

module {
  // Unlike synchronous scopes, async tokens with the same name identify
  // independent static sites and do not need structurally paired endpoints.
  // expected-remark @below {{async_scope_conditional}}
  tt.func @async_scope_conditional(%cond: i1) {
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    %token = proton.allocate_async_token "async" : i32
    proton.async_record start %token : i32
    scf.if %cond {
      proton.async_record end %token : i32
    } else {
      // expected-remark @below {{scope id = 1}}
      // expected-remark @below {{scope parent id = -1}}
      %other = proton.allocate_async_token "async" : i32
      proton.async_record start %other : i32
      proton.async_record end %other : i32
    }
    tt.return
  }
}

// -----

module {
  // Each branch ends the token for one pipeline slot and carries its
  // replacement to a later loop iteration.
  // expected-remark @below {{async_scope_loop_alternating_tokens}}
  tt.func @async_scope_loop_alternating_tokens() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c10 = arith.constant 10 : index
    // expected-remark @below {{scope id = 0}}
    // expected-remark @below {{scope parent id = -1}}
    %initial0 = proton.allocate_async_token "async_copy0" : i32
    proton.async_record start %initial0 : i32
    // expected-remark @below {{scope id = 1}}
    // expected-remark @below {{scope parent id = -1}}
    %initial1 = proton.allocate_async_token "async_op1" : i32
    proton.async_record start %initial1 : i32
    %final0, %final1 = scf.for %i = %c1 to %c10 step %c1
        iter_args(%token0 = %initial0, %token1 = %initial1) -> (i32, i32) {
      %remainder = arith.remui %i, %c2 : index
      %is_odd = arith.cmpi eq, %remainder, %c1 : index
      %next0, %next1 = scf.if %is_odd -> (i32, i32) {
        proton.async_record end %token0 : i32
        // expected-remark @below {{scope id = 2}}
        // expected-remark @below {{scope parent id = -1}}
        %new0 = proton.allocate_async_token "async_copy0" : i32
        proton.async_record start %new0 : i32
        scf.yield %new0, %token1 : i32, i32
      } else {
        proton.async_record end %token1 : i32
        // expected-remark @below {{scope id = 3}}
        // expected-remark @below {{scope parent id = -1}}
        %new1 = proton.allocate_async_token "async_op1" : i32
        proton.async_record start %new1 : i32
        scf.yield %token0, %new1 : i32, i32
      }
      scf.yield %next0, %next1 : i32, i32
    }
    proton.async_record end %final0 : i32
    proton.async_record end %final1 : i32
    tt.return
  }
}

