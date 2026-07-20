// RUN: triton-opt %s -split-input-file --verify-diagnostics

module {
  llvm.func @masked_load_false_value_type_mismatch(%mask: i1, %src: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    // expected-error @+1 {{all of {falseVal, result} have same type}}
    %a = amdg.masked_load %src, %mask, %zero : (!llvm.ptr, i1, i32) -> f32
    llvm.return
  }
}

// -----

module {
  llvm.func @false_value_type_mismatch(%mask: i1) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    // expected-error @+1 {{false value types must match result types}}
    %a = amdg.masked_region %mask else (%zero) {
      %value = llvm.mlir.constant(0.000000e+00 : f32) : f32
      amdg.masked_yield %value : f32
    } : i32 -> f32
    llvm.return
  }
}

// -----

module {
  llvm.func @yield_type_mismatch(%mask: i1) {
    %zero = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %one = llvm.mlir.constant(1 : i32) : i32
    %a = amdg.masked_region %mask else (%zero) {
      // expected-error @+1 {{operand types must match parent result types}}
      amdg.masked_yield %one : i32
    } : f32 -> f32
    llvm.return
  }
}

// -----

module {
  llvm.func @region_body_argument(%mask: i1) {
    // expected-error @+1 {{body block must not have arguments}}
    amdg.masked_region %mask {
    ^bb0(%arg0: i32):
      amdg.masked_yield
    }
    llvm.return
  }
}

// -----

module {
  llvm.func @nested_region(%mask: i1) {
    amdg.masked_region %mask {
      // expected-error @+1 {{cannot be nested in `amdg.masked_region`}}
      amdg.masked_region %mask {
        amdg.masked_yield
      }
      amdg.masked_yield
    }
    llvm.return
  }
}

// -----

module {
  llvm.func @side_effect()

  llvm.func @unsupported_call(%mask: i1) {
    amdg.masked_region %mask {
      // expected-error @+1 {{has unsupported side effects in `amdg.masked_region`}}
      llvm.call @side_effect() : () -> ()
      amdg.masked_yield
    }
    llvm.return
  }
}

// -----

module {
  llvm.func @cluster_load_intrinsic_rejected(%mask: i1, %src: !llvm.ptr<1>) {
    %cache = llvm.mlir.constant(0 : i32) : i32
    %multicast = llvm.mlir.constant(3 : i32) : i32
    amdg.masked_region %mask {
      // expected-error @+1 {{has unsupported side effects in `amdg.masked_region`}}
      %v = llvm.call_intrinsic "llvm.amdgcn.cluster.load.b32"(%src, %cache, %multicast) : (!llvm.ptr<1>, i32, i32) -> i32
      amdg.masked_yield
    }
    llvm.return
  }
}

// -----

module {
  llvm.func @unsupported_nested_region_op(%mask: i1) {
    amdg.masked_region %mask {
      // expected-error @+1 {{with nested regions is not supported in `amdg.masked_region`}}
      scf.execute_region {
        scf.yield
      }
      amdg.masked_yield
    }
    llvm.return
  }
}

// -----

module {
  llvm.func @atomic_load_rejected(%mask: i1, %src: !llvm.ptr) {
    amdg.masked_region %mask {
      // expected-error @+1 {{is not supported in `amdg.masked_region` because it is atomic}}
      %v = llvm.load %src atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
      amdg.masked_yield
    }
    llvm.return
  }
}

// -----

module {
  llvm.func @atomic_store_rejected(%mask: i1, %dst: !llvm.ptr, %value: i32) {
    amdg.masked_region %mask {
      // expected-error @+1 {{is not supported in `amdg.masked_region` because it is atomic}}
      llvm.store %value, %dst atomic monotonic {alignment = 4 : i64} : i32, !llvm.ptr
      amdg.masked_yield
    }
    llvm.return
  }
}
