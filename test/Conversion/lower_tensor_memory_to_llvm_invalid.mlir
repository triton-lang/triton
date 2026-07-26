// RUN: triton-opt %s -split-input-file --convert-nv-gpu-to-llvm -allow-unregistered-dialect -verify-diagnostics

module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 4 : i32, ttg.tensor_memory_size = 0 : i32} {
  // expected-error @below {{cannot initialize tensor memory for a zero-sized allocation}}
  llvm.func @tmem_zero_size_kernel() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    %base = nvg.tensor_memory_base
    %base_i32 = llvm.ptrtoint %base : !llvm.ptr<6> to i32
    "use"(%base_i32) : (i32) -> ()
    llvm.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 4 : i32, ttg.tensor_memory_size = 128 : i32} {
  // expected-error @below {{cannot pass tensor memory base to a variadic device function}}
  llvm.func internal @tmem_vararg_helper(%value: i32, ...) attributes {passthrough = ["noinline"], sym_visibility = "private"} {
    %base = nvg.tensor_memory_base
    %base_i32 = llvm.ptrtoint %base : !llvm.ptr<6> to i32
    "use"(%base_i32) : (i32) -> ()
    llvm.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 4 : i32, ttg.tensor_memory_size = 128 : i32} {
  llvm.func internal @tmem_addressed_helper() attributes {passthrough = ["noinline"], sym_visibility = "private"} {
    %base = nvg.tensor_memory_base
    %base_i32 = llvm.ptrtoint %base : !llvm.ptr<6> to i32
    "use"(%base_i32) : (i32) -> ()
    llvm.return
  }

  llvm.func @tmem_addressof_kernel() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    // expected-error @below {{cannot take the address of a function that uses tensor memory}}
    %fn = llvm.mlir.addressof @tmem_addressed_helper : !llvm.ptr
    "use"(%fn) : (!llvm.ptr) -> ()
    llvm.return
  }
}
