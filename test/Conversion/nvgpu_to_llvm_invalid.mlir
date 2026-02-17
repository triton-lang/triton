// RUN: triton-opt %s --convert-nv-gpu-to-llvm -allow-unregistered-dialect -verify-diagnostics

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @helper(%shared: !llvm.ptr<3>, %gmem: !llvm.ptr<1>) -> i32 {
    // expected-error@+1 {{tensor memory base in non-kernel functions is not supported; TMEM requires inlined usage within a kernel function}}
    %0 = nvg.tensor_memory_base
    %1 = llvm.ptrtoint %0 : !llvm.ptr<6> to i32
    llvm.return %1 : i32
  }

  llvm.func @kernel() -> i32 attributes {nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    %0 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %1 = llvm.mlir.null : !llvm.ptr<1>
    %2 = llvm.call @helper(%0, %1) : (!llvm.ptr<3>, !llvm.ptr<1>) -> i32
    llvm.return %2 : i32
  }
}
