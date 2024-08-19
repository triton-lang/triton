// RUN: triton-opt --convert-builtin-func-to-llvm %s | FileCheck %s

// Trying to merge those blocks will cause a lot of duplication in the block arguments, which will cause
// an exponential growth of the argument length. Make sure we don't try to merge those blocks.
// CHECK-COUNT-7: ^bb{{[0-9]+}}:
module {
  llvm.func @rand() -> i1
  llvm.func @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(!llvm.ptr<1>, i32, i1) attributes {libname = "", libpath = ""}

  llvm.func @top(%arg0: i64, %1 : !llvm.ptr<1>, %2 : !llvm.ptr<1>, %3 : !llvm.ptr<1>, %4 : !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.icmp "eq" %arg0, %0 : i64
    %true = llvm.mlir.constant(1 : i1) : i1
    %c = llvm.mlir.constant(1 : i32) : i32
    llvm.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.call @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(%1, %c, %true) : (!llvm.ptr<1>, i32, i1) -> ()
    llvm.br ^bb3

  ^bb2:  // pred: ^bb0
    llvm.call @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(%4, %c, %true) : (!llvm.ptr<1>, i32, i1) -> ()
    llvm.br ^bb3

  ^bb3:  // 2 preds: ^bb1, ^bb2
    llvm.return
  }
}
