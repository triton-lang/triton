// RUN: triton-opt --convert-builtin-func-to-llvm %s | FileCheck %s

// Trying to merge those blocks will cause a lot of duplication in the block arguments, which will cause
// an exponential growth of the argument length. Make sure we don't try to merge those blocks.
// CHECK-COUNT-31: ^bb{{.*}}:
module {
  llvm.func @rand() -> i1
  llvm.func @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(!llvm.ptr<1>, i32, i1) attributes {libname = "", libpath = ""}

  llvm.func @top(%arg0: i64, %1 : !llvm.ptr<1>, %2 : !llvm.ptr<1>, %3 : !llvm.ptr<1>, %4 : !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.icmp "eq" %arg0, %0 : i64
    %true = llvm.mlir.constant(1 : i1) : i1
    %c = llvm.mlir.constant(1 : i32) : i32
    llvm.cond_br %10, ^bb1, ^bb14
  ^bb1:  // pred: ^bb0
    %11 = llvm.call @rand() : () -> i1
    llvm.cond_br %11, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(%1, %c, %true) : (!llvm.ptr<1>, i32, i1) -> ()
    llvm.br ^bb4
  ^bb3:  // pred: ^bb1
    llvm.call @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(%2, %c, %true) : (!llvm.ptr<1>, i32, i1) -> ()
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %14 = llvm.call @rand() : () -> i1
    llvm.cond_br %14, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(%3, %c, %true) : (!llvm.ptr<1>, i32, i1) -> ()
    llvm.br ^bb13
  ^bb6:  // pred: ^bb4
    llvm.call @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(%4, %c, %true) : (!llvm.ptr<1>, i32, i1) -> ()
    llvm.br ^bb13
  ^bb13:  // 2 preds: ^bb11, ^bb12
    llvm.br ^bb27
  ^bb14:  // pred: ^bb0
    %23 = llvm.call @rand() : () -> i1
    llvm.cond_br %23, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    llvm.call @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(%4, %c, %true) : (!llvm.ptr<1>, i32, i1) -> ()
    llvm.br ^bb17
  ^bb16:  // pred: ^bb14
    llvm.call @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(%3, %c, %true) : (!llvm.ptr<1>, i32, i1) -> ()
    llvm.br ^bb17
  ^bb17:  // 2 preds: ^bb15, ^bb16
    %26 = llvm.call @rand() : () -> i1
    llvm.cond_br %26, ^bb18, ^bb19
  ^bb18:  // pred: ^bb17
    llvm.call @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(%2, %c, %true) : (!llvm.ptr<1>, i32, i1) -> ()
    llvm.br ^bb26
  ^bb19:  // pred: ^bb17
    llvm.call @"__predicated_store_!llvm.void_!llvm.ptr<1>_i32_i1_"(%1, %c, %true) : (!llvm.ptr<1>, i32, i1) -> ()
    llvm.br ^bb26
  ^bb26:  // 2 preds: ^bb24, ^bb25
    llvm.br ^bb27
  ^bb27:  // 2 preds: ^bb13, ^bb26
    llvm.return
  }
}
