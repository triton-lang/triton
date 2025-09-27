// RUN: triton-opt %s -o - --mlir-print-debuginfo --mlir-use-nameloc-as-prefix --enable-line-info --extract-variable-info | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32 } {
  llvm.func @add_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>,
                        %arg2: !llvm.ptr<1>, %arg3: i32, %arg4: !llvm.ptr<1>) {
    %constant_i32 = llvm.mlir.constant(3 : index) : i32

    // CHECK: %pid = rocdl.workgroup.id.x
    // CHECK-NEXT: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %pid :
    %pid = rocdl.workgroup.id.x : i32 loc(#loc14)

    // CHECK: %block_start = llvm.mul %pid
    // CHECK-NEXT: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %block_start :
    %block_start = llvm.mul %pid, %constant_i32 : i32 loc(#loc15)

    // CHECK: %offsets = llvm.add %block_start
    // CHECK-NEXT: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %offsets :
    %offsets = llvm.add %block_start, %constant_i32 : i32 loc(#loc16)
    %mask = llvm.icmp "slt" %offsets, %arg3 : i32 loc(#loc17)

    llvm.return
  }
}
#loc2 = loc("01-vector-add.py":39:10)
#loc3 = loc("01-vector-add.py":44:18)
#loc5 = loc("01-vector-add.py":45:14)
#loc6 = loc("01-vector-add.py":47:11)
#loc14 = loc("pid"(#loc2))
#loc15 = loc("block_start"(#loc3))
#loc16 = loc("offsets"(#loc5))
#loc17 = loc("mask"(#loc6))
